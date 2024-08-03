import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# 设置随机种子
torch.manual_seed(0)

# 生成数据集
pi_k = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])  # 每个混合分量的权重
mu_k = torch.tensor([[-2.0, 1.0], [-1.0, 1.5], [0.0, 2.0], [1.0, 2.5], [2.0, 3.0]])  # 每个混合分量的二维均值
sigma_k = torch.tensor([[0.5, 0.3], [0.5, 0.3], [0.5, 0.3], [0.5, 0.3], [0.5, 0.3]])  # 每个混合分量的二维标准差

mix = D.Categorical(probs=pi_k)
comp = D.Normal(loc=mu_k, scale=sigma_k)
gmm = D.MixtureSameFamily(mixture_distribution=mix, component_distribution=D.Independent(comp, 1))

num_samples = 10000
x = gmm.sample((num_samples,))
y = torch.tanh(x)

# 数据集
dataset = y  # 仅使用 y 作为训练数据

# 划分数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleGMM(nn.Module):
    def __init__(self, num_components, act_dim):
        super(SimpleGMM, self).__init__()
        self.num_components = num_components
        self.act_dim = act_dim
        
        # 可学习的参数
        self.logits = nn.Parameter(torch.randn(num_components))
        self.mu = nn.Parameter(torch.randn(num_components, act_dim))
        self.log_std = nn.Parameter(torch.randn(num_components, act_dim))
        self.epsilon = 1e-6
        self.act_limit = 1.0
        
    def forward(self, actions):
        std = torch.exp(self.log_std)
        comp = D.Normal(loc=self.mu, scale=std)
        comp = D.Independent(comp, 1)
        mix = D.Categorical(logits=self.logits)
        gmm = D.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
        
        # Clip actions to avoid inf values in atanh
        clipped_actions = torch.clamp(actions / self.act_limit, -1 + self.epsilon, 1 - self.epsilon)
        unsquashed_actions = torch.atanh(clipped_actions)  # reverse the squashing
        log_prob = gmm.log_prob(unsquashed_actions)
        correction = (2 * (torch.log(torch.tensor(2.0)) - unsquashed_actions - F.softplus(-2 * unsquashed_actions))).sum(axis=-1)
        log_prob = log_prob - correction
        return log_prob

    def compute_bc_loss(self, actions):
        log_prob = self.forward(actions=actions)
        bc_loss = -log_prob.mean()
        return bc_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleGMM(num_components=5, act_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        actions = batch.to(device)  # 使用y作为训练数据
        optimizer.zero_grad()
        loss = model.compute_bc_loss(actions)
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            actions = batch.to(device)  # 使用y作为测试数据
            test_loss += model.compute_bc_loss(actions).item()
    test_loss /= len(test_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss}")

# 打印学习到的参数值
print("Learned logits:", model.logits.data)
print("Learned mu:", model.mu.data)
print("Learned log_std:", model.log_std.data)
print("Learned std:", torch.exp(model.log_std.data))