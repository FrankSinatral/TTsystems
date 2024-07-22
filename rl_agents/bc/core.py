import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# For ResNet agent
import random
from operator import add
import collections
import math
import itertools

class CustomFlatten(nn.Module):
    def forward(self, x):
        if x.dim() == 4:  # x: (batch_size, channels, height, width)
            return x.view(x.size(0), -1)  # remain batch_size dim, flatten the rest
        elif x.dim() == 3:  # x: (channels, height, width)
            return x.view(-1)  # flatten all dims
        else:
            raise ValueError("Unsupported tensor shape.")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def cnn(layers, activation=nn.ReLU, output_activation=nn.Identity):
    network_layers = []
    network_layers_type = []
    for layer in layers:
        if isinstance(layer, tuple):
            network_layers_type.append(1)
        else:
            network_layers_type.append(0)
            
    # Check if network_layers_type starts with consecutive 1s
    if not all(x == 1 for x in itertools.takewhile(lambda x: x == 1, network_layers_type)):
        raise ValueError("Invalid CNN structure. The layers should start with tuple(s).")
    
    loop_range = len(layers) if all(x == 1 for x in network_layers_type) else len(layers) - 1
    
    # this is different from mlp
    for i in range(loop_range):
        if isinstance(layers[i], tuple):  # Convolutional layer
            in_channels, out_channels, kernel_size, stride = layers[i]
            network_layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride), activation()]
        elif isinstance(layers[i], int) and isinstance(layers[i+1], int):  # Linear layer
            if i == 0 or isinstance(layers[i-1], tuple):  # Check if previous layer was Convolutional
                network_layers += [CustomFlatten()]  # Add Custom Flatten layer
            network_layers += [nn.Linear(layers[i], layers[i+1])]
            if i < len(layers) - 2:  # Not the last layer
                network_layers += [activation()]
            else:  # Last layer
                network_layers += [output_activation()]
    return nn.Sequential(*network_layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, actions=None, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        
        if actions is None:
            if deterministic:
                # Only used for evaluating policy at test time.
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()

            if with_logprob:
                # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
                # NOTE: The correction formula is a little bit magic. To get an understanding 
                # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
                # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
                # Try deriving it yourself as a (very difficult) exercise. :)
                logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
                logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
            else:
                logp_pi = None

            pi_action = torch.tanh(pi_action)
            pi_action = self.act_limit * pi_action

            return pi_action, logp_pi
        else:
            # Compute log_prob for given actions
            unsquashed_actions = torch.atanh(actions / self.act_limit)
            logp_pi = pi_distribution.log_prob(unsquashed_actions).sum(axis=-1)
            logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - actions - F.softplus(-2 * actions))).sum(axis=-1)
            return actions, logp_pi

    def compute_bc_loss(self, obs, actions):
        _, logp_pi = self.forward(obs, actions=actions)
        bc_loss = -logp_pi.mean()
        return bc_loss
    
    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, obstacles, deterministic=deterministic)
        return a.cpu().numpy()

class SquashedGaussianTransformerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]
        
        self.obs_embedding = nn.Linear(state_dim + 2 * goal_dim, self.latent_dim)
        self.obstacle_embedding = nn.Linear(obstacle_dim, self.latent_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        
        self.mu_layer = nn.Linear(self.latent_dim, act_dim)
        self.log_std_layer = nn.Linear(self.latent_dim, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, obstacles, actions=None, deterministic=False, with_logprob=True):
        """
        obs: (batch_size, state_dim + 2 * goal_dim) or (state_dim + 2 * goal_dim)
        obstacles: (batch_size, obstacle_dim + 1, obstacles_num) or (obstacle_dim + 1, obstacles_num)
        actions: (batch_size, act_dim) - actual actions for computing log_prob
        """
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
            if actions is not None:
                actions = actions.unsqueeze(0)
        else:
            squeeze = False
        device = obs.device
        obs_embedded = self.obs_embedding(obs).unsqueeze(0)  # (1, batch_size, latent_dim)

        mask = obstacles[:, -1, :].squeeze(1)  # (batch_size, obstacles_num)
        obstacles_data = obstacles[:, :-1, :]  # (batch_size, obstacle_dim, obstacles_num)

        obstacles_embedded = self.obstacle_embedding(obstacles_data.permute(2, 0, 1))  # (obstacles_num, batch_size, latent_dim)

        # Combine the embedded observation and obstacles
        combined_input = torch.cat((obs_embedded, obstacles_embedded), dim=0)  # (1 + obstacles_num, batch_size, latent_dim)

        # Generate attention mask for Transformer
        attention_mask = torch.cat((torch.zeros(mask.size(0), 1, dtype=torch.bool, device=device), mask == 0), dim=1)  # (batch_size, 1 + obstacles_num)
        
        transformer_output = self.transformer_encoder(combined_input, src_key_padding_mask=attention_mask)  # (1 + obstacles_num, batch_size, latent_dim)
        transformer_output = transformer_output[0, :, :]  # 取第一个token的输出，代表全局信息 (batch_size, latent_dim)

        mu = self.mu_layer(transformer_output)
        log_std = self.log_std_layer(transformer_output)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if squeeze:
            mu = mu.squeeze()
            std = std.squeeze()

        pi_distribution = Normal(mu, std)
        if actions is None:
            if deterministic:
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()

            if with_logprob:
                logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
                logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
            else:
                logp_pi = None

            pi_action = torch.tanh(pi_action)
            pi_action = self.act_limit * pi_action

            return pi_action, logp_pi
        else:
            # Clip actions to avoid inf values in atanh
            clipped_actions = torch.clamp(actions / self.act_limit, -1 + self.epsilon, 1 - self.epsilon)
            unsquashed_actions = torch.atanh(clipped_actions) # reverse the squashing
            logp_pi = pi_distribution.log_prob(unsquashed_actions).sum(axis=-1)
            logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - actions - F.softplus(-2 * actions))).sum(axis=-1)
            return actions, logp_pi

    def compute_bc_loss(self, obs, obstacles, actions):
        _, logp_pi = self.forward(obs, obstacles, actions)
        bc_loss = -logp_pi.mean()
        return bc_loss
    
    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, obstacles, deterministic=deterministic)
        return a.cpu().numpy()
         
class SquashedGaussianAttentionActor(nn.Module):
    def __init__(self, state_dim, goal_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]

        self.goal_reaching_net = mlp([state_dim + 2 * goal_dim] + list(hidden_sizes), activation, activation)
        qkv_input_dim = state_dim + 2 * goal_dim + obstacle_dim
        qkv_hidden_sizes = [qkv_input_dim] + list(hidden_sizes)
        
        self.q_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation)
        self.k_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation)
        self.v_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation, nn.ReLU) 
        
        self.mu_layer = nn.Linear(self.latent_dim + hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(self.latent_dim + hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.epsilon = 1e-6  # Small value to avoid inf

    def forward(self, obs, obstacles, actions=None, deterministic=False, with_logprob=True):
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
            if actions is not None:
                actions = actions.unsqueeze(0)
        else:
            squeeze = False

        obs_replicated = obs.unsqueeze(-1).repeat(1, 1, self.obstacle_num) 

        obstacles_data = obstacles[:, :self.obstacle_dim, :] 
        combined_obs = torch.cat((obs_replicated, obstacles_data), dim=1) 

        query = self.q_proj(combined_obs.permute(0, 2, 1)) 
        key = self.k_proj(combined_obs.permute(0, 2, 1)) 
        value = self.v_proj(combined_obs.permute(0, 2, 1)) 

        mask = obstacles[:, -1, :].unsqueeze(1) 

        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        scores = scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value) 
        attn_output = attn_output * mask.permute(0, 2, 1) 
        attn_output = attn_output.sum(dim=1) 
        
        goal_reaching_out = self.goal_reaching_net(obs) 
        combined_out = torch.cat((goal_reaching_out, attn_output), dim=-1) 

        mu = self.mu_layer(combined_out)
        log_std = self.log_std_layer(combined_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if squeeze:
            mu = mu.squeeze()
            std = std.squeeze()

        pi_distribution = Normal(mu, std)
        if actions is None:
            if deterministic:
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()

            if with_logprob:
                logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
                logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
            else:
                logp_pi = None

            pi_action = torch.tanh(pi_action)
            pi_action = self.act_limit * pi_action

            return pi_action, logp_pi
        else:
            # Clip actions to avoid inf values in atanh
            clipped_actions = torch.clamp(actions / self.act_limit, -1 + self.epsilon, 1 - self.epsilon)
            unsquashed_actions = torch.atanh(clipped_actions) # reverse the squashing
            logp_pi = pi_distribution.log_prob(unsquashed_actions).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=-1)
            return actions, logp_pi

    def compute_bc_loss(self, obs, obstacles, actions):
        _, logp_pi = self.forward(obs, obstacles, actions)
        bc_loss = -logp_pi.mean()
        return bc_loss
    
    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, obstacles, deterministic=deterministic)
        return a.cpu().numpy()