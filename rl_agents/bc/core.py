import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.distributions as D

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

class GaussianPolicyHead(nn.Module):
    def __init__(self, latent_dim, act_dim, act_limit):
        super().__init__()
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.mu_layer = nn.Linear(latent_dim, act_dim)
        self.log_std_layer = nn.Linear(latent_dim, act_dim)
        self.epsilon = 1e-6  # Small value to avoid inf

    def forward(self, x, deterministic=False, with_logprob=True, actions=None):
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

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
            unsquashed_actions = torch.atanh(clipped_actions)  # reverse the squashing
            logp_pi = pi_distribution.log_prob(unsquashed_actions).sum(axis=-1)
            logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - unsquashed_actions - F.softplus(-2 * unsquashed_actions))).sum(axis=-1)
            return actions, logp_pi

class GMMPolicyHead(nn.Module):
    def __init__(self, latent_dim, act_dim, num_components):
        super().__init__()
        self.num_components = num_components
        self.act_dim = act_dim
        self.mu_layer = nn.Linear(latent_dim, act_dim * num_components)
        self.log_std_layer = nn.Linear(latent_dim, act_dim * num_components)
        self.logits_layer = nn.Linear(latent_dim, num_components)
        self.epsilon = 1e-6  # Small value to avoid inf
    
    def forward(self, x, deterministic=False, with_logprob=True, actions=None, act_limit=1.0):
        """x: (batch_size, latent_dim)"""
        mu = self.mu_layer(x).view(-1, self.num_components, self.act_dim) # (batch_size, num_components, act_dim)
        log_std = self.log_std_layer(x).view(-1, self.num_components, self.act_dim) # (batch_size, num_components, act_dim)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        logits = self.logits_layer(x)
        
        compo = Normal(loc=mu, scale=std)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(mixture_distribution=mix, component_distribution=compo)

        if actions is None:
            if deterministic:
                pi_action = gmm.mean
            else:
                pi_action = gmm.sample()
            
            if with_logprob:
                logp_pi = gmm.log_prob(pi_action)
                logp_pi = logp_pi - (2 * (torch.log(torch.tensor(2.0)) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
            else:
                logp_pi = None

            pi_action = torch.tanh(pi_action)
            pi_action = act_limit * pi_action

            return pi_action, logp_pi
        else:
            clipped_actions = torch.clamp(actions / act_limit, -1 + self.epsilon, 1 - self.epsilon)
            unsquashed_actions = torch.atanh(clipped_actions)
            logp_pi = gmm.log_prob(unsquashed_actions)
            logp_pi = logp_pi - (2 * (torch.log(torch.tensor(2.0)) - unsquashed_actions - F.softplus(-2 * unsquashed_actions))).sum(axis=-1)
            return actions, logp_pi

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
    def __init__(self, state_dim, goal_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit, n_head=8, num_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]
        
        self.obs_embedding = nn.Linear(state_dim + 2 * goal_dim, self.latent_dim)
        self.obstacle_embedding = nn.Linear(obstacle_dim, self.latent_dim)
        self.n_head = n_head
        self.num_layers = num_layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        
        self.gaussian_policy_head = GaussianPolicyHead(self.latent_dim, act_dim, act_limit)
        self.act_limit = act_limit

    def forward(self, observations, obstacles, actions=None, deterministic=False, with_logprob=True):
        """
        observations: (batch_size, state_dim + 2 * goal_dim) or (state_dim + 2 * goal_dim)
        obstacles: (batch_size, obstacle_dim + 1, obstacles_num) or (obstacle_dim + 1, obstacles_num)
        actions: (batch_size, act_dim) - actual actions for computing log_prob
        """
        if len(observations.shape) == 1:
            squeeze = True
            observations = observations.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
            if actions is not None:
                actions = actions.unsqueeze(0)
        else:
            squeeze = False
        device = observations.device
        obs_embedded = self.obs_embedding(observations).unsqueeze(0)  # (1, batch_size, latent_dim)

        mask = obstacles[:, -1, :]  # (batch_size, obstacles_num)
        obstacles_data = obstacles[:, :-1, :]  # (batch_size, obstacle_dim, obstacles_num)

        obstacles_embedded = self.obstacle_embedding(obstacles_data.permute(2, 0, 1))  # (obstacles_num, batch_size, latent_dim)

        # Combine the embedded observation and obstacles
        combined_input = torch.cat((obs_embedded, obstacles_embedded), dim=0)  # (1 + obstacles_num, batch_size, latent_dim)

        # Generate attention mask for Transformer
        attention_mask = torch.cat((torch.zeros(mask.size(0), 1, dtype=torch.bool, device=device), mask == 0), dim=1)  # (batch_size, 1 + obstacles_num)
        
        transformer_output = self.transformer_encoder(combined_input, src_key_padding_mask=attention_mask)  # (1 + obstacles_num, batch_size, latent_dim)
        transformer_output = transformer_output[0, :, :]  # 取第一个token的输出，代表全局信息 (batch_size, latent_dim)

        pi_action, logp_pi = self.gaussian_policy_head(transformer_output, deterministic, with_logprob, actions)
        
        if squeeze:
            pi_action = pi_action.squeeze()
            if logp_pi is not None:
                logp_pi = logp_pi.squeeze()
        
        return pi_action, logp_pi

    def compute_bc_loss(self, obs, obstacles, actions):
        _, logp_pi = self.forward(obs, obstacles, actions)
        bc_loss = -logp_pi.mean()
        return bc_loss
    
    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, obstacles, deterministic=deterministic)
        return a.cpu().numpy()
         
class SquashedGaussianAttentionActor(nn.Module):
    def __init__(self, state_dim, goal_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit, pooling_type="average"):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]
        self.pooling_type = pooling_type
        self.goal_reaching_net = mlp([state_dim + 2 * goal_dim] + list(hidden_sizes), activation, activation)
        qkv_input_dim = state_dim + 2 * goal_dim + obstacle_dim
        
        self.q_proj = nn.Linear(qkv_input_dim, self.latent_dim)
        self.k_proj = nn.Linear(qkv_input_dim, self.latent_dim)
        self.v_proj = nn.Linear(qkv_input_dim, self.latent_dim)
        
        self.gaussian_policy_head = GaussianPolicyHead(self.latent_dim + hidden_sizes[-1], act_dim, act_limit)
        self.act_limit = act_limit

    def forward(self, obs, obstacles, actions=None, deterministic=False, with_logprob=True):
        """
        obs: (batch_size, state_dim + 2 * goal_dim) or (state_dim + 2 * goal_dim)
        obstacles: (batch_size, obstacle_dim + 1, obstacles_num) or (obstacle_dim + 1, obstacles_num)
        """
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
            if actions is not None:
                actions = actions.unsqueeze(0)
        else:
            squeeze = False

        obs_replicated = obs.unsqueeze(-1).repeat(1, 1, self.obstacle_num)  # (batch_size, state_dim + 2 * goal_dim, obstacles_num)

        obstacles_data = obstacles[:, :self.obstacle_dim, :] # (batch_size, obstacle_dim, obstacles_num)
        combined_obs = torch.cat((obs_replicated, obstacles_data), dim=1)  # (batch_size, state_dim + 2 * goal_dim + obstacle_dim, obstacles_num)

        query = self.q_proj(combined_obs.permute(0, 2, 1))  # (batch_size, obstacles_num, latent_dim)
        key = self.k_proj(combined_obs.permute(0, 2, 1)) 
        value = self.v_proj(combined_obs.permute(0, 2, 1)) 

        mask = obstacles[:, -1, :].unsqueeze(1)  # (batch_size, 1, obstacles_num)

        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        scores = scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32)) # (batch_size, obstacles_num, obstacles_num)

        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value) # (batch_size, obstacles_num, latent_dim)
        if self.pooling_type == "max":
            pooling_output = self.max_pooling(attn_output, mask)
        else:
            pooling_output = self.average_pooling(attn_output, mask) 
        
        goal_reaching_out = self.goal_reaching_net(obs) 
        combined_out = torch.cat((goal_reaching_out, pooling_output), dim=-1) 

        pi_action, logp_pi = self.gaussian_policy_head(combined_out, deterministic, with_logprob, actions)
        
        if squeeze:
            pi_action = pi_action.squeeze()
            if logp_pi is not None:
                logp_pi = logp_pi.squeeze()
        
        return pi_action, logp_pi

    def compute_bc_loss(self, obs, obstacles, actions):
        _, logp_pi = self.forward(obs, obstacles, actions)
        bc_loss = -logp_pi.mean()
        return bc_loss
    
    def average_pooling(self, output, mask):
        """
        Fullfill average pooling for the output of transformer encoder
        output: (batch_size, obstacles_num, latent_dim), mask: (batch_size, 1, obstacles_num)
        """
        mask = mask.squeeze()  # (batch_size, obstacles_num)
        mask_expanded = mask.unsqueeze(-1).expand_as(output)  # (batch_size, obstacles_num, latent_dim)
        output_mask = output * mask_expanded  # (batch_size, obstacles_num, latent_dim)
        valid_counts = mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        output_sum = output_mask.sum(dim=1)  # (batch_size, latent_dim)
        output_avg = output_sum / valid_counts  # (batch_size, latent_dim)
        output_avg[valid_counts.squeeze() == 0] = 0
        return output_avg
    
    def max_pooling(self, output, mask):
        """
        Fullfill max pooling for the output of transformer encoder
        output: (batch_size, obstacles_num, latent_dim), mask: (batch_size, 1, obstacles_num)
        """
        mask = mask.squeeze()  # (batch_size, obstacles_num)
        mask_expanded = mask.unsqueeze(-1).expand_as(output)  # (batch_size, obstacles_num, latent_dim)
        small_value = -1e9
        output_mask = output * mask_expanded + small_value * (1 - mask_expanded)
        valid_counts = mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        output_max = output_mask.max(dim=1)[0]
        output_max[valid_counts.squeeze() == 0] = 0
        return output_max
    
    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, obstacles, deterministic=deterministic)
        return a.cpu().numpy()
    
class SquashedGaussianMixtureTransformerActorVersion1(nn.Module):
    def __init__(self, state_dim, goal_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit, num_components=5, num_layers=2, n_heads=8):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]
        self.act_limit = act_limit
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.obs_embedding = nn.Linear(state_dim + 2 * goal_dim, self.latent_dim)
        self.obstacle_embedding = nn.Linear(obstacle_dim, self.latent_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        
        self.gmm_policy_head = GMMPolicyHead(self.latent_dim, act_dim, num_components)

    def forward(self, observations, obstacles, actions=None, deterministic=False, with_logprob=True):
        if len(observations.shape) == 1:
            squeeze = True
            observations = observations.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
            if actions is not None:
                actions = actions.unsqueeze(0)
        else:
            squeeze = False

        device = observations.device
        observations_embedded = self.obs_embedding(observations).unsqueeze(0)  # (1, batch_size, latent_dim)

        mask = obstacles[:, -1, :]  # (batch_size, obstacles_num)
        obstacles_data = obstacles[:, :-1, :]  # (batch_size, obstacle_dim, obstacles_num)

        obstacles_embedded = self.obstacle_embedding(obstacles_data.permute(2, 0, 1))  # (obstacles_num, batch_size, latent_dim)

        combined_input = torch.cat((observations_embedded, obstacles_embedded), dim=0)  # (1 + obstacles_num, batch_size, latent_dim)

        attention_mask = torch.cat((torch.zeros(mask.size(0), 1, dtype=torch.bool, device=device), mask == 0), dim=1)  # (batch_size, 1 + obstacles_num)
        
        transformer_output = self.transformer_encoder(combined_input, src_key_padding_mask=attention_mask)  # (1 + obstacles_num, batch_size, latent_dim)
        transformer_output = transformer_output[0, :, :]  # 取第一个token的输出，代表全局信息 (batch_size, latent_dim)

        pi_action, logp_pi = self.gmm_policy_head(transformer_output, deterministic, with_logprob, actions, self.act_limit)
        
        if squeeze:
            pi_action = pi_action.squeeze()
            if logp_pi is not None:
                logp_pi = logp_pi.squeeze()
        
        return pi_action, logp_pi

    def compute_bc_loss(self, obs, obstacles, actions):
        _, logp_pi = self.forward(obs, obstacles, actions)
        bc_loss = -logp_pi.mean()
        return bc_loss
    
    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, obstacles, deterministic=deterministic)
        return a.cpu().numpy()   

class SquashedGaussianMixtureTransformerActorVersion2(nn.Module):
    def __init__(self, state_dim, goal_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit, num_components=5, pooling_type="average", num_layers=2, n_heads=8):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]
        self.num_components = num_components
        self.act_dim = act_dim 
        self.pooling_type = pooling_type
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.goal_reaching_net = mlp([state_dim + 2 * goal_dim] + list(hidden_sizes), activation, activation)
        self.observation_with_obstacle_embedding = nn.Linear(state_dim + 2 * goal_dim + obstacle_dim, self.latent_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        
        self.gmm_policy_head = GMMPolicyHead(2 * self.latent_dim, act_dim, num_components)
        self.act_limit = act_limit

    def forward(self, observations, obstacles, actions=None, deterministic=False, with_logprob=True):
        """
        observations: vehicle state and goal position (batch_size, state_dim + 2 * goal_dim) or (state_dim + 2 * goal_dim,)
        obstacles: obstacle features with masking (batch_size, obstacle_dim + 1, obstacles_num) or (obstacle_dim + 1, obstacles_num)
        actions: if None: output actions, else: compute log_prob for given actions (batch_size, act_dim)
        """
        if len(observations.shape) == 1:
            squeeze = True
            observations = observations.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
            if actions is not None:
                actions = actions.unsqueeze(0)
        else:
            squeeze = False
        device = observations.device
        observations_replicated = observations.unsqueeze(-1).repeat(1, 1, self.obstacle_num)  # (batch_size, state_dim + 2 * goal_dim, obstacles_num)
        
        
        # Extract mask and obstacle data
        mask = obstacles[:, -1, :]  # (batch_size, obstacles_num)
        obstacles_data = obstacles[:, :-1, :]  # (batch_size, obstacle_dim, obstacles_num)
        observations_with_obstacles = torch.cat((observations_replicated, obstacles_data), dim=1)  # (batch_size, state_dim + 2 * goal_dim + obstacle_dim, obstacles_num)
        
        # Obtain the embedding for first combining observations and obstacles
        observations_with_obstacles_embedded = self.observation_with_obstacle_embedding(observations_with_obstacles.permute(2, 0, 1))  # (obstacles_num, batch_size, latent_dim)

        # Generate attention mask for Transformer
        attention_mask = (mask == 0)  # (batch_size, obstacles_num)
        
        transformer_output = self.transformer_encoder(observations_with_obstacles_embedded, src_key_padding_mask=attention_mask)  # (obstacles_num, batch_size, latent_dim)
        if self.pooling_type == "max":
            pooling_output = self.max_pooling(transformer_output, mask)  # (batch_size, latent_dim)
        else: 
            pooling_output = self.average_pooling(transformer_output, mask)  # (batch_size, latent_dim)
        
        goal_reaching_out = self.goal_reaching_net(observations)
        combined_out = torch.cat((goal_reaching_out, pooling_output), dim=-1)
        
        pi_action, logp_pi = self.gmm_policy_head(combined_out, deterministic, with_logprob, actions, self.act_limit)
        
        if squeeze:
            pi_action = pi_action.squeeze()
            if logp_pi is not None:
                logp_pi = logp_pi.squeeze()
        
        return pi_action, logp_pi

    def average_pooling(self, output, mask):
        """
        Fullfill average pooling for the output of transformer encoder
        output: (obstacles_num, batch_size, latent_dim), mask: (batch_size, obstacles_num)
        """
        output = output.permute(1, 0, 2)  # (batch_size, obstacles_num, latent_dim)
        mask_expanded = mask.unsqueeze(-1).expand_as(output)  # (batch_size, obstacles_num, latent_dim)
        output_mask = output * mask_expanded  # (batch_size, obstacles_num, latent_dim)
        valid_counts = mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        output_sum = output_mask.sum(dim=1)  # (batch_size, latent_dim)
        output_avg = output_sum / valid_counts  # (batch_size, latent_dim)
        output_avg[valid_counts.squeeze() == 0] = 0
        return output_avg
    
    def max_pooling(self, output, mask):
        """
        Fullfill max pooling for the output of transformer encoder
        output: (obstacles_num, batch_size, latent_dim), mask: (batch_size, obstacles_num)
        """
        output = output.permute(1, 0, 2)  # (batch_size, obstacles_num, latent_dim)
        mask_expanded = mask.unsqueeze(-1).expand_as(output)
        small_value = -1e9
        output_mask = output * mask_expanded + small_value * (1 - mask_expanded)
        valid_counts = mask.sum(dim=1, keepdim=True).float()
        output_max = output_mask.max(dim=1)[0]
        output_max[valid_counts.squeeze() == 0] = 0
        return output_max

    def compute_bc_loss(self, obs, obstacles, actions):
        _, logp_pi = self.forward(obs, obstacles, actions)
        bc_loss = -logp_pi.mean()
        return bc_loss
    
    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, obstacles, deterministic=deterministic)
        return a.cpu().numpy()