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

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
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
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class SquashedGaussianCNNActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.cnn_net = cnn([(1, 32, 8, 4), (32, 64, 4, 2), (64, 128, 4, 2), (128, 256, 4, 2), 256*23*23, 512], activation, activation)
        self.net = mlp([obs_dim + 512] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, image_obs, deterministic=False, with_logprob=True):
        # this may still have to be changed
        cnn_net_out = self.cnn_net(image_obs/255.0) # Normalization
        net_out = self.net(torch.cat([obs, cnn_net_out], dim=-1))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
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
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi
    

class SquashedGaussianSetActor(nn.Module):

    def __init__(self, state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.perception_dim = perception_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.perception_net = mlp([2*state_dim + goal_dim + perception_dim] + list(hidden_sizes), activation, activation)
        self.obstacle_net = mlp([2*state_dim + goal_dim + obstacle_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(2*hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(2*hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, obstacles, deterministic=False, with_logprob=True):
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0) # [1, obs_dim]
            obstacles = obstacles.unsqueeze(0)
        else:
            squeeze = False
        perception_out = self.perception_net(obs)
        
        obs_vehicle = obs[:, :(2*self.state_dim + self.goal_dim)] # (batch_size, (2*state_dim + goal_dim))
        obs_vehicle = obs_vehicle.unsqueeze(2).expand(-1, -1, obstacles.size(2)) #(batch_size, (2*state_dim + goal_dim), obstacles_num)
        
        
        obstacles_data = obstacles[:, :self.obstacle_dim, :] # (batch_size, obstacle_dim, obstacles_num)
        obs_with_obstacles = torch.cat([obs_vehicle, obstacles_data], dim=1) # (batch_size, (2*state_dim + goal_dim + obstacle_dim), obstacles_num)
        obs_with_obstacles = obs_with_obstacles.permute(0,2,1) # (batch_size, obstacles_num, (2*state_dim + goal_dim + obstacle_dim)
        obs_with_obstacles = obs_with_obstacles.contiguous().view(-1, 2*self.state_dim + self.goal_dim + self.obstacle_dim) # (batch_size*obstacles_num, (2*state_dim + goal_dim + obstacle_dim))
        # obs_with_obstacles = obs_with_obstacles.view(-1, 2*self.state_dim + self.goal_dim + self.obstacle_dim) # （batch_size*obstacles_num, (2*state_dim + goal_dim + obstacle_dim))

        obstacle_outs = self.obstacle_net(obs_with_obstacles) # (batch_size*obstacles_num, hidden_sizes[-1])
        obstacle_outs = obstacle_outs.view(obs.size(0), obstacles.size(2), -1).permute(0, 2, 1) # (batch_size, hidden_sizes[-1], self.obstacles_num)

        
        obstacle_masks = obstacles[:, -1, :].unsqueeze(1) #(batch_size, 1, self.obstacle_num)
        
        
        weighted_obstacle_outs = obstacle_outs * obstacle_masks
        mean_obstacle_out = weighted_obstacle_outs.sum(dim=2) # (batch_size, hidden_sizes[-1])
        
        combined_out = torch.cat([perception_out, mean_obstacle_out], dim=-1)
        mu = self.mu_layer(combined_out)
        log_std = self.log_std_layer(combined_out)
        
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        if squeeze:
            mu = mu.squeeze()
            std = std.squeeze()
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.
    
class CNNQFunction(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # gray image as input: the calculation might not be correct
        self.cnn_net = cnn([(1, 32, 8, 4), (32, 64, 4, 2), (64, 128, 4, 2), (128, 256, 4, 2), 256*23*23, 512], activation, activation)
        self.q = mlp([obs_dim + act_dim + 512] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, image_obs, act):
        cnn_net_out = self.cnn_net(image_obs/255.0)
        q = self.q(torch.cat([obs, cnn_net_out, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class SetQFunction(nn.Module):

    def __init__(self, state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.perception_dim = perception_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        # Network for processing each obstacle
        self.obstacle_net = mlp([2*state_dim + goal_dim + obstacle_dim] + list(hidden_sizes), activation, activation)
        # Q-value output layer
        self.q_layer = mlp([2*state_dim + goal_dim + perception_dim + hidden_sizes[-1] + act_dim] + list(hidden_sizes) + [1], activation)
        self.hidden_size = hidden_sizes[-1]

    def forward(self, obs, obstacles, act):
        
        obs_vehicle = obs[:, :(2*self.state_dim + self.goal_dim)] # (batch_size, (2*state_dim + goal_dim))
        
        obs_vehicle = obs_vehicle.unsqueeze(2).expand(-1, -1, obstacles.size(2)) #(batch_size, (2*state_dim + goal_dim), obstacles_num)

        obstacles_data = obstacles[:, :self.obstacle_dim, :] # (batch_size, obstacle_dim, obstacles_num)
        obs_with_obstacles = torch.cat([obs_vehicle, obstacles_data], dim=1) # (batch_size, (2*state_dim + goal_dim + obstacle_dim), obstacles_num)
        obs_with_obstacles = obs_with_obstacles.permute(0,2,1) # (batch_size, obstacles_num, (2*state_dim + goal_dim + obstacle_dim)
        obs_with_obstacles = obs_with_obstacles.contiguous().view(-1, 2*self.state_dim + self.goal_dim + self.obstacle_dim) # (batch_size*obstacles_num, (2*state_dim + goal_dim + obstacle_dim))
        # obs_with_obstacles = obs_with_obstacles.view(-1, 2*self.state_dim + self.goal_dim + self.obstacle_dim) # （batch_size*obstacles_num, (2*state_dim + goal_dim + obstacle_dim))

        obstacle_outs = self.obstacle_net(obs_with_obstacles) # (batch_size*obstacles_num, hidden_sizes[-1])
        obstacle_outs = obstacle_outs.view(obs.size(0), obstacles.size(2), -1).permute(0, 2, 1) # (batch_size, hidden_sizes[-1], self.obstacles_num)

        
        obstacle_masks = obstacles[:, -1, :].unsqueeze(1) #(batch_size, 1, self.obstacle_num)
        
        
        weighted_obstacle_outs = obstacle_outs * obstacle_masks
        mean_obstacle_out = weighted_obstacle_outs.sum(dim=2) # (batch_size, hidden_sizes[-1])
        combined_out = torch.cat([obs, mean_obstacle_out, act], dim=-1)
        q = self.q_layer(combined_out)
        return torch.squeeze(q, -1)  # Ensure q has the right shape
    
# class SquashedGaussianAttentionActor(nn.Module):
#     def __init__(self, state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         self.state_dim = state_dim # this refer to the state dimension
#         self.goal_dim = goal_dim # this refer to the goal dimension(usually the same as state_dim)
#         self.perception_dim = perception_dim
#         self.obstacle_dim = obstacle_dim
#         self.obstacle_num = obstacle_num
#         self.latent_dim = hidden_sizes[-1]
        
#         self.goal_reaching_net = mlp([2*state_dim + goal_dim] + list(hidden_sizes), activation, activation)
#         self.q_proj = nn.Linear(2*state_dim + goal_dim, self.latent_dim)
#         self.k_proj = nn.Linear(obstacle_dim, self.latent_dim)
#         self.v_proj = nn.Linear(obstacle_dim, self.latent_dim)
#         self.mu_layer = nn.Linear(self.latent_dim, act_dim)
#         self.log_std_layer = nn.Linear(self.latent_dim, act_dim)
#         self.act_limit = act_limit
        
#     def forward(self, obs, obstacles, deterministic=False, with_logprob=True):
#         """the input
#         - obs: (batch_size, (2*state_dim + goal_dim)) or (2*state_dim + goal_dim)
#         - obstacles: (batch_size, obstacle_dim + 1, obstacles_num) or (obstacle_dim + 1, obstacles_num)
#         """
#         if len(obs.shape) == 1:
#             squeeze = True
#             obs = obs.unsqueeze(0)
#             obstacles = obstacles.unsqueeze(0)
#         else:
#             squeeze = False
#         # Extract the relevant part of obs
#         obs_vehicle = obs[:, :(2*self.state_dim + self.goal_dim)]  # (batch_size, (2*state_dim + goal_dim)) #TODO
        
#         goal_reaching_out = self.goal_reaching_net(obs_vehicle)  
        
#         # Project to Query, Key & Value
#         query = self.q_proj(obs_vehicle).unsqueeze(1)  # (batch_size, 1, latent_dim)
#         obstacles_data = obstacles[:, :self.obstacle_dim, :]  # (batch_size, obstacle_dim, obstacles_num)
#         key = self.k_proj(obstacles_data.permute(0, 2, 1))  # (batch_size, obstacles_num, latent_dim)
#         value = self.v_proj(obstacles_data.permute(0, 2, 1))  # (batch_size, obstacles_num, latent_dim)
        
#         # Masking
#         mask = obstacles[:, -1, :].unsqueeze(1)  # (batch_size, 1, obstacles_num)
        
#         dk = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, 1, obstacles_num) for each obstacle, there is a score
#         scores = scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        
#         # Apply mask by setting masked positions to a very large negative number
#         scores = scores.masked_fill(mask == 0, -1e9)
#         attn_weights = F.softmax(scores, dim=-1)  # (batch_size, 1, obstacles_num)
#         attn_output = torch.matmul(attn_weights, value)  # (batch_size, 1, latent_dim)
#         attn_output = attn_output.squeeze(1)  # [batch_size, latent_dim]
#         combined_out = goal_reaching_out + attn_output  # [batch_size, latent_dim]
        
#         mu = self.mu_layer(combined_out)
#         log_std = self.log_std_layer(combined_out)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)

#         if squeeze:
#             mu = mu.squeeze()
#             std = std.squeeze()

#         # Pre-squash distribution and sample
#         pi_distribution = Normal(mu, std)
#         if deterministic:
#             pi_action = mu
#         else:
#             pi_action = pi_distribution.rsample()

#         if with_logprob:
#             logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
#             logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
#         else:
#             logp_pi = None

#         pi_action = torch.tanh(pi_action)
#         pi_action = self.act_limit * pi_action

#         return pi_action, logp_pi

class SquashedGaussianAttentionActor(nn.Module):
    def __init__(self, state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.perception_dim = perception_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]

        self.goal_reaching_net = mlp([state_dim + 2 * goal_dim] + list(hidden_sizes), activation, activation)
        qkv_input_dim = state_dim + 2 * goal_dim + obstacle_dim
        qkv_hidden_sizes = [qkv_input_dim] + list(hidden_sizes)
        
        self.q_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation)
        self.k_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation)
        self.v_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation, nn.ReLU) # TODO: take out the relu
        # self.v_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation) # TODO: may need to change the layer structure
        
        self.mu_layer = nn.Linear(self.latent_dim + hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(self.latent_dim + hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, obstacles, deterministic=False, with_logprob=True):
        """
        obs: (batch_size, (state_dim + 2*goal_dim)) or (state_dim + 2*goal_dim)
        obstacles: (batch_size, obstacle_dim + 1, obstacles_num) or (obstacle_dim + 1, obstacles_num)
        TODO: may change to mixture of Gaussian
        """
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
        else:
            squeeze = False

        obs_replicated = obs.unsqueeze(-1).repeat(1, 1, self.obstacle_num) # (batch_size, (state_dim + 2*goal_dim), obstacles_num)

        obstacles_data = obstacles[:, :self.obstacle_dim, :] # (batch_size, obstacle_dim, obstacles_num)
        combined_obs = torch.cat((obs_replicated, obstacles_data), dim=1) # (batch_size, (state_dim + 2*goal_dim + obstacle_dim), obstacles_num)

        query = self.q_proj(combined_obs.permute(0, 2, 1)) # (batch_size, obstacles_num, latent_dim)
        key = self.k_proj(combined_obs.permute(0, 2, 1)) # (batch_size, obstacles_num, latent_dim)
        value = self.v_proj(combined_obs.permute(0, 2, 1)) # (batch_size, obstacles_num, latent_dim)

        mask = obstacles[:, -1, :].unsqueeze(1) # (batch_size, 1, obstacles_num)

        dk = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) # (batch_size, obstacles_num, obstacles_num)
        scores = scores / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value) # (batch_size, obstacles_num, latent_dim)
        attn_output = attn_output * mask.permute(0, 2, 1) # (batch_size, obstacles_num, latent_dim)
        # attn_output, _ = attn_output.max(dim=1) # (batch_size, latent_dim)
        attn_output = attn_output.sum(dim=1) # TODO: this way you have to take out ReLU
        # TODO: average
        goal_reaching_out = self.goal_reaching_net(obs) # (batch_size, hidden_sizes[-1])
        combined_out = torch.cat((goal_reaching_out, attn_output), dim=-1) # (batch_size, latent_dim + hidden_sizes[-1])

        mu = self.mu_layer(combined_out)
        log_std = self.log_std_layer(combined_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if squeeze:
            mu = mu.squeeze()
            std = std.squeeze()

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class AttentionQFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.perception_dim = perception_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]

        qkv_input_dim = state_dim + 2 * goal_dim + obstacle_dim
        qkv_hidden_sizes = [qkv_input_dim] + list(hidden_sizes)

        self.q_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation)
        self.k_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation)
        self.v_proj = mlp(qkv_hidden_sizes + [self.latent_dim], activation, nn.ReLU)
        
        self.q_layer = mlp([state_dim + 2 * goal_dim + hidden_sizes[-1] + act_dim] + list(hidden_sizes) + [1], activation)
        self.hidden_size = hidden_sizes[-1]

    def forward(self, obs, obstacles, act):
        """the input shape will always have batch_size dimension"""
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
            act = act.unsqueeze(0)
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

        attn_output, _ = attn_output.max(dim=1)

        combined_out = torch.cat([obs, attn_output, act], dim=-1)
        q = self.q_layer(combined_out)
        return torch.squeeze(q, -1)


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

        # Add a special embedding for the first position
        # self.first_pos_embedding = nn.Parameter(torch.zeros(1, self.latent_dim))

    def forward(self, obs, obstacles, deterministic=False, with_logprob=True):
        """
        obs: (batch_size, state_dim + 2 * goal_dim) or (state_dim + 2 * goal_dim)
        obstacles: (batch_size, obstacle_dim + 1, obstacles_num) or (obstacle_dim + 1, obstacles_num)
        """
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
        else:
            squeeze = False
        device = obs.device
        obs_embedded = self.obs_embedding(obs).unsqueeze(0)  # (1, batch_size, latent_dim)

        # Add the first position embedding to the observation embedding
        # obs_embedded = obs_embedded # + self.first_pos_embedding.unsqueeze(1)  # (1, batch_size, latent_dim)

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
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        if squeeze:
            mu = mu.squeeze()
            std = std.squeeze()

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class TransformerQFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.latent_dim = hidden_sizes[-1]

        self.obs_with_act_embedding = nn.Linear(state_dim + 2 * goal_dim + act_dim, self.latent_dim)
        self.obstacle_embedding = nn.Linear(obstacle_dim, self.latent_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        self.q_layer = nn.Linear(self.latent_dim, 1)

        # Add a special embedding for the first position
        # self.first_pos_embedding = nn.Parameter(torch.zeros(1, self.latent_dim))

    def forward(self, obs, obstacles, act):
        """
        obs: (batch_size, state_dim + 2 * goal_dim) or (state_dim + 2 * goal_dim)
        obstacles: (batch_size, obstacle_dim + 1, obstacles_num) or (obstacle_dim + 1, obstacles_num)
        act: (batch_size, act_dim) or (act_dim)
        
        # TODO: act with obs
        """
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
            act = act.unsqueeze(0)
        else:
            squeeze = False
        device = obs.device
        obs_with_act = torch.cat([obs, act], dim=-1)
        obs_embedded = self.obs_with_act_embedding(obs_with_act).unsqueeze(0)  # (1, batch_size, latent_dim)

        # Add the first position embedding to the observation embedding
        # obs_embedded = obs_embedded + self.first_pos_embedding.unsqueeze(1)  # (1, batch_size, latent_dim)

        mask = obstacles[:, -1, :].squeeze(1)  # (batch_size, obstacles_num)
        obstacles_data = obstacles[:, :-1, :]  # (batch_size, obstacle_dim, obstacles_num)

        obstacles_embedded = self.obstacle_embedding(obstacles_data.permute(2, 0, 1))  # (obstacles_num, batch_size, latent_dim)

        # Combine the embedded observation and obstacles
        combined_input = torch.cat((obs_embedded, obstacles_embedded), dim=0)  # (1 + obstacles_num, batch_size, latent_dim)

        # Generate attention mask for Transformer
        attention_mask = torch.cat((torch.zeros(mask.size(0), 1, dtype=torch.bool, device=device), mask == 0), dim=1)  # (batch_size, 1 + obstacles_num)

        transformer_output = self.transformer_encoder(combined_input, src_key_padding_mask=attention_mask)  # (1 + obstacles_num, batch_size, latent_dim)
        transformer_output = transformer_output[0, :, :]  # 取第一个token的输出，代表全局信息 (batch_size, latent_dim)

        # combined_out = torch.cat([transformer_output, act], dim=-1)  # (batch_size, latent_dim + act_dim)
        q = self.q_layer(transformer_output)  # (batch_size, 1)
        return q.squeeze(-1)
    

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            # I have to change here for GPU
            return a.cpu().numpy()
        
class AttentionActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        state_dim = 6
        goal_dim = 6
        perception_dim = obs_dim - (2*state_dim + goal_dim)
        obstacle_dim = 4 # currently this is fixed
        obstacle_num = 10

        # build policy and value functions
        self.pi = SquashedGaussianAttentionActor(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, \
            act_dim, hidden_sizes, activation, act_limit)
        self.q1 = AttentionQFunction(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, \
                               act_dim, hidden_sizes, activation)
        self.q2 = AttentionQFunction(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, \
                               act_dim, hidden_sizes, activation)

    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, obstacles, deterministic, False)
            # I have to change here for GPU
            return a.cpu().numpy()
        
class TransformerActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        state_dim = 6
        goal_dim = 6
        obstacle_dim = 4 # currently this is fixed
        obstacle_num = 20

        # build policy and value functions
        self.pi = SquashedGaussianTransformerActor(state_dim, goal_dim, obstacle_dim, obstacle_num, \
            act_dim, hidden_sizes, activation, act_limit)
        self.q1 = TransformerQFunction(state_dim, goal_dim, obstacle_dim, obstacle_num, \
                               act_dim, hidden_sizes, activation)
        self.q2 = TransformerQFunction(state_dim, goal_dim, obstacle_dim, obstacle_num, \
                               act_dim, hidden_sizes, activation)

    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, obstacles, deterministic, False)
            # I have to change here for GPU
            return a.cpu().numpy()

class SetActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        state_dim = 6
        goal_dim = 6
        perception_dim = obs_dim - (2*state_dim + goal_dim)
        obstacle_dim = 4
        obstacle_num = 10

        # build policy and value functions
        self.pi = SquashedGaussianSetActor(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, \
            act_dim, hidden_sizes, activation, act_limit)
        self.q1 = SetQFunction(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, \
                               act_dim, hidden_sizes, activation)
        self.q2 = SetQFunction(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, \
                               act_dim, hidden_sizes, activation)

    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, obstacles, deterministic, False)
            # I have to change here for GPU
            return a.cpu().numpy()      
        
class OnlySetActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        state_dim = 6
        goal_dim = 6
        perception_dim = obs_dim - (2*state_dim + goal_dim)
        obstacle_dim = 4
        obstacle_num = 10

        # build policy and value functions
        self.pi = SquashedGaussianOnlySetActor(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = OnlySetQFunction(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation)
        self.q2 = OnlySetQFunction(state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation)

    def act(self, obs, obstacles, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, obstacles, deterministic, False)
            return a.cpu().numpy()

class SquashedGaussianOnlySetActor(nn.Module):
    def __init__(self, state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        """we take the perception out"""
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.perception_dim = perception_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.obstacle_net = mlp([2*state_dim + goal_dim + obstacle_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, obstacles, deterministic=False, with_logprob=True):
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0)
        else:
            squeeze = False
        
        obs_vehicle = obs[:, :(2*self.state_dim + self.goal_dim)] # (batch_size, (2*state_dim + goal_dim))
        obs_vehicle = obs_vehicle.unsqueeze(2).expand(-1, -1, obstacles.size(2)) #(batch_size, (2*state_dim + goal_dim), obstacles_num)
        
        
        obstacles_data = obstacles[:, :self.obstacle_dim, :] # (batch_size, obstacle_dim, obstacles_num)
        obs_with_obstacles = torch.cat([obs_vehicle, obstacles_data], dim=1) # (batch_size, (2*state_dim + goal_dim + obstacle_dim), obstacles_num)
        obs_with_obstacles = obs_with_obstacles.permute(0,2,1) # (batch_size, obstacles_num, (2*state_dim + goal_dim + obstacle_dim)
        obs_with_obstacles = obs_with_obstacles.contiguous().view(-1, 2*self.state_dim + self.goal_dim + self.obstacle_dim) # (batch_size*obstacles_num, (2*state_dim + goal_dim + obstacle_dim))
        
        obstacle_outs = self.obstacle_net(obs_with_obstacles) # (batch_size*obstacles_num, hidden_sizes[-1])
        obstacle_outs = obstacle_outs.view(obs.size(0), obstacles.size(2), -1).permute(0, 2, 1) # (batch_size, hidden_sizes[-1], self.obstacles_num)

        
        obstacle_masks = obstacles[:, -1, :].unsqueeze(1) #(batch_size, 1, self.obstacle_num)
        
        
        weighted_obstacle_outs = obstacle_outs * obstacle_masks
        mean_obstacle_out = weighted_obstacle_outs.sum(dim=2) # (batch_size, hidden_sizes[-1])
        
        mu = self.mu_layer(mean_obstacle_out)
        log_std = self.log_std_layer(mean_obstacle_out)
        
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        if squeeze:
            mu = mu.squeeze()
            std = std.squeeze()
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class OnlySetQFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, perception_dim, obstacle_dim, obstacle_num, act_dim, hidden_sizes, activation):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.perception_dim = perception_dim
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        self.obstacle_net = mlp([2*state_dim + goal_dim + obstacle_dim] + list(hidden_sizes), activation, activation)
        self.q_layer = mlp([2*state_dim + goal_dim + hidden_sizes[-1] + act_dim] + list(hidden_sizes) + [1], activation)
        self.hidden_size = hidden_sizes[-1]

    def forward(self, obs, obstacles, act):
        
        obs_vehicle = obs[:, :(2*self.state_dim + self.goal_dim)] # (batch_size, (2*state_dim + goal_dim))
        
        obs_vehicle = obs_vehicle.unsqueeze(2).expand(-1, -1, obstacles.size(2)) #(batch_size, (2*state_dim + goal_dim), obstacles_num)

        obstacles_data = obstacles[:, :self.obstacle_dim, :] # (batch_size, obstacle_dim, obstacles_num)
        obs_with_obstacles = torch.cat([obs_vehicle, obstacles_data], dim=1) # (batch_size, (2*state_dim + goal_dim + obstacle_dim), obstacles_num)
        obs_with_obstacles = obs_with_obstacles.permute(0,2,1) # (batch_size, obstacles_num, (2*state_dim + goal_dim + obstacle_dim)
        obs_with_obstacles = obs_with_obstacles.contiguous().view(-1, 2*self.state_dim + self.goal_dim + self.obstacle_dim) # (batch_size*obstacles_num, (2*state_dim + goal_dim + obstacle_dim))

        obstacle_outs = self.obstacle_net(obs_with_obstacles) # (batch_size*obstacles_num, hidden_sizes[-1])
        obstacle_outs = obstacle_outs.view(obs.size(0), obstacles.size(2), -1).permute(0, 2, 1) # (batch_size, hidden_sizes[-1], self.obstacles_num)

        
        obstacle_masks = obstacles[:, -1, :].unsqueeze(1) #(batch_size, 1, self.obstacle_num)
        
        
        weighted_obstacle_outs = obstacle_outs * obstacle_masks
        mean_obstacle_out = weighted_obstacle_outs.sum(dim=2) # (batch_size, hidden_sizes[-1])
        combined_out = torch.cat([obs[:, :(2*self.state_dim + self.goal_dim)], mean_obstacle_out, act], dim=-1)
        q = self.q_layer(combined_out)
        return torch.squeeze(q, -1)  # Ensure q has the right shape
    

    
 
class CNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianCNNActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = CNNQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = CNNQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, image_obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, image_obs, deterministic, False)
            # I have to change here for GPU
            return a.cpu().numpy()
   
# (s, a) s: vector + image, a: 2-dimensional vector
# pi(s) -> a
# q(s, a) -> q-value   
# class Renet18ActorCritic(nn.Module):
#     pass

# # ResNet18 agent
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

# class ResNetAgent(nn.Module):

#     def __init__(self, params):
#         super(ResNetAgent, self).__init__()
#         # stands for mean & variance
#         self.num_classes = 2
#         block = BasicBlock
#         layers = [2, 2, 2, 2]
#         self.graph_shape = (3, 84, 84)
#         # self.vector_shape = 18 # 

#         self.reward = 0
#         self.gamma = 0.9
#         self.short_memory = np.array([])
#         self.agent_target = 1
#         self.agent_predict = 0
#         self.learning_rate = params['learning_rate']
#         self.epsilon = 1
#         self.actual = []
#         self.zero_layer = params['graph_output_size']
#         self.first_layer = params['first_layer_size']
#         self.second_layer = params['second_layer_size']
#         self.third_layer = params['third_layer_size']
        
#         self.weights = params['weights_path']
#         self.load_weights = params['load_weights']
#         self.optimizer = None

#         self.inplanes = 16
#         self.conv1 = nn.Conv2d(self.graph_shape[0], self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         self.layer1 = self._make_layer(block, self.inplanes, layers[0])
#         self.layer2 = self._make_layer(block, 2 * self.inplanes, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 4 * self.inplanes, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 8 * self.inplanes, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(3, stride=1)
#         self.fc = nn.Linear(8 * self.inplanes * block.expansion, self.zero_layer)

#         # self.f1 = nn.Linear(self.zero_layer + self.vector_shape, self.first_layer)
#         # self.f2 = nn.Linear(self.first_layer, self.second_layer)
#         # self.f3 = nn.Linear(self.second_layer, self.third_layer)
#         # self.f4 = nn.Linear(self.third_layer, self.num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
        
#         if self.load_weights:
#             self.model = self.load_state_dict(torch.load(self.weights))
#             print("weights loaded")

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x, y):
#         # for input size: 7 * 22 * 22
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc(x))

#         # x = F.relu(self.f1(torch.cat((x, y), 1)))
#         # x = F.relu(self.f2(x))
#         # x = F.relu(self.f3(x))
#         # x = F.relu(self.f4(x))

#         # x = F.softmax(x, dim=-1)
#         return x