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

# class SquashedGaussianMLPActor(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
#         self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
#         self.act_limit = act_limit

#     def forward(self, obs, deterministic=False, with_logprob=True):
#         net_out = self.net(obs)
#         mu = self.mu_layer(net_out)
#         log_std = self.log_std_layer(net_out)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)

#         # Pre-squash distribution and sample
#         pi_distribution = Normal(mu, std)
#         if deterministic:
#             # Only used for evaluating policy at test time.
#             pi_action = mu
#         else:
#             pi_action = pi_distribution.rsample()

#         if with_logprob:
#             # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
#             # NOTE: The correction formula is a little bit magic. To get an understanding 
#             # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
#             # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
#             # Try deriving it yourself as a (very difficult) exercise. :)
#             logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
#             logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
#         else:
#             logp_pi = None

#         pi_action = torch.tanh(pi_action)
#         pi_action = self.act_limit * pi_action

#         return pi_action, logp_pi


# class MLPQFunction(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

#     def forward(self, obs, act):
#         q = self.q(torch.cat([obs, act], dim=-1))
#         return torch.squeeze(q, -1) # Critical to ensure q has right shape.
    


# class MLPActorCritic(nn.Module):

#     def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
#                  activation=nn.ReLU):
#         super().__init__()

#         obs_dim = observation_space.shape[0]
#         act_dim = action_space.shape[0]
#         act_limit = action_space.high[0]

#         # build policy and value functions
#         self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
#         self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
#         self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

#     def act(self, obs, deterministic=False):
#         with torch.no_grad():
#             a, _ = self.pi(obs, deterministic, False)
#             # I have to change here for GPU
#             return a.cpu().numpy()
        

        
        
class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, skill_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim + skill_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, skill, deterministic=False, with_logprob=True):
        net_out = self.net(torch.cat([obs, skill], dim=-1))
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
    
class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, skill_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + skill_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, skill, act):
        q = self.q(torch.cat([obs, skill, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.
    

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, skill_dim, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, skill_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, skill_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, skill_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, skill, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, skill, deterministic, False)
            return a.cpu().numpy()
        
class MLPDiscriminator(nn.Module):
    # TODO: test batch_size input
    def __init__(self, obs_dim, skill_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.discriminator_network = mlp([obs_dim] + list(hidden_sizes) + [skill_dim], activation)

    def forward(self, state):
        logits = self.discriminator_network(state)  # 输出每个技能的logits
        return logits

    def log_prob(self, state, skill):
        """
        state (Tensor): (batch_size, obs_dim)
        skill (Tensor): (batch_size, skill_dim) (one-hot encoding)
        """
        logits = self.forward(state)
        log_prob = F.log_softmax(logits, dim=-1)  
        skill_index = skill.argmax(dim=-1).long()  # 获取技能的索引并转换为 int64
        return log_prob.gather(1, skill_index.unsqueeze(-1)).squeeze(-1)  # (batch_size,)
        # return log_prob.gather(1, skill.unsqueeze(-1)).squeeze(-1)  # (batch_size,)
