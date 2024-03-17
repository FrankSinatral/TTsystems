from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
# import gym
import time
import os
import os.path as osp
import sys
from tqdm import trange
from tqdm import tqdm
import pickle

# import core
# Some of the nn defined here
import rl_agents.sac.core as core
# Try to add logger
from rl_agents.utils.logx import EpochLogger
from rl_agents.query_expert import query_hybrid_astar_one_trailer, query_hybrid_astar_three_trailer
import gymnasium as gym
from gymnasium.spaces import Box
import random
import tractor_trailer_envs as tt_envs
import threading
import multiprocessing
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from PIL import Image
import io

def find_expert_trajectory(o, vehicle_type):
    goal = o["desired_goal"]
    input = o["observation"]
    try:
        obstacles_info = o["obstacles_info"]
    except:
        obstacles_info = None
        
    config = {
        "plot_final_path": False,
        "plot_rs_path": False,
        "plot_expand_tree": False,
        "mp_step": 12, # Important
        "N_steps": 20, # Important
        "range_steer_set": 20,
        "max_iter": 50,
        "controlled_vehicle_config": {
                    "w": 2.0, #[m] width of vehicle
                    "wb": 3.5, #[m] wheel base: rear to front steer
                    "wd": 1.4, #[m] distance between left-right wheels (0.7 * W)
                    "rf": 4.5, #[m] distance from rear to vehicle front end
                    "rb": 1.0, #[m] distance from rear to vehicle back end
                    "tr": 0.5, #[m] tyre radius
                    "tw": 1.0, #[m] tyre width
                    "rtr": 2.0, #[m] rear to trailer wheel
                    "rtf": 1.0, #[m] distance from rear to trailer front end
                    "rtb": 3.0, #[m] distance from rear to trailer back end
                    "rtr2": 2.0, #[m] rear to second trailer wheel
                    "rtf2": 1.0, #[m] distance from rear to second trailer front end
                    "rtb2": 3.0, #[m] distance from rear to second trailer back end
                    "rtr3": 2.0, #[m] rear to third trailer wheel
                    "rtf3": 1.0, #[m] distance from rear to third trailer front end
                    "rtb3": 3.0, #[m] distance from rear to third trailer back end   
                    "max_steer": 0.6, #[rad] maximum steering angle
                    "v_max": 2.0, #[m/s] maximum velocity 
                    "safe_d": 0.0, #[m] the safe distance from the vehicle to obstacle 
                    "xi_max": (np.pi) / 4, # jack-knife constraint  
                },
        "acceptance_error": 0.5,
        }
    if vehicle_type == "one_trailer":
        pack_transition_list = query_hybrid_astar_one_trailer(input, goal)
    elif vehicle_type == "three_trailer":
        pack_transition_list = query_hybrid_astar_three_trailer(input, goal, obstacles_info, config)
    return pack_transition_list

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

class ImageReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.obs_image_buf = np.zeros((size, 3, 84, 84), dtype=np.uint8)  # Buffer for current observation images
        self.obs2_image_buf = np.zeros((size, 3, 84, 84), dtype=np.uint8)  # Buffer for next observation images
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, obs_image, act, rew, next_obs, next_obs_image, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.obs_image_buf[self.ptr] = obs_image  # Store current observation image
        self.obs2_image_buf[self.ptr] = next_obs_image  # Store next observation image
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                 obs2=self.obs2_buf[idxs],
                 act=self.act_buf[idxs],
                 rew=self.rew_buf[idxs],
                 done=self.done_buf[idxs],
                 obs_image=self.obs_image_buf[idxs].astype(np.float32),  # Convert to np.float32
                 obs2_image=self.obs2_image_buf[idxs].astype(np.float32))  # Convert to np.float32
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

class SACASTAR:
    def __init__(self, 
                 env_fn, 
                 algo='sac',
                 actor_critic=core.MLPActorCritic, 
                 ac_kwargs=dict(), 
                 seed=0, 
                 steps_per_epoch=4000, 
                 epochs=50, 
                 replay_size=int(1e6),
                 gamma=0.99, 
                 polyak=0.995,
                 lr=1e-3,
                 alpha=0.2,
                 batch_size=100,
                 start_steps=10000,
                 update_after=1000,
                 update_every=50,
                 max_ep_len=1000,
                 logger_kwargs=dict(), 
                 save_freq=10, 
                 num_test_episodes=10,
                 log_dir='runs_rl/',
                 whether_her=True,
                 use_automatic_entropy_tuning=False,
                 log_alpha_lr=1e-3,
                 env_name='reaching-v0',
                 pretrained=False,
                 pretrained_itr=None,
                 pretrained_dir=None,
                 config: dict = None,
                 device = None,
                 with_env_number = 100,
                 with_astar_number = 100,
                 align_with_env = False,
                 args = None):    
        """
        new function for astar + sac update
        """ 
        # TODO: we now take MPI tools out
        self.logger = EpochLogger(**logger_kwargs)
        # save your configuration in a json file
        self.logger.save_config(locals()) 
        if device is not None:
            self.device = device
        else:
            self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env_name = env_name
        # Instantiate environment
        if self.env_name == "standard_parking":
            # Fank: register only once the highway_env
            # Now seldom use this env
            from highway_env import register_highway_envs
            register_highway_envs()
            self.env, self.test_env = env_fn(), env_fn() # using gym to instantiate env
            self.state_dim = self.env.observation_space['observation'].shape[0]
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim,), np.float64)
            self.obs_dim = self.box.shape
            self.act_dim = self.env.action_space.shape[0]
            # Action limit for clamping: critically, assumes all dimensions share the same bound!
            self.act_limit = self.env.action_space.high[0]
            # recent seed 
            self.env.action_space.seed(seed)
            # self.env.observation_space.seed(seed)
            self.test_env.action_space.seed(seed)
            # self.test_env.observation_space.seed(seed)
            # Create actor-critic module and target networks
            self.ac = actor_critic(self.box, self.env.action_space, **ac_kwargs).to(self.device)
            self.ac_targ = deepcopy(self.ac)
        else:
            # Fank: only need to register tt envs once
            from tractor_trailer_envs import register_tt_envs
            register_tt_envs()
            self.env, self.test_env = env_fn(config), env_fn(config)
            self.state_dim = self.env.observation_space['observation'].shape[0]
            if self.env_name.startswith("cluttered"):
                self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 8,), np.float32)
            else:
                self.box = Box(-np.inf, np.inf, (3 * self.state_dim,), np.float32)
            self.obs_dim = self.box.shape
            self.act_dim = self.env.action_space.shape[0]
            # Action limit for clamping: critically, assumes all dimensions share the same bound!
            self.act_limit = self.env.action_space.high[0]
            # Fank: only need to seed action space
            self.env.action_space.seed(seed)
            self.test_env.action_space.seed(seed)
            # Create actor-critic module and target networks
            self.ac = actor_critic(self.box, self.env.action_space, **ac_kwargs).to(self.device)
            self.ac_targ = deepcopy(self.ac)
            
            # vehicle_type
            self.vehicle_type = config["vehicle_type"]
        
        # set up summary writer
        self.exp_name = logger_kwargs['exp_name']
        
        self.save_model_path = log_dir  + self.exp_name
        self.writer = SummaryWriter(log_dir=self.save_model_path)
        
        # Initialize Hyperparameters for SAC algorithm
        self.gamma = gamma
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.replay_size = replay_size
        self.start_steps = start_steps
        self.update_every = update_every
        self.update_after = update_after
        self.lr = lr
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.polyak = polyak
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        if self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
            self.replay_buffer = ImageReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size, device=self.device)
        else:
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size, device=self.device)
        
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        
        self.algo = algo
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        
        
        # setting whether this is standard highway parking env
        # set HER replay buffer
        self.whether_her = whether_her
        
        
        # Fank: automatic tuning for alpha
        self.log_alpha_lr = log_alpha_lr
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        
        target_entropy = None
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                target_entropy = -self.act_dim  # heuristic value
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(0.0).to(self.device)
            self.alpha=1.0
            self.log_alpha.requires_grad=True
            self.alpha_optimizer = Adam(
                [self.log_alpha],
                lr=self.log_alpha_lr,
            )
        else:
            self.alpha = alpha
        
        if pretrained == True:
            # Fank: use any pretrained model
            itr = str(pretrained_itr) if pretrained_itr >= 0 else 'final'
            pretrained_file = pretrained_dir + 'model_' + itr + '.pth'
            print("Using pretrained model from {}".format(pretrained_file))
            self.load(pretrained_file)
            self.ac_targ = deepcopy(self.ac)
            
        self.with_env_number = with_env_number
        self.with_astar_number = with_astar_number
        self.align_with_env = align_with_env
        print("Running off-policy RL algorithm: {}".format(self.algo))
    

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        if self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
            o, o_image, a, r, o2, o2_image, d = data['obs'], data['obs_image'], data['act'], data['rew'], data['obs2'], data['obs2_image'], data['done']
            q1 = self.ac.q1(o,o_image,a)
            q2 = self.ac.q2(o,o_image,a)
        
        else: 
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    
            q1 = self.ac.q1(o,a)
            q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            if self.env_name.startswith('cluttered') and self.env.unwrapped.config["use_rgb"]:
                # Target actions come from *current* policy
                a2, logp_a2 = self.ac.pi(o2, o2_image)
                # Target Q-values
                q1_pi_targ = self.ac_targ.q1(o2, o2_image, a2)
                q2_pi_targ = self.ac_targ.q2(o2, o2_image, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
            else:
                # Target actions come from *current* policy
                a2, logp_a2 = self.ac.pi(o2)

                # Target Q-values
                q1_pi_targ = self.ac_targ.q1(o2, a2)
                q2_pi_targ = self.ac_targ.q2(o2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        # critic loss that you can save
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        if self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
            o, o_image = data['obs'], data['obs_image']
            pi, logp_pi = self.ac.pi(o, o_image)
            q1_pi = self.ac.q1(o, o_image, pi)
            q2_pi = self.ac.q2(o, o_image, pi)
        else:   
            o = data['obs']
            pi, logp_pi = self.ac.pi(o)
            q1_pi = self.ac.q1(o, pi)
            q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info, logp_pi
    
    def compute_loss_alpha(self, logp_pi):
        average_entropy = 0

        if self.use_automatic_entropy_tuning:
            with torch.no_grad():
                average_entropy = - (logp_pi.mean())
            alpha_loss = self.log_alpha * (average_entropy - self.target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp() # alpha is detached
            alpha_loss = alpha_loss.detach()
        else:
            alpha_loss = 0
        return self.alpha, alpha_loss, average_entropy

    def update(self, data, global_step):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Here record Critic Loss and q_info
        # self.logger.store(LossQ=loss_q.item(), **q_info)
        self.writer.add_scalar('train/critic_loss', loss_q.item(), global_step=global_step)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info, logp_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        
        # Perform Automatically Tuning Alpha
        alpha, alpha_loss, average_entropy = self.compute_loss_alpha(logp_pi)

        if self.use_automatic_entropy_tuning:
            self.writer.add_scalar("train/alpha", alpha.item(), global_step=global_step)
            self.writer.add_scalar("train/alpha_loss", alpha_loss.item(), global_step=global_step)
            self.writer.add_scalar("train/average_entropy", average_entropy.item(), global_step=global_step)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Here record Actor Loss and pi_info
        # self.logger.store(LossPi=loss_pi.item(), **pi_info)
        self.writer.add_scalar('train/actor_loss', loss_pi.item(), global_step=global_step)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False, rgb_image=None):
        if self.env_name.startswith('cluttered') and self.env.unwrapped.config["use_rgb"]:
            # the only different env is the cluttered env with use_rgb image wanted
            return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), torch.as_tensor(rgb_image, dtype=torch.float32).to(self.device),
                      deterministic)
        else:
            return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), 
                        deterministic)

    def test_agent(self, global_steps):
        average_ep_ret = 0.0
        average_ep_len = 0.0
        success_rate = 0.0
        jack_knife_rate = 0.0
        crash_rate = 0.0
        for j in range(self.num_test_episodes):
            o, info = self.test_env.reset(seed=j)
            terminated, truncated, ep_ret, ep_len = False, False, 0, 0
            while not(terminated or truncated):
                # Take deterministic actions at test time 
                if self.env_name.startswith('cluttered') and not self.env.unwrapped.config["use_rgb"]:
                    o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), True))
                elif self.env_name.startswith('cluttered') and self.env.unwrapped.config["use_rgb"]:
                    o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), True, rgb_image=o["achieved_rgb_image"]))
                else:  
                    o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), True))
                ep_ret += r
                ep_len += 1
            average_ep_ret += ep_ret
            average_ep_len += ep_len
            if info['is_success']:
                success_rate += 1
            if info['jack_knife']:
                jack_knife_rate += 1
            if info['crashed']:
                crash_rate += 1
            self.logger.store(ep_rew_mean=ep_ret, ep_len_mean=ep_len)
        jack_knife_rate /= self.num_test_episodes
        success_rate /= self.num_test_episodes  
        crash_rate /= self.num_test_episodes 
        average_ep_ret /= self.num_test_episodes
        average_ep_len /= self.num_test_episodes
        self.logger.store(success_rate=success_rate, jack_knife_rate=jack_knife_rate, crash_rate=crash_rate)
        self.logger.store(total_timesteps=global_steps)
        self.writer.add_scalar('evaluate/ep_rew_mean', average_ep_ret, global_step=global_steps) 
        self.writer.add_scalar('evaluate/ep_len_mean', average_ep_len, global_step=global_steps)   
        self.writer.add_scalar('evaluate/success_rate', success_rate, global_step=global_steps)
        self.writer.add_scalar('evaluate/jack_knife_rate', jack_knife_rate, global_step=global_steps)
        self.writer.add_scalar('evaluate/crash_rate', crash_rate, global_step=global_steps)
        self.logger.log_tabular('ep_rew_mean', with_min_and_max=False, average_only=True)
        self.logger.log_tabular('ep_len_mean', with_min_and_max=False, average_only=True)
        self.logger.log_tabular('success_rate', with_min_and_max=False, average_only=True)
        self.logger.log_tabular('jack_knife_rate', with_min_and_max=False, average_only=True)
        self.logger.log_tabular('crash_rate', with_min_and_max=False, average_only=True)
        self.logger.log_tabular('total_timesteps', with_min_and_max=False, average_only=True)
        self.logger.dump_tabular()
        
    def update_process(self):
        print("start update")
        update_start_time = time.time()
        for _ in range(self.update_every):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            self.update(data=batch, global_step=self.total_timesteps)
        update_end_time = time.time()
        print("update time:", update_end_time - update_start_time)
        
    def save_and_evaluate(self):
        epoch = (self.total_timesteps + 1) // self.steps_per_epoch
        # Save model
        if (epoch % self.save_freq == 0) or (epoch == self.epochs):
            self.save(self.save_model_path +'/model_' + str(self.total_timesteps) + '.pth')
        print("start testing")
        test_start_time = time.time()
        self.test_agent(self.total_timesteps)
        test_end_time = time.time()
        print("test time:", test_end_time - test_start_time)
        
    def collect_rollout_with_env(self, seed_number):
        """In this function, we collect the rollout with the environment"""
        ## total time-step and finish episode number
        episode_start_time = time.time()
        # give a new task
        o, info = self.env.reset(seed=seed_number)
        ep_ret, ep_len = 0, 0
        # Loop until the episode finish
        while True:
            if self.total_timesteps <= self.start_steps:
                a = self.env.action_space.sample()
            else:
                if self.env_name.startswith("cluttered") and not self.env.unwrapped.config["use_rgb"]:
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]))
                elif self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), rgb_image=o["achieved_rgb_image"])
                else:
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]))
            o2, r, terminated, truncated, info = self.env.step(a)
            ep_ret += r
            ep_len += 1
            d = terminated and (not truncated)
            if self.env_name.startswith("cluttered") and not self.env.unwrapped.config["use_rgb"]:
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['obstacles_info']]), d)
            elif self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), o['achieved_rgb_image'], a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['obstacles_info']]), o2["achieved_rgb_image"], d)
            else:
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal']]), d)
            self.env_timesteps += 1
            self.total_timesteps += 1
            # check whether need to update
            if self.total_timesteps >= self.update_after and self.total_timesteps % self.update_every == 0:
                self.update_process()
            
            # check whether need to save the model and evaluate the model
            if (self.total_timesteps + 1) % self.steps_per_epoch == 0:
                self.save_and_evaluate()
            
            # can exit just like astar do 
            if self.total_timesteps >= self.steps_per_epoch * self.epochs:
                return
            
            # check whether the episode is terminated or truncated:
            if terminated or truncated:
                self.finish_episode_number_with_env += 1
                episode_end_time = time.time()
                print("Episode Time:", episode_end_time - episode_start_time)
                print("Episode Length:", ep_len)
                return
            o = o2
    
    def collect_lists_for_planning(self, start_seed, total_number):
        lists_for_planning = []
        for j in range(total_number):
            o, info = self.env.reset(seed=(start_seed+j))
            lists_for_planning.append(o)
        return lists_for_planning
          
    def collect_rollout_with_astar(self, lists_for_planning, total_number):
        """astar rollout collection function"""
        planning_start_time = time.time()
        if lists_for_planning is not None:
            print("Start Collecting Rollouts from Astar (align tasks)")
        else:
            lists_for_planning = []
            for j in range(total_number):
                o, info = self.env.reset()
                lists_for_planning.append(o)
            print("Start Collecting Rollouts from Astar (random tasks)")
        
        # Running planning tasks using joblib
        astar_results = Parallel(n_jobs=-1)(delayed(find_expert_trajectory)(o, self.vehicle_type) for o in lists_for_planning)
        # substitude for debugging
        # astar_results = [find_expert_trajectory(o, self.vehicle_type) for o in lists_for_planning]
        for pack_transition_list in astar_results:
            if pack_transition_list is None:
                pass
            else:
                for transition in pack_transition_list:
                    o, a, o2, r, d = transition
                    if self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
                        o_image = self.env.unwrapped.reconstruct_image_from_observation(o.astype(np.float32))
                        o2_image = self.env.unwrapped.reconstruct_image_from_observation(o2.astype(np.float32))
                        self.replay_buffer.store(o.astype(np.float32), o_image, a.astype(np.float32), r, o2.astype(np.float32), o2_image, d)
                    else:
                        self.replay_buffer.store(o.astype(np.float32), a.astype(np.float32), r, o2.astype(np.float32), d)
                    self.astar_timesteps += 1
                    self.total_timesteps += 1
                    # check whether need to update
                    if self.total_timesteps >= self.update_after and self.total_timesteps % self.update_every == 0:
                        self.update_process()
                    
                    # check whether need to save the model and evaluate the model
                    if (self.total_timesteps + 1) % self.steps_per_epoch == 0:
                        self.save_and_evaluate()   
                    
                    if self.total_timesteps >= self.steps_per_epoch * self.epochs:
                        return
                        
                self.finish_episode_number_with_astar += 1
        planning_end_time = time.time()
        print("Planning Time:", planning_end_time - planning_start_time)
        
        
    def run(self):
        
        self.finish_episode_number_with_env = 0
        self.finish_episode_number_with_astar = 0
        self.total_timesteps = 0
        self.env_timesteps = 0
        self.astar_timesteps = 0
        self.seed_number_with_env = self.seed
        self.seed_number_with_astar = self.seed
        exit_loop = False
        assert self.with_env_number > 0 or self.with_astar_number > 0, "One Should Beyond 0"
        while True:
            if self.with_env_number > 0:
                for _ in range(self.with_env_number):
                    self.collect_rollout_with_env(self.seed_number_with_env)
                    if self.total_timesteps >= self.steps_per_epoch * self.epochs:
                        print("Reaching Maximum Steps")
                        exit_loop = True
                        break
                    self.seed_number_with_env += 1
                if exit_loop:
                    break
            if self.with_astar_number > 0:
                if self.align_with_env:
                    lists_for_planning = self.collect_lists_for_planning(self.seed_number_with_astar, self.with_astar_number)
                    self.seed_number_with_astar += self.with_astar_number
                else: 
                    lists_for_planning = None
                self.collect_rollout_with_astar(lists_for_planning, self.with_astar_number)
                if self.total_timesteps >= self.steps_per_epoch * self.epochs:
                    print("Reaching Maximum Steps")
                    break   
        self.save(self.save_model_path +'/model_final.pth')
                
    def save(self, filename):
        state = {'ac_state_dict': self.ac.state_dict(),
                 'alpha': self.alpha,
                 'pi_optimizer': self.pi_optimizer.state_dict(),
                 'q_optimizer': self.q_optimizer.state_dict(),
                 'alpha_optimizer': self.alpha_optimizer.state_dict(),
                 }
        torch.save(state, filename)
        
        # # add save buffer
        # buffer_filename = filename.replace('.pth', '_buffer.pkl')
        # with open(buffer_filename, 'wb') as f:
        #     pickle.dump(self.sreplay_buffer, f)
        
        return filename
    
    def load(self, filename, whether_load_buffer=True):
        checkpoint = torch.load(filename, map_location=self.device)
        self.ac.load_state_dict(checkpoint['ac_state_dict'])
        self.alpha = checkpoint['alpha']
        self.pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        # TODO: this may have to change
        if whether_load_buffer:
            buffer_filename = filename.replace('.pth', '_buffer.pkl')
            with open(buffer_filename, 'rb') as f:
                self.replay_buffer = pickle.load(f)
        
        return filename
    
    def her_process_episode(self, episode_data, goal_selection_strategy="future", k=4):
        """
        fullfill all strategy except random
        because this strategy needs the whole buffer and there's no crashed value tracking
        TODO: due to the data shape, now still only use for standard parking env
        """
        assert goal_selection_strategy in ["final", "future", "episode"]
        
        if goal_selection_strategy == "final":
            new_goal = episode_data[-1][3][:self.state_dim]
            for j in range(len(episode_data)):
                transition = episode_data[j]
                o_concatenate, a, r, o2_concatenate, d, crashed = transition
                new_obs = np.concatenate([o_concatenate[:self.state_dim], new_goal])
                new_obs2 = np.concatenate([o2_concatenate[:self.state_dim], new_goal])
                if self.env_name == "standard_parking":
                    new_reward = self.compute_standard_parking_reward(new_obs2, crashed)
                else:
                    new_reward = self.env.unwrapped.reward(o_concatenate[:self.state_dim], o2_concatenate[:self.state_dim],
                                                      new_goal)
                self.replay_buffer.store(new_obs, a, new_reward, new_obs2, d)
        elif goal_selection_strategy == "future":
            for j in range(len(episode_data) - 1):
                transition = episode_data[j]
                o_concatenate, a, r, o2_concatenate, d, crashed = transition
                try:
                    picked_index = random.sample(range(j + 1, len(episode_data)), k)
                except ValueError:
                    picked_index = list(range(j + 1, len(episode_data)))
                
                new_goals = [episode_data[i][3][:self.state_dim] for i in picked_index]
                if self.env_name == "standard_parking":
                    for new_goal in new_goals:
                        new_obs = np.concatenate([o_concatenate[:self.state_dim], new_goal])
                        new_obs2 = np.concatenate([o2_concatenate[:self.state_dim], new_goal])
                        new_reward = self.compute_standard_parking_reward(new_obs2, crashed)
                        self.replay_buffer.store(new_obs, a, new_reward, new_obs2, d)
                else:
                    for new_goal in new_goals:
                        new_obs = np.concatenate([o_concatenate[:self.state_dim], new_goal])
                        new_obs2 = np.concatenate([o2_concatenate[:self.state_dim], new_goal])
                        new_reward = self.env.unwrapped.reward(o_concatenate[:self.state_dim], o2_concatenate[:self.state_dim],
                                                      new_goal)
                        self.replay_buffer.store(new_obs, a, new_reward, new_obs2, d)
        elif goal_selection_strategy == "episode":
            for j in range(len(episode_data)):
                transition = episode_data[j]
                o_concatenate, a, r, o2_concatenate, d, crashed = transition
                picked_index = random.sample(range(0, len(episode_data)), k)
                new_goals = [episode_data[i][3][:self.state_dim] for i in picked_index]
                if self.env_name == "standard_parking":
                    for new_goal in new_goals:
                        new_obs = np.concatenate([o_concatenate[:self.state_dim], new_goal])
                        new_obs2 = np.concatenate([o2_concatenate[:self.state_dim], new_goal])
                        new_reward = self.compute_standard_parking_reward(new_obs2, crashed)
                        self.replay_buffer.store(new_obs, a, new_reward, new_obs2, d)
                else:
                    for new_goal in new_goals:
                        new_obs = np.concatenate([o_concatenate[:self.state_dim], new_goal])
                        new_obs2 = np.concatenate([o2_concatenate[:self.state_dim], new_goal])
                        new_reward = self.env._reward(o_concatenate[:self.state_dim], o2_concatenate[:self.state_dim],
                                                      new_goal)
                        self.replay_buffer.store(new_obs, a, new_reward, new_obs2, d)
                
    
    
    # use the same logic in highway env         
    def compute_standard_parking_reward(self, new_obs, crashed):
        achieved_goal = new_obs[:self.state_dim]
        desired_goal = new_obs[self.state_dim:]
        reward_weights = [1, 0.3, 0.0, 0.0, 0.02, 0.02]
        p = 0.5
        collision_reward = -5
        reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(reward_weights)), p)
        if crashed:
            reward += collision_reward
        return reward