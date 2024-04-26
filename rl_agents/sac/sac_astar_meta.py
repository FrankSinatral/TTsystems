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
from rl_agents.query_expert import find_expert_trajectory_meta
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

    # def store(self, obs, act, rew, next_obs, done):
    #     self.obs_buf[self.ptr] = obs
    #     self.obs2_buf[self.ptr] = next_obs
    #     self.act_buf[self.ptr] = act
    #     self.rew_buf[self.ptr] = rew
    #     self.done_buf[self.ptr] = done
    #     self.ptr = (self.ptr+1) % self.max_size
    #     self.size = min(self.size+1, self.max_size)
        
    def store(self, obs, act, rew, next_obs, done):
        # Check if any of the inputs except 'done' contain NaN, Inf, or -Inf.
        if not (np.any(np.isnan(obs)) or np.any(np.isnan(next_obs)) or 
                np.any(np.isnan(act)) or np.any(np.isnan(rew)) or 
                np.any(np.isinf(obs)) or np.any(np.isinf(next_obs)) or 
                np.any(np.isinf(act)) or np.any(np.isinf(rew))):
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            # Directly store 'done' without NaN/Inf check, converting boolean to float
            self.done_buf[self.ptr] = float(done)
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        else:
            print("Invalid sample encountered, skipping...")

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

class SAC_ASTAR_META:
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
                 env_name='meta-reaching-v0',
                 pretrained=False,
                 pretrained_itr=None,
                 pretrained_dir=None,
                 whether_astar=True,
                 astar_ablation=False,
                 astar_mp_steps=10,
                 astar_N_steps=10,
                 astar_max_iter=50,
                 astar_heuristic_type='traditional',
                 config: dict = None,
                 device = None,
                 whether_dataset = False,
                 dataset_path = 'datasets/goal_with_obstacles_info_list.pickle',
                 use_logger = True,
                 args = None):    
        """
        Fank: SAC_Astar meta version
        """ 
        # TODO: we now take MPI tools out
        if use_logger:
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
        
        # Fank: only need to register tt envs once
        from tractor_trailer_envs import register_tt_envs
        register_tt_envs()
        self.env, self.test_env = env_fn(config), env_fn(config)
        # vehicle_type
        self.vehicle_type = config["vehicle_type"]
        if self.vehicle_type == "single_tractor":
            self.number_bounding_box = 1
        elif self.vehicle_type == "one_trailer":
            self.number_bounding_box = 2
        elif self.vehicle_type == "two_trailer":
            self.number_bounding_box = 3
        else:
            self.number_bounding_box = 4
        if whether_dataset:
            # Fank: directly use the data from the datasets
            with open(dataset_path, 'rb') as f:
                goal_with_obstacles_info_list = pickle.load(f)
            self.env.unwrapped.update_goal_with_obstacles_info_list(goal_with_obstacles_info_list)
            # here we also use the test_env as a demonstration
            # self.test_env.unwrapped.update_goal_with_obstacles_info_list(goal_with_obstacles_info_list)  
        self.state_dim = self.env.observation_space['observation'].shape[0]
        if self.env.unwrapped.observation_type == 'original':
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim,), np.float32)
        elif self.env.unwrapped.observation_type == "lidar_detection" or self.env.unwrapped.observation_type == "one_hot_representation":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + config["perception"]["one_hot_representation"]["number"]*self.number_bounding_box,), np.float32)
        elif self.env.unwrapped.observation_type == "lidar_detection_one_hot":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 9*self.number_bounding_box,), np.float32)
        elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 98,), np.float32)
        else:
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 4,), np.float32)
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
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size, device=self.device)
        
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        if use_logger:
            self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        
        self.algo = algo
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        
        # Set up model saving, We don't use this
        # self.logger.setup_pytorch_saver(self.ac)
        
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
        # Fank: whether using astar as our expert   
        self.whether_astar = whether_astar
        # add a ablation
        self.astar_ablation = astar_ablation
        if self.astar_ablation:
            self.big_number = int(1e9)
            self.count_unrelated_task = 0
        if self.whether_astar:
            # self.lock = threading.Lock()
            self.add_astar_number = 0
        self.finish_episode_number = 0
        
        if pretrained == True:
            # Fank: use any pretrained model
            itr = str(pretrained_itr) if pretrained_itr >= 0 else 'final'
            pretrained_file = pretrained_dir + 'model_' + itr + '.pth'
            print("Using pretrained model from {}".format(pretrained_file))
            self.load(pretrained_file)
            self.ac_targ = deepcopy(self.ac)
            
        self.astar_mp_steps = astar_mp_steps
        self.astar_N_steps = astar_N_steps
        self.astar_max_iter = astar_max_iter
        self.astar_heuristic_type = astar_heuristic_type
        if use_logger:
            print("Running off-policy RL algorithm: {}".format(self.algo))
    

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
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
                if self.env.unwrapped.observation_type == "original" :
                    o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), True))
                elif self.env.unwrapped.observation_type == "lidar_detection":
                    o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection']]), True))
                elif self.env.unwrapped.observation_type == "one_hot_representation":
                    o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation']]), True))
                elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
                    o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation_enhanced']]), True))
                elif self.env.unwrapped.observation_type == "lidar_detection_one_hot":
                    o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), True))
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
        # try:
        #     self.logger.log_tabular('LossQ')
        #     self.logger.log_tabular('LossPi')
        # except:
        #     pass
        self.logger.dump_tabular()
        
    
    def add_results_to_buffer(self, results):
        # only add reconstruct for requiring image
        for pack_transition_list in results:
            if pack_transition_list is None:
                pass
            else:
                for transition in pack_transition_list:
                        o, a, o2, r, d = transition
                        self.replay_buffer.store(o.astype(np.float32), a.astype(np.float32), r, o2.astype(np.float32), d)
            self.add_astar_number += 1
        print("Add to Replay Buffer:", self.add_astar_number)
        
        
    def run(self):
        # Prepare for interaction with environment
        
        total_steps = self.steps_per_epoch * self.epochs
        # ep_ret: sum over all the rewards of an episode
        # ep_len: calculate the timesteps of an episode
        # reset this value every time we use run
        self.finish_episode_number = 0
        o, info = self.env.reset(seed=self.seed)
        episode_start_time = time.time()
        if self.whether_astar and not self.astar_ablation:
            self.add_astar_number = 0
            # to Fasten the code, we first sample the all the start list
            encounter_start_list = []
            encounter_start_list.append((o, info["obstacles_info"]))
            # add_thread = threading.Thread(target=self.add_expert_trajectory_to_buffer, args=(o,))
            # add_thread.start()
            # astar_result = find_expert_trajectory(o, self.vehicle_type)
            # self.add_results_to_buffer([astar_result])
        ep_ret, ep_len = 0, 0
        # o, ep_ret, ep_len = self.env.reset(), 0, 0
        if self.whether_her:
            temp_buffer = []

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                # TODO
                if self.env.unwrapped.observation_type == "original":
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]))
                elif self.env.unwrapped.observation_type == "lidar_detection":
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection']]))
                elif self.env.unwrapped.observation_type == "one_hot_representation":
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation']]))
                elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation_enhanced']]))
                elif self.env.unwrapped.observation_type == "lidar_detection_one_hot":
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]))
                # a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['collision_metric']]))
            else:
                a = self.env.action_space.sample()

            # Step the env
            # here is the problem to play with the env
            o2, r, terminated, truncated, info = self.env.step(a)
            
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = terminated and (not truncated)

            # Store experience to replay buffer
            #TODO: pad the observation to 6-dim
            
            if self.env.unwrapped.observation_type == "original":
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal']]), d)
            elif self.env.unwrapped.observation_type == "lidar_detection":
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection']]), d)
            elif self.env.unwrapped.observation_type == "lidar_detection_one_hot":
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection_one_hot']]), d)
            elif self.env.unwrapped.observation_type == "one_hot_representation":
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['one_hot_representation']]), d)
            elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation_enhanced']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['one_hot_representation_enhanced']]), d)
            if self.whether_her:
                # currently not used
                temp_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal']]), d, info['crashed']))
                # self.her_process_episode(temp_buffer)
            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
                
            if terminated or truncated:
                # self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                # Fank: next you give a brand new case
                episode_end_time = time.time()
                print("Finish Episode time:", episode_end_time - episode_start_time)
                self.finish_episode_number += 1
                print("Finish Episode number:", self.finish_episode_number)
                print("Episode Length:", ep_len)
                # Two Strategy for astar
                if self.whether_astar and not self.astar_ablation:
                    # TODO: change for testing
                    if self.finish_episode_number % 1000 == 0: # put this number smaller
                        print("Start Collecting Buffer from Astar")
                        # astar_results = [find_expert_trajectory_meta(o, self.env.unwrapped.map, self.astar_mp_steps, self.astar_N_steps, self.astar_max_iter, self.astar_heuristic_type, self.env.unwrapped.observation_type) for o in encounter_start_list]
                        astar_results = Parallel(n_jobs=-1)(delayed(find_expert_trajectory_meta)(o, self.env.unwrapped.map, self.astar_mp_steps, self.astar_N_steps, self.astar_max_iter, self.astar_heuristic_type, self.env.unwrapped.observation_type) for o in encounter_start_list)
                        # Clear the result
                        encounter_start_list = []
                        self.add_results_to_buffer(astar_results)
                elif self.whether_astar and self.astar_ablation:
                    # Only for reaching env
                    if self.finish_episode_number % 50 == 0:
                        print("Start Collecting Buffer from Astar(not related to the episode)")
                        # Clear the result at first
                        encounter_start_list = []
                        for _ in range(20):
                            o, info = self.test_env.reset(seed=(self.big_number + self.count_unrelated_task)) # may need to change to test_env
                            self.count_unrelated_task += 1
                            encounter_start_list.append((o, info["obstacles_info"]))
                            
                        
                        astar_results = Parallel(n_jobs=-1)(delayed(find_expert_trajectory_meta)(o, self.vehicle_type) for o in encounter_start_list)
                        
                        # astar_results = [find_expert_trajectory(o, self.vehicle_type) for o in encounter_start_list]
                        self.add_results_to_buffer(astar_results)
                o, info = self.env.reset(seed=(self.seed + t))
                if self.whether_astar and not self.astar_ablation:
                    encounter_start_list.append((o, info["obstacles_info"]))
                    
                episode_start_time = time.time()
                # if self.whether_astar:
                #     # try:
                #     #     self.add_expert_trajectory_to_buffer(o)
                #     # except:
                #     #     pass
                #     add_thread = threading.Thread(target=self.add_expert_trajectory_to_buffer, args=(o,))
                #     add_thread.start()
                ep_ret, ep_len = 0, 0
                if self.whether_her:
                    self.her_process_episode(temp_buffer)
                    temp_buffer = []
            

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                print("start update")
                update_start_time = time.time()
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch, global_step=t)
                update_end_time = time.time()
                print("done update(update time):", update_end_time - update_start_time)
            
            # This is evaluate step and save model step
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.save(self.save_model_path +'/model_' + str(t) + '.pth')
                    # pass
                    # self.logger.save_state({'env': self.env}, itr=epoch)
                print("start testing")
                test_start_time = time.time()
                # Test the performance of the deterministic version of the agent.
                self.test_agent(t)
                test_end_time = time.time()
                print("done testing(test time):", test_end_time - test_start_time)
            # if t % 1000 == 0:
            #     print("Timestep:", t)
            #     print("Total Time:", time.time() - all_start_time)
            # # print("Timestep:", t)
            # if t == 5001:
            #     break
                
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