from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
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
from utils import planner
import gymnasium as gym
from gymnasium.spaces import Box
import random
import tractor_trailer_envs as tt_envs
import threading
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import io
from tractor_trailer_envs import register_tt_envs
register_tt_envs()
def get_current_time_format():
    # get current time
    current_time = datetime.now()
    # demo: 20230130_153042
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_time

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

class ReplayBuffer_With_Obstacles:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, obstacle_dim, obstacle_num, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.obstacle_buf = np.zeros((size, obstacle_dim + 1, obstacle_num), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        self.obstacle_dim = obstacle_dim
        self.obstacle_num = obstacle_num
        
    def process_obstacles_list(self, obstacles_list):
        # Store obstacles as a (obstacle_dim + 1) * obstacle_num matrix
        if obstacles_list is not None and len(obstacles_list) > 0:
            obstacles_array = np.array(obstacles_list).T
            obstacle_dim_act, obstacle_count_act = obstacles_array.shape
            filled_obstacles = np.zeros((self.obstacle_dim + 1, self.obstacle_num), dtype=np.float32)
            filled_obstacles[:obstacle_dim_act, :obstacle_count_act] = obstacles_array
            # Set the mask row: 1 for existing obstacles, 0 for the rest
            filled_obstacles[obstacle_dim_act, :obstacle_count_act] = 1.0
            self.obstacle_buf[self.ptr] = filled_obstacles

    def store(self, obs, act, rew, next_obs, done, obstacles_list):
        # Check if any of the inputs except 'done' contain NaN, Inf, or -Inf.
        if not (np.any(np.isnan(obs)) or np.any(np.isnan(next_obs)) or 
                np.any(np.isnan(act)) or np.any(np.isnan(rew)) or 
                np.any(np.isinf(obs)) or np.any(np.isinf(next_obs)) or 
                np.any(np.isinf(act)) or np.any(np.isinf(rew))):
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = float(done)
            self.process_obstacles_list(obstacles_list)
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
                     done=self.done_buf[idxs],
                     obstacles=self.obstacle_buf[idxs])

        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}

class SAC_ASTAR_META_NEW:
    def __init__(self, 
                 env_fn,
                 device=None,
                 config: dict = None):    
        """
        Fank: SAC_Astar meta version
        """ 
        self.algo = "sac_astar"
        self.config = config
        self.steps_per_epoch = self.config.get("sac_steps_per_epoch", 4000)
        self.epochs = self.config.get("sac_epochs", 50)
        self.gamma = self.config.get("gamma", 0.99)
        self.polyak = self.config.get("polyak", 0.995)
        self.lr = self.config.get("lr", 1e-3)
        self.alpha = self.config.get("alpha", 0.2)
        self.log_alpha_lr = self.config.get("log_alpha_lr", 1e-3)
        self.batch_size = self.config.get("batch_size", 100)
        self.start_steps = self.config.get("start_steps", 10000)
        self.update_after = self.config.get("update_after", 1000)
        self.update_every = self.config.get("update_every", 50)
        self.save_freq = self.config.get("save_freq", 10)
        self.num_test_episodes = self.config.get("num_test_episodes", 10)
        self.log_dir = self.config.get("log_dir", 'runs_rl/')
        self.whether_her = self.config.get("whether_her", False)
        self.use_automatic_entropy_tuning = self.config.get("use_auto", False)
        self.env_name = self.config.get("env_name", 'planning-v0')
        self.pretrained = self.config.get("pretrained", False)
        self.pretrained_itr = self.config.get("pretrained_itr", None)
        self.pretrianed_dir = self.config.get("pretrained_dir", None)
        self.whether_astar = self.config.get("whether_astar", False)
        self.astar_ablation = self.config.get("astar_ablation", False)
        self.astar_mp_steps = self.config.get("astar_mp_steps", 10)
        self.astar_N_steps = self.config.get("astar_N_steps", 10)
        self.astar_max_iter = self.config.get("astar_max_iter", 5)
        self.astar_heuristic_type = self.config.get("astar_heuristic_type", 'traditional')
        self.whether_dataset = self.config.get("whether_dataset", False)
        self.dataset_path = self.config.get("dataset_path", 'datasets/goal_with_obstacles_info_list.pickle')
        self.env_config = self.config.get("env_config", None)
        self.seed = self.config.get("seed", 0)
        self.replay_size = self.config.get("replay_size", int(1e6))
        self.use_logger = self.config.get("use_logger", True)
        self.device = device
        
        if config['activation'] == 'ReLU':
            activation_fn = nn.ReLU
        elif config['activation'] == 'Tanh':
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {config['activation']}")
        ac_kwargs = {
            "hidden_sizes": tuple(config['hidden_sizes']),
            "activation": activation_fn
        }
        
        self.vehicle_type = config['env_config']['vehicle_type']
        exp_name = self.env_name + '_' + config['algo_name'] + '_' + self.vehicle_type + '_' + str(self.seed) + '_' + get_current_time_format()
        logger_kwargs = {
            'output_dir': config['logging_dir'] + exp_name,
            'output_fname': config['output_fname'],
            'exp_name': exp_name,
        }
        
        
        # TODO: we now take MPI tools out
        if self.use_logger:
            self.logger = EpochLogger(**logger_kwargs)
            # save your configuration in a json file
            self.logger.save_config(locals()) 
        if self.device is None:
            self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Instantiate environment
        self.env, self.test_env = env_fn(self.env_config), env_fn(self.env_config)
        
        if self.vehicle_type == "single_tractor":
            self.number_bounding_box = 1
        elif self.vehicle_type == "one_trailer":
            self.number_bounding_box = 2
        elif self.vehicle_type == "two_trailer":
            self.number_bounding_box = 3
        else:
            self.number_bounding_box = 4
        if self.whether_dataset: #TODO: change the update ways
            # Fank: directly use the data from the datasets
            with open(self.dataset_path, 'rb') as f:
                task_list = pickle.load(f)
            # Update the training data distribution using an existing data file
            self.env.unwrapped.update_task_list(task_list)
            self.test_env.unwrapped.update_task_list(task_list) # TODO: set the same as self.env
        self.state_dim = self.env.observation_space['observation'].shape[0]
        if self.env.unwrapped.observation_type == 'original':
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim,), np.float32)
        elif self.env.unwrapped.observation_type == "lidar_detection" or self.env.unwrapped.observation_type == "one_hot_representation":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + config["perception"]["one_hot_representation"]["number"]*self.number_bounding_box,), np.float32)
        elif self.env.unwrapped.observation_type == "lidar_detection_one_hot":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 9*self.number_bounding_box,), np.float32)
        elif self.env.unwrapped.observation_type == "lidar_detection_one_hot_triple":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 27*self.number_bounding_box,), np.float32)
        elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 98,), np.float32)
        else:
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 4,), np.float32)
        self.obs_dim = self.box.shape
        self.act_dim = self.env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        # Fank: only need to seed action space
        self.env.action_space.seed(self.seed)
        self.test_env.action_space.seed(self.seed)
        
        # Create actor-critic module and target networks
        if self.env.unwrapped.config["with_obstacles_info"]:
            actor_critic = core.AttentionActorCritic # change our model for testing
            self.ac = actor_critic(self.box, self.env.action_space, **ac_kwargs).to(self.device)
        else:
            actor_critic = core.MLPActorCritic
            self.ac = actor_critic(self.box, self.env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = deepcopy(self.ac)
        
        # set up summary writer
        self.exp_name = logger_kwargs['exp_name']
        
        self.save_model_path = self.log_dir  + self.exp_name
        self.writer = SummaryWriter(log_dir=self.save_model_path)
        
        
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        if self.env.unwrapped.config["with_obstacles_info"]:
            self.replay_buffer = ReplayBuffer_With_Obstacles(obs_dim=self.obs_dim, act_dim=self.act_dim, obstacle_dim=4, obstacle_num=10, size=self.replay_size, device=self.device)
        else:
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size, device=self.device)
        
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        if self.use_logger:
            self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        
        # Set up model saving, We don't use this
        # self.logger.setup_pytorch_saver(self.ac)
        
        
        # Fank: automatic tuning for alpha
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
        # Fank: whether using astar as our expert   
        # add a ablation
        if self.astar_ablation:
            self.big_number = int(1e9)
            self.count_unrelated_task = 0
        if self.whether_astar:
            # self.lock = threading.Lock()
            self.add_astar_number = 0
            self.add_astar_trajectory = 0
        self.finish_episode_number = 0
        
        if self.pretrained == True:
            # Fank: use any pretrained model
            itr = str(self.pretrained_itr) if self.pretrained_itr >= 0 else 'final'
            pretrained_file = self.pretrained_dir + 'model_' + itr + '.pth'
            print("Using pretrained model from {}".format(pretrained_file))
            self.load(pretrained_file, whether_load_buffer=False)
            self.ac_targ = deepcopy(self.ac)    
        
        if self.use_logger:
            print("Running off-policy RL algorithm: {}".format(self.algo))
    

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        if self.env.unwrapped.config["with_obstacles_info"]:
            obstacle = data["obstacles"]
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        if self.env.unwrapped.config["with_obstacles_info"]:
            q1 = self.ac.q1(o, obstacle, a)
            q2 = self.ac.q2(o, obstacle, a)
        else:
            q1 = self.ac.q1(o,a)
            q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            # Target Q-values
            if self.env.unwrapped.config["with_obstacles_info"]:
                a2, logp_a2 = self.ac.pi(o2, obstacle)
                q1_pi_targ = self.ac_targ.q1(o2, obstacle, a2)
                q2_pi_targ = self.ac_targ.q2(o2, obstacle, a2)
            else:
                a2, logp_a2 = self.ac.pi(o2)
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
        if self.env.unwrapped.config["with_obstacles_info"]:
            obs = data["obstacles"]
            pi, logp_pi = self.ac.pi(o, obs)
            q1_pi = self.ac.q1(o, obs, pi)
            q2_pi = self.ac.q2(o, obs, pi)
        else:
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
                
    def process_obstacles_list(self, obstacles_list):
        """This is a function that process the previous obstacles_properties_list to a filled obstacles array
        that is used for the model

        Args:
            obstacles_list: list of the obstacles properties(acquired from the env info)

        Returns:
            filled_obstacles: with an added action mask (obstacle_dim + 1, obstacle_num)
        """
        if obstacles_list is not None and len(obstacles_list) > 0:
            obstacles_array = np.array(obstacles_list).T
            obstacle_dim_act, obstacle_count_act = obstacles_array.shape
            filled_obstacles = np.zeros((self.replay_buffer.obstacle_dim + 1, self.replay_buffer.obstacle_num), dtype=np.float32)
            filled_obstacles[:obstacle_dim_act, :obstacle_count_act] = obstacles_array
            # Set the mask row: 1 for existing obstacles, 0 for the rest
            filled_obstacles[obstacle_dim_act, :obstacle_count_act] = 1.0
        else:
            filled_obstacles = np.zeros((self.replay_buffer.obstacle_dim + 1, self.replay_buffer.obstacle_num), dtype=np.float32)
        return filled_obstacles

    def get_action(self, o, info=None, deterministic=False, rgb_image=None):
        if info is None:
            return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), 
                        deterministic)
        else:
            obstacles_list = info["obstacles_properties"]
            
            filled_obstacles = self.process_obstacles_list(obstacles_list)
            return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device),
                               torch.as_tensor(filled_obstacles, dtype=torch.float32).to(self.device),
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
            observation_type = self.env.unwrapped.observation_type
            while not(terminated or truncated):
                # Take deterministic actions at test time
                obs_list = [o['observation'], o['achieved_goal'], o['desired_goal']]
                if observation_type != "original":
                    obs_list.append(o[observation_type])
                action_input = np.concatenate(obs_list)
                if self.env.unwrapped.config.get("with_obstacles_info", False):
                    action = self.get_action(action_input, info, deterministic=True)
                else:
                    action = self.get_action(action_input, deterministic=True)
                o, r, terminated, truncated, info = self.test_env.step(action)
                # if self.env.unwrapped.observation_type == "original" :
                #     o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), deterministic=True))
                # elif self.env.unwrapped.observation_type == "lidar_detection":
                #     o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection']]), deterministic=True))
                # elif self.env.unwrapped.observation_type == "one_hot_representation":
                #     o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation']]), deterministic=True))
                # elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
                #     o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation_enhanced']]), deterministic=True))
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot" and not self.env.unwrapped.config["with_obstacles_info"]:
                #     o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), deterministic=True))
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot" and self.env.unwrapped.config["with_obstacles_info"]:
                #     o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), info, deterministic=True))
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot_triple" and not self.env.unwrapped.config["with_obstacles_info"]:
                #     o, r, terminated, truncated, info = self.test_env.step(self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot_triple']]), deterministic=True))
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
        
    
    def add_results_to_buffer(self, task_list, result_list):
        """
        the task list is a list of all the proposed tasks, of each task is defined as a tuple
        start, goal, obstacles_info, map_vertices, obstacles_properties, map_properties
        result_list is a list of all the results, each result is a dictionary
        
        """
        assert len(task_list) == len(result_list), "The length of task_list and result_list should be the same"
        for j in range(len(task_list)):
            goal_reached = result_list[j].get("goal_reached")
            task_tuple = task_list[j]
            if goal_reached:
                self.add_astar_trajectory += 1
                result_dict = result_list[j]
                obstacles_properties = task_list[j][4] # obstacles_properties
                trajectory_length = len(result_list[j]["control_list"])
                for i in range(trajectory_length):
                    if i == trajectory_length - 1:
                        d = True
                        r =  15
                    else:
                        d = False
                        r = -1
                    a = result_dict["control_list"][i]
                    if self.env.unwrapped.observation_type == "original":
                        o = np.concatenate((result_dict["state_list"][i], result_dict["state_list"][i], task_tuple[1]))
                        o2 = np.concatenate((result_dict["state_list"][i+1], result_dict["state_list"][i+1], task_tuple[1]))
                    else:
                        o = np.concatenate((result_dict["state_list"][i], result_dict["state_list"][i], task_tuple[1], result_dict["perception_list"][i]))
                        o2 = np.concatenate((result_dict["state_list"][i+1], result_dict["state_list"][i+1], task_tuple[1], result_dict["perception_list"][i+1]))
                    if not self.env.unwrapped.config["with_obstacles_info"]:    
                        self.replay_buffer.store(o.astype(np.float32), a.astype(np.float32), r, o2.astype(np.float32), d)
                    else:
                        self.replay_buffer.store(o.astype(np.float32), a.astype(np.float32), r, o2.astype(np.float32), d, obstacles_properties)
        print("Add to Replay Buffer:", self.add_astar_trajectory)
                
                
        
        
    def run(self):
        # Prepare for interaction with environment
        planner_config = {
            "plot_final_path": False,
            "plot_rs_path": False,
            "plot_expand_tree": False,
            "mp_step": self.astar_mp_steps, # Important
            "N_steps": self.astar_N_steps, # Important
            "range_steer_set": 20,
            "max_iter": self.astar_max_iter,
            "heuristic_type": self.astar_heuristic_type,
            "save_final_plot": False,
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
                "safe_metric": 3.0, # the safe distance from the vehicle to obstacle
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
            "acceptance_error": 0.5,
        }
        total_steps = self.steps_per_epoch * self.epochs
        # ep_ret: sum over all the rewards of an episode
        # ep_len: calculate the timesteps of an episode
        # reset this value every time we use run
        self.finish_episode_number = 0
        o, info = self.env.reset(seed=self.seed)
        episode_start_time = time.time()
        if self.whether_astar and not self.astar_ablation:
            self.add_astar_number = 0
            self.add_astar_trajectory = 0
            # to Fasten the code, we first sample the all the start list
            encounter_task_list = []
            # Save the sufficient information for the planner to solve
            encounter_task_list.append((o["achieved_goal"], o["desired_goal"], info["obstacles_info"], info["map_vertices"], info["obstacles_properties"], info["map_properties"]))
            
        ep_ret, ep_len = 0, 0
        # o, ep_ret, ep_len = self.env.reset(), 0, 0
        if self.whether_her:
            trajectory_buffer = []

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                # TODO
                observation_type = self.env.unwrapped.observation_type
                with_obstacles_info = self.env.unwrapped.config.get("with_obstacles_info", False)
                obs_list = [o['observation'], o['achieved_goal'], o['desired_goal']]
                if observation_type != "original":
                    obs_list.append(o[observation_type])
                action_input = np.concatenate(obs_list)
                if with_obstacles_info:
                    a = self.get_action(action_input, info)
                else:
                    a = self.get_action(action_input)
                # if self.env.unwrapped.observation_type == "original" and not self.env.unwrapped.config["with_obstacles_info"]:
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]))
                # elif self.env.unwrapped.observation_type == "original" and self.env.unwrapped.config["with_obstacles_info"]:
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), info)
                # # not working for lidar_detection and one_hot_representation
                # elif self.env.unwrapped.observation_type == "lidar_detection":
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection']]))
                # elif self.env.unwrapped.observation_type == "one_hot_representation":
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation']]))
                # elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation_enhanced']]))
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot" and not self.env.unwrapped.config["with_obstacles_info"]:
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]))
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot" and self.env.unwrapped.config["with_obstacles_info"]:
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), info)
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot_triple" and not self.env.unwrapped.config["with_obstacles_info"]:
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot_triple']]))
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot_triple" and self.env.unwrapped.config["with_obstacles_info"]:
                #     a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot_triple']]), info)
                #     # a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), info)
                
                # # a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['collision_metric']]))
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
            observation_type = self.env.unwrapped.observation_type
            with_obstacles_info = self.env.unwrapped.config.get("with_obstacles_info", False)
            obs_list = [o['observation'], o['achieved_goal'], o['desired_goal']]
            next_obs_list = [o2['observation'], o2['achieved_goal'], o2['desired_goal']]
            if observation_type != "original":
                obs_list.append(o[observation_type])
                next_obs_list.append(o2[observation_type])
            obs_input = np.concatenate(obs_list)
            next_obs_input = np.concatenate(next_obs_list)
            if with_obstacles_info:
                self.replay_buffer.store(obs_input, a, r, next_obs_input, d, info.get("obstacles_properties"))
            else:
                self.replay_buffer.store(obs_input, a, r, next_obs_input, d)
            
            # if self.env.unwrapped.observation_type == "original" and not self.env.unwrapped.config["with_obstacles_info"]:
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal']]), d)
            # elif self.env.unwrapped.observation_type == "original" and self.env.unwrapped.config["with_obstacles_info"]:
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal']]), d, info["obstacles_properties"])
            # elif self.env.unwrapped.observation_type == "lidar_detection":
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection']]), d)
            # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot" and not self.env.unwrapped.config["with_obstacles_info"]:
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection_one_hot']]), d)
            # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot" and self.env.unwrapped.config["with_obstacles_info"]:
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection_one_hot']]), d, info["obstacles_properties"])
            # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot_triple" and not self.env.unwrapped.config["with_obstacles_info"]:
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot_triple']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection_one_hot_triple']]), d)
            # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot_triple" and self.env.unwrapped.config["with_obstacles_info"]:
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot_triple']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection_one_hot_triple']]), d, info["obstacles_properties"])
            # elif self.env.unwrapped.observation_type == "one_hot_representation":
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['one_hot_representation']]), d)
            # elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
            #     self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation_enhanced']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['one_hot_representation_enhanced']]), d)
            
            if self.whether_her:
                
                observation_type = self.env.unwrapped.observation_type
                obs_list = [o['observation'], o['achieved_goal'], o['desired_goal']]
                next_obs_list = [o2['observation'], o2['achieved_goal'], o2['desired_goal']]
                if observation_type != "original":
                    obs_list.append(o[observation_type])
                    next_obs_list.append(o2[observation_type])
                    
                obs_input = np.concatenate(obs_list)
                next_obs_input = np.concatenate(next_obs_list)
                trajectory_buffer.append((obs_input, a, r, next_obs_input, d, info))  
                # # TODO: We try to use the HER to improve the performance
                # if self.env.unwrapped.observation_type == "original":
                #     trajectory_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal']]), d, info))
                # elif self.env.unwrapped.observation_type == "lidar_detection":
                #     trajectory_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection']]), d, info))
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot":
                #     trajectory_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection_one_hot']]), d, info))
                # elif self.env.unwrapped.observation_type == "lidar_detection_one_hot_triple":
                #     trajectory_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot_triple']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['lidar_detection_one_hot_triple']]), d, info))
                # elif self.env.unwrapped.observation_type == "one_hot_representation":
                #     trajectory_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['one_hot_representation']]), d, info))
                # elif self.env.unwrapped.observation_type == "one_hot_representation_enhanced":
                #     trajectory_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['one_hot_representation_enhanced']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['one_hot_representation_enhanced']]), d, info))
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
                        astar_results = Parallel(n_jobs=-1)(delayed(planner.find_astar_trajectory)(task[0], task[1], task[2], task[3], planner_config, self.env.unwrapped.observation_type) for task in encounter_task_list)
                        # astar_results = [planner.find_astar_trajectory(task[0], task[1], task[2], task[3], planner_config, self.env.unwrapped.observation_type) for task in encounter_task_list]
                        # Clear the result
                        self.add_results_to_buffer(encounter_task_list, astar_results)
                        encounter_task_list = []
                elif self.whether_astar and self.astar_ablation:
                    # TODO: not finished
                    if self.finish_episode_number % 50 == 0:
                        print("Start Collecting Buffer from Astar(not related to the episode)")
                        # Clear the result at first
                        encounter_task_list = []
                        for _ in range(20):
                            o, info = self.test_env.reset(seed=(self.big_number + self.count_unrelated_task)) # may need to change to test_env
                            self.count_unrelated_task += 1
                            encounter_task_list.append((o, info["obstacles_info"], info["obstacles_properties"]))
                        astar_results = Parallel(n_jobs=-1)(delayed(find_expert_trajectory_meta)(o, self.vehicle_type) for o in encounter_task_list)
                        
                        # astar_results = [find_expert_trajectory(o, self.vehicle_type) for o in encounter_start_list]
                        self.add_results_to_buffer(astar_results)
                o, info = self.env.reset(seed=(self.seed + t))
                if self.whether_astar and not self.astar_ablation:
                    encounter_task_list.append((o["achieved_goal"], o["desired_goal"], info["obstacles_info"], info["map_vertices"], info["obstacles_properties"], info["map_properties"]))
                    
                episode_start_time = time.time()
                ep_ret, ep_len = 0, 0
                if self.whether_her:
                    # Process Hindsight Experience Replay
                    self.her_process_episode(trajectory_buffer)
                    trajectory_buffer = []
            

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
    
    def process_transition(self, transition, new_goal):
        """for her process unit"""
        o_concatenate, a, r, o2_concatenate, d, info = transition
        new_obs = np.copy(o_concatenate)
        new_obs2 = np.copy(o2_concatenate)
        new_obs[2*self.state_dim: 2*self.state_dim + len(new_goal)] = new_goal
        new_obs2[2*self.state_dim: 2*self.state_dim + len(new_goal)] = new_goal 
        new_reward = self.env.unwrapped.reward(o_concatenate[:self.state_dim], o2_concatenate[:self.state_dim], new_goal)
        new_d = False
        if info["crashed"]:
            new_d = True
            new_reward += self.env.unwrapped.config["collision_penalty"]
        if info["jack_knife"]:
            new_d = True
            new_reward += self.env.unwrapped.config["jack_knife_penalty"]
        if new_reward >= self.env.unwrapped.config["sucess_goal_reward_sparse"]:
            new_d = True
        if self.env.unwrapped.config["with_obstacles_info"]:
            self.replay_buffer.store(new_obs, a, new_reward, new_obs2, new_d, info["obstacles_properties"])
        else:
            self.replay_buffer.store(new_obs, a, new_reward, new_obs2, new_d)
        return new_obs, new_obs2, new_reward, new_d

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
                self.process_transition(transition, new_goal)
        elif goal_selection_strategy == "future":
            for j in range(len(episode_data) - 1):
                transition = episode_data[j]
                try:
                    picked_index = random.sample(range(j + 1, len(episode_data)), k)
                except ValueError:
                    picked_index = list(range(j + 1, len(episode_data)))
                
                new_goals = [episode_data[i][3][:self.state_dim] for i in picked_index]
                for new_goal in new_goals:
                    self.process_transition(transition, new_goal)
        elif goal_selection_strategy == "episode":
            for j in range(len(episode_data)):
                transition = episode_data[j]
                picked_index = random.sample(range(0, len(episode_data)), k)
                new_goals = [episode_data[i][3][:self.state_dim] for i in picked_index]
                
                for new_goal in new_goals:
                    self.process_transition(transition, new_goal)