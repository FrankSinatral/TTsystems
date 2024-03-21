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
        "mp_step": 12,
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

class SAC_ASTAR:
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
                 whether_astar=True,
                 astar_ablation=False,
                 config: dict = None,
                 device = None,
                 args = None):    
        """
        Soft Actor-Critic (SAC) with Astar as Expert Trajectory
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment satisfy the OpenAI Gym API.
                Here we can use the standard parking from highway and tt_envs
                written by Fank.

            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.
                Notice that this epoch is set manually.

            epochs (int): Number of epochs to run and train agent.
                steps_per_epoch * epochs stands for the total timestep.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)
                for tt_envs we choose gamma to be 0.99(notice for the long horizon effect)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)
                Fank: Here we implement an auto alpha

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
                Fank: Random taking actions at the first.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
                Fank: update_start.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
                Fank: You can update_every 4000, meaning after 4000 timesteps
                      you update 4000 sac.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
                Fank: we set num_test_episodes to 10.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.
                Fank: this parameter not used.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

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
                self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 8,), np.float64)
            else:
                self.box = Box(-np.inf, np.inf, (3 * self.state_dim,), np.float64)
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
        # try:
        #     self.logger.log_tabular('LossQ')
        #     self.logger.log_tabular('LossPi')
        # except:
        #     pass
        self.logger.dump_tabular()
        
    def add_expert_trajectory_to_buffer(self, o):
        # Currently not used
        goal = o["desired_goal"]
        input = o["observation"]
        if self.vehicle_type == "one_trailer":
            pack_transition_list = query_hybrid_astar_one_trailer(input, goal)
        elif self.vehicle_type == "three_trailer":
            pack_transition_list = query_hybrid_astar_three_trailer(input, goal)
        if pack_transition_list is None:
            # print("Failed Finding Astar Episode")
            pass
        else:
            print("Astar Episode Length:", len(pack_transition_list))
            with self.lock:
                for transition in pack_transition_list:
                    o, a, o2, r, d = transition
                    self.replay_buffer.store(o, a.astype(np.float32), r, o2, d)
                self.add_astar_number += 1
                print("Add to Replay Buffer:", self.add_astar_number)
        
    
    def add_results_to_buffer(self, results):
        # only add reconstruct for requiring image
        for pack_transition_list in results:
            if pack_transition_list is None:
                pass
            else:
                if self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
                    for transition in pack_transition_list:
                        o, a, o2, r, d = transition
                        o_image = self.env.unwrapped.reconstruct_image_from_observation(o.astype(np.float32))
                        o2_image = self.env.unwrapped.reconstruct_image_from_observation(o2.astype(np.float32))
                        self.replay_buffer.store(o.astype(np.float32), o_image, a.astype(np.float32), r, o2.astype(np.float32), o2_image, d)
                else:
                    for transition in pack_transition_list:
                            o, a, o2, r, d = transition
                            self.replay_buffer.store(o.astype(np.float32), a.astype(np.float32), r, o2.astype(np.float32), d)
            self.add_astar_number += 1
        print("Add to Replay Buffer:", self.add_astar_number)
        
        
    def run(self):
        # Prepare for interaction with environment
        all_start_time = time.time()
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
            encounter_start_list.append(o)
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
            
            
            # # test code
            # # Clear the result at first
            # encounter_start_list = []
            # for j in range(2):
            #     o, _ = self.test_env.reset(seed=j) # may need to change to test_env
            #     encounter_start_list.append(o)
            # print("Start Collecting Buffer from Astar(not related to the episode)")
            # t1 = time.time()
            # # astar_results = Parallel(n_jobs=-1)(delayed(find_expert_trajectory)(o, self.vehicle_type) for o in encounter_start_list)
            # astar_results = [find_expert_trajectory(o, self.vehicle_type) for o in encounter_start_list]
            # self.add_results_to_buffer(astar_results)
            # t2 = time.time()
            # print("done plan:", t2 - t1)
            # break
            
            
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                if self.env_name.startswith("cluttered") and not self.env.unwrapped.config["use_rgb"]:
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]))
                elif self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), rgb_image=o["achieved_rgb_image"])
                else:
                    a = self.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]))
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
            if self.env_name.startswith("cluttered") and not self.env.unwrapped.config["use_rgb"]:
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['obstacles_info']]), d)
            elif self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), o['achieved_rgb_image'], a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['obstacles_info']]), o2["achieved_rgb_image"], d)
            else:
                self.replay_buffer.store(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal']]), d)
            if self.whether_her:
                # currently not used
                if self.env_name.startswith("cluttered") and not self.env.unwrapped.config["use_rgb"]:
                    temp_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['obstacles_info']]), d, info['crashed']))
                elif self.env_name.startswith("cluttered") and self.env.unwrapped.config["use_rgb"]:
                    temp_buffer.append((np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['obstacles_info']]), o["achieved_rgb_image"], a, r, np.concatenate([o2['observation'], o2['achieved_goal'], o2['desired_goal'], o2['obstacles_info']]), o2["achieved_rgb_image"], d, info['crashed']))
                else:
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
                    if self.finish_episode_number % 50 == 0: # put this number smaller
                        print("Start Collecting Buffer from Astar")
                        astar_results = Parallel(n_jobs=-1)(delayed(find_expert_trajectory)(o, self.vehicle_type) for o in encounter_start_list)
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
                            o, _ = self.test_env.reset(seed=(self.big_number + self.count_unrelated_task)) # may need to change to test_env
                            self.count_unrelated_task += 1
                            encounter_start_list.append(o)
                        astar_results = Parallel(n_jobs=-1)(delayed(find_expert_trajectory)(o, self.vehicle_type) for o in encounter_start_list)
                        # astar_results = [find_expert_trajectory(o, self.vehicle_type) for o in encounter_start_list]
                        self.add_results_to_buffer(astar_results)
            o, _ = self.env.reset(seed=(self.seed + t))
            if self.whether_astar and not self.astar_ablation:
                encounter_start_list.append(o)
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