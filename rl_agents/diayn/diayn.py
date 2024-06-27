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
import torch.nn.functional as F

# import core
# Some of the nn defined here
import rl_agents.diayn.core as core
# Try to add logger
from rl_agents.utils.logx import EpochLogger
from rl_agents.query_expert import find_expert_trajectory_meta
from utils import planner
import gymnasium as gym
from gymnasium.spaces import Box
import random

from joblib import Parallel, delayed
from datetime import datetime


def get_current_time_format():
    # get current time
    current_time = datetime.now()
    # demo: 20230130_153042
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_time

class DIAYNReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, skill_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.skill_buf = np.zeros(core.combined_shape(size, skill_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        
    def store(self, obs, act, next_obs, done, skill):
        
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.skill_buf[self.ptr] = skill
        # Directly store 'done' without NaN/Inf check, converting boolean to float
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     done=self.done_buf[idxs],
                     skill=self.skill_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

class DIAYN:
    def __init__(self, 
                 env_fn,
                 device=None,
                 config: dict = None):    
        """
        Fank: DIAYN agent
        """ 
        self.algo = "diayn"
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
        self.use_automatic_entropy_tuning = self.config.get("use_auto", False)
        self.env_name = self.config.get("env_name", 'planning-v0')
        self.pretrained = self.config.get("pretrained", False)
        self.pretrained_itr = self.config.get("pretrained_itr", None)
        self.pretrianed_dir = self.config.get("pretrained_dir", None)
        self.whether_dataset = self.config.get("whether_dataset", False)
        self.dataset_path = self.config.get("dataset_path", 'datasets/goal_with_obstacles_info_list.pickle')
        self.env_config = self.config.get("env_config", None)
        self.seed = self.config.get("seed", 0)
        self.replay_size = self.config.get("replay_size", int(1e6))
        self.use_logger = self.config.get("use_logger", True)
        self.device = device
        self.skill_dim = self.config.get("skill_dim", 10)
        self.max_episode_steps = self.config.get("max_episode_steps", 60)
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
        task_list = [{
            "map_vertices": [(-100, -100), (-100, 100), (100, 100), (100, -100)]
        }]
        self.env.unwrapped.update_task_list(task_list)
        self.test_env.unwrapped.update_task_list(task_list)
        if self.whether_dataset: #TODO: change the update ways
            # Fank: directly use the data from the datasets
            with open(self.dataset_path, 'rb') as f:
                task_list = pickle.load(f)
            # Update the training data distribution using an existing data file
            self.env.unwrapped.update_task_list(task_list)
            self.test_env.unwrapped.update_task_list(task_list) # TODO: set the same as self.env
        self.state_dim = self.env.observation_space['observation'].shape[0]
        self.box = Box(-np.inf, np.inf, (self.state_dim,), np.float32)
        self.obs_dim = self.box.shape
        self.act_dim = self.env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        # Fank: only need to seed action space
        self.env.action_space.seed(self.seed)
        self.test_env.action_space.seed(self.seed)
        
        # Create actor-critic module and target networks
        actor_critic = core.MLPActorCritic
        self.ac = actor_critic(self.box, self.env.action_space, self.skill_dim, **ac_kwargs).to(self.device)
        discriminator = core.MLPDiscriminator
        self.ac_targ = deepcopy(self.ac)
        
        self.discriminator = discriminator(self.box.shape[0], self.skill_dim, **ac_kwargs).to(self.device)
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
        self.replay_buffer = DIAYNReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, skill_dim=self.skill_dim, size=self.replay_size, device=self.device)
        
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        if self.use_logger:
            self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.lr)
        
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
        self.finish_episode_number = 0
        
        if self.pretrained == True:
            # Fank: use any pretrained model
            itr = str(self.pretrained_itr) if self.pretrained_itr >= 0 else 'final'
            pretrained_file = self.pretrained_dir + 'model_' + itr + '.pth'
            print("Using pretrained model from {}".format(pretrained_file))
            self.load(pretrained_file, whether_load_buffer=False)
            self.ac_targ = deepcopy(self.ac)   
        self.skill_prior = torch.distributions.Categorical(logits=torch.ones(self.skill_dim).to(self.device)) 
        # self.skill_prior = torch.distributions.Categorical(probs=torch.ones(self.skill_dim)/self.skill_dim) 
        
        if self.use_logger:
            print("Running off-policy RL algorithm: {}".format(self.algo))
    
    def compute_pseudo_reward(self, next_obs, skill):
        """
        next_obs: (batch_size, obs_dim)
        skill: (batch_size, skill_dim)
        output:
        pseudo_reward: (batch_size,)
        """
        with torch.no_grad():
            log_q_phi_z = self.discriminator.log_prob(next_obs, skill) # (batch_size,)
            skill_index = torch.argmax(skill, dim=-1)
            log_p_z = torch.log(self.skill_prior.probs[skill_index])
            pseudo_rewards = log_q_phi_z - log_p_z
        return pseudo_rewards # (batch_size,)
    
    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        
        o, a, o2, d, skill = data['obs'], data['act'], data['obs2'], data['done'], data['skill']
        q1 = self.ac.q1(o, skill, a)
        q2 = self.ac.q2(o, skill, a)
        # Compute the pseudo reward
        r = self.compute_pseudo_reward(o2, skill)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            # Target Q-values
            a2, logp_a2 = self.ac.pi(o2, skill)
            q1_pi_targ = self.ac_targ.q1(o2, skill, a2)
            q2_pi_targ = self.ac_targ.q2(o2, skill, a2)
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
        skill = data['skill']
        pi, logp_pi = self.ac.pi(o, skill)
        q1_pi = self.ac.q1(o, skill, pi)
        q2_pi = self.ac.q2(o, skill, pi)
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

    def compute_loss_discriminator(self, data):
        o, a, o2, d, skill = data['obs'], data['act'], data['obs2'], data['done'], data['skill']
        logits = self.discriminator(o2)
        discriminator_loss = F.cross_entropy(logits, skill.argmax(dim=-1))

        return discriminator_loss
    
    def discriminator_update(self, data, global_step):
        self.discriminator_optimizer.zero_grad()
        loss_discriminator = self.compute_loss_discriminator(data)
        loss_discriminator.backward()
        self.discriminator_optimizer.step()

    def sac_update(self, data, global_step):
        
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

    def get_action(self, o, skill, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), 
                           torch.as_tensor(skill, dtype=torch.float32).to(self.device),
                    deterministic)
    
               

    def run(self):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        # ep_ret: sum over all the rewards of an episode
        # ep_len: calculate the timesteps of an episode
        # reset this value every time we use run
        self.finish_episode_number = 0
        o, info = self.env.reset(seed=self.seed)
        sampled_skill = self.skill_prior.sample()
        # turn to a one-hot vector for saving and model usage
        skill = torch.nn.functional.one_hot(sampled_skill, num_classes=self.skill_dim).cpu().float() # one-hot sensor
        
        episode_start_time = time.time() 
        ep_len = 0
        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                # TODO
                a = self.get_action(o['observation'], np.asarray(skill))
            else:
                a = self.env.action_space.sample()

            # Step the env
            # here is the problem to play with the env
            o2, _, _, _, info = self.env.step(a)
            ep_len += 1
            # reload the terminated and truncated to this setting
            terminated = info["jack_knife"] or info["crashed"]
            truncated = (ep_len == self.max_episode_steps)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            
            d = terminated and (not truncated)

            # Store experience to replay buffer
            self.replay_buffer.store(np.concatenate([o['observation']]), a, np.concatenate([o2['observation']]), d, np.asarray(skill))

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if terminated or truncated:
                episode_end_time = time.time()
                print("Finish Episode time:", episode_end_time - episode_start_time)
                self.finish_episode_number += 1
                print("Finish Episode number:", self.finish_episode_number)
                print("Episode Length:", ep_len)
                o, info = self.env.reset(seed=(self.seed + t))
                # resample a skill
                sampled_skill = self.skill_prior.sample()
                skill = torch.nn.functional.one_hot(sampled_skill, num_classes=self.skill_dim).cpu().float()
                episode_start_time = time.time()
                ep_len = 0
            
            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                print("start update")
                update_start_time = time.time()
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.sac_update(data=batch, global_step=t)
                    self.discriminator_update(data=batch, global_step=t)
                update_end_time = time.time()
                print("done update(update time):", update_end_time - update_start_time)
            
            # This is evaluate step and save model step
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.save(self.save_model_path +'/model_' + str(t) + '.pth')
            
                
        self.save(self.save_model_path +'/model_final.pth')
                
    def save(self, filename):
        state = {'ac_state_dict': self.ac.state_dict(),
                 'alpha': self.alpha,
                 'discriminator': self.discriminator.state_dict(),
                 'pi_optimizer': self.pi_optimizer.state_dict(),
                 'q_optimizer': self.q_optimizer.state_dict(),
                 'alpha_optimizer': self.alpha_optimizer.state_dict(),
                 'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
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
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.alpha = checkpoint['alpha']
        self.pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
        # TODO: this may have to change
        if whether_load_buffer:
            buffer_filename = filename.replace('.pth', '_buffer.pkl')
            with open(buffer_filename, 'rb') as f:
                self.replay_buffer = pickle.load(f)
        
        return filename