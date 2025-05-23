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
import sys
import pickle
import gc
# import core
# Some of the nn defined here
import rl_agents.bc.core as core
# Try to add logger
from rl_agents.utils.logx import EpochLogger
from utils import planner

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

class AstarPlanningDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class BC:
    def __init__(self, 
                 env_fn,
                 device=None,
                 config: dict = None):    
        """
        Fank: SAC_Astar meta version
        """ 
        ## get all parameters from the config file
        self.algo = "bc"
        self.config = config
        self.steps_per_epoch = self.config.get("steps_per_epoch", 50000)
        self.lr = self.config.get("lr", 1e-3)
        
        
        self.batch_size = self.config.get("batch_size", 1024)
        self.save_freq = self.config.get("save_freq", 10000)
        self.num_test_episodes = self.config.get("num_test_episodes", 100)
        self.log_dir = self.config.get("log_dir", 'runs_rl/')
    
        self.env_name = self.config.get("env_name", 'planning-v0')
        self.pretrained = self.config.get("pretrained", False)
        self.pretrained_itr = self.config.get("pretrained_itr", None)
        self.pretrained_dir = self.config.get("pretrained_dir", None)
        
        
        self.astar_mp_steps = self.config.get("astar_mp_steps", 10)
        self.astar_N_steps = self.config.get("astar_N_steps", 10)
        self.astar_max_iter = self.config.get("astar_max_iter", 5)
        self.astar_heuristic_type = self.config.get("astar_heuristic_type", 'traditional')
        self.astar_dataset_dir = self.config.get("astar_dataset_dir", 'datasets/data/')
    
        self.whether_dataset = self.config.get("whether_dataset", False)
        self.dataset_path = self.config.get("dataset_path", 'datasets/goal_with_obstacles_info_list.pickle')
        self.env_config = self.config.get("env_config", None)
        self.whether_fix_number = self.config.get("fixed_number", False) # fix a typo
        self.number_obstacles = self.env_config["generate_obstacles_config"].get("number_obstacles", 10)
        self.seed = self.config.get("seed", 0)
        self.use_logger = self.config.get("use_logger", True)
        self.pooling_type = self.config.get("pooling_type", "average")
        self.nn_version = self.config.get("nn_version", "1")
        self.device = device
        self.policy_head = self.config.get("policy_head", "gaussian")
        
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
        self.env = env_fn(self.env_config)
        
        # Use a fixed dataset for training and testing
        if self.whether_dataset: #TODO: change the update ways
            # Fank: directly use the data from the datasets
            with open(self.dataset_path, 'rb') as f:
                task_list = pickle.load(f)
            # Update the training data distribution using an existing data file
            self.env.unwrapped.update_task_list(task_list)
        self.observation_type = self.env.unwrapped.observation_type
        self.whether_attention = self.env.unwrapped.config.get("with_obstacles_info", False)
        
        if self.vehicle_type == "single_tractor":
            self.number_bounding_box = 1
        elif self.vehicle_type == "one_trailer":
            self.number_bounding_box = 2
        elif self.vehicle_type == "two_trailer":
            self.number_bounding_box = 3
        else:
            self.number_bounding_box = 4
        
        self.state_dim = self.env.observation_space['observation'].shape[0]
        if self.observation_type == 'original':
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim,), np.float32)
        elif self.observation_type == "lidar_detection_one_hot":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 9*self.number_bounding_box,), np.float32)
        elif self.observation_type == "lidar_detection_one_hot_triple":
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 27*self.number_bounding_box,), np.float32)
        else:
            self.box = Box(-np.inf, np.inf, (3 * self.state_dim + 40,), np.float32)
        self.obs_dim = self.box.shape
        self.act_dim = self.env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        # Fank: only need to seed action space
        self.env.action_space.seed(self.seed)
        
        if self.nn_version == "1":
            # fank version
            print("Choose Transformer Version")
            print("Using policy head: {}".format(self.policy_head))
            self.actor = core.SquashedTransformerActor(state_dim=self.state_dim, goal_dim=self.state_dim, obstacle_dim=4, obstacle_num=10,
                                                            act_dim=self.act_dim, hidden_sizes=tuple(self.config["hidden_sizes"]),
                                                            activation=nn.ReLU, act_limit=self.act_limit, 
                                                            policy_head=self.policy_head).to(self.device)
        elif self.nn_version == "2":
            print("Choose Attention Version")
            print("Using pooling type: {}".format(self.pooling_type))
            print("Using policy head: {}".format(self.policy_head))
            self.actor = core.SquashedAttentionActor(state_dim=self.state_dim, goal_dim=self.state_dim, obstacle_dim=4, obstacle_num=10,
                                                            act_dim=self.act_dim, hidden_sizes=tuple(self.config["hidden_sizes"]),
                                                            activation=nn.ReLU, act_limit=self.act_limit, pooling_type=self.pooling_type,
                                                            policy_head=self.policy_head).to(self.device)  
        elif self.nn_version == "3":
            # rzz version
            self.actor = core.SquashedGaussianMixtureTransformerActorVersion2(state_dim=self.state_dim, goal_dim=self.state_dim, obstacle_dim=4, obstacle_num=10,
                                                            act_dim=self.act_dim, hidden_sizes=tuple(self.config["hidden_sizes"]),
                                                            activation=nn.ReLU, act_limit=self.act_limit, pooling_type=self.pooling_type).to(self.device)
        elif self.nn_version == "4":
            print("choose MLP with perception data")
            self.actor = core.SquashedMLPActor(obs_dim=self.state_dim + 2*self.state_dim + 36*3,
                                                            act_dim=self.act_dim, hidden_sizes=tuple(self.config["hidden_sizes"]),
                                                            activation=nn.ReLU, act_limit=self.act_limit, 
                                                            policy_head=self.policy_head).to(self.device)
        else:
            print("choose New Attention Architecture")
            self.actor = core.SquashedNewAttentionActor(state_dim=self.state_dim, goal_dim=self.state_dim, obstacle_dim=4, obstacle_num=10,
                                                            act_dim=self.act_dim, hidden_sizes=tuple(self.config["hidden_sizes"]),
                                                            activation=nn.ReLU, act_limit=self.act_limit, pooling_type=self.pooling_type,
                                                            policy_head=self.policy_head).to(self.device) 
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        
        
        # set up summary writer
        self.exp_name = logger_kwargs['exp_name']
        
        self.save_model_path = self.log_dir + self.exp_name
        if self.use_logger:
            self.writer = SummaryWriter(log_dir=self.save_model_path)
        
        
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.actor])
        if self.use_logger:
            self.logger.log('\nNumber of parameters: \t actor: %d\n'%var_counts)
        

        
        self.finish_episode_number = 0
        
        if self.pretrained == True:
            # Fank: use any pretrained model
            itr = str(self.pretrained_itr) if self.pretrained_itr >= 0 else 'final'
            pretrained_file = self.pretrained_dir + 'model_' + itr + '.pth'
            print("Using pretrained model from {}".format(pretrained_file))
            self.load_model(pretrained_file)
            
        
        self.planner_config = {
            "plot_final_path": False,
            "plot_rs_path": False,
            "plot_expand_tree": False,
            "mp_step": self.astar_mp_steps,
            "N_steps": self.astar_N_steps,
            "range_steer_set": 20,
            "max_iter": self.astar_max_iter,
            "heuristic_type": self.astar_heuristic_type,
            "save_final_plot": False,
            "controlled_vehicle_config": {
                "w": 2.0,
                "wb": 3.5,
                "wd": 1.4,
                "rf": 4.5,
                "rb": 1.0,
                "tr": 0.5,
                "tw": 1.0,
                "rtr": 2.0,
                "rtf": 1.0,
                "rtb": 3.0,
                "rtr2": 2.0,
                "rtf2": 1.0,
                "rtb2": 3.0,
                "rtr3": 2.0,
                "rtf3": 1.0,
                "rtb3": 3.0,
                "max_steer": 0.6,
                "v_max": 2.0,
                "safe_d": 0.0,
                "safe_metric": 3.0,
                "xi_max": (np.pi) / 4,
            },
            "acceptance_error": 0.5,
        }  
        self.check_planner_config = {
            "plot_final_path": False,
            "plot_rs_path": False,
            "plot_expand_tree": False,
            "mp_step": self.astar_mp_steps,
            "N_steps": self.astar_N_steps,
            "range_steer_set": 20,
            "max_iter": self.astar_max_iter,
            "heuristic_type": "traditional",
            "save_final_plot": False,
            "controlled_vehicle_config": {
                "w": 2.0,
                "wb": 3.5,
                "wd": 1.4,
                "rf": 4.5,
                "rb": 1.0,
                "tr": 0.5,
                "tw": 1.0,
                "rtr": 2.0,
                "rtf": 1.0,
                "rtb": 3.0,
                "rtr2": 2.0,
                "rtf2": 1.0,
                "rtb2": 3.0,
                "rtr3": 2.0,
                "rtf3": 1.0,
                "rtb3": 3.0,
                "max_steer": 0.6,
                "v_max": 2.0,
                "safe_d": 0.0,
                "safe_metric": 3.0,
                "xi_max": (np.pi) / 4,
            },
            "acceptance_error": 0.5,
        }
        
        if self.use_logger:
            print("Running BC algorithm: {}".format(self.algo))
    
    def compute_bc_loss(self, data):
        """compute bc loss and update"""
        if self.nn_version == "4":
            state_goal_with_perception = data["state_goal_with_perception"].to(self.device)
            action = data["action"].to(self.device)
        else:
            state_goal = data["state_goal"].to(self.device)
            filled_obstacles = data["filled_obstacles"].to(self.device)
            action = data["action"].to(self.device)
        self.actor_optimizer.zero_grad()
        if self.nn_version == "4":
            bc_loss = self.actor.compute_bc_loss(state_goal_with_perception, action)
        else:
            bc_loss = self.actor.compute_bc_loss(state_goal, filled_obstacles, action)
        bc_loss.backward()
        self.actor_optimizer.step()
        bc_loss_info = dict(BC_loss=bc_loss.cpu().detach().numpy())
        return bc_loss, bc_loss_info

    
    def get_action(self, o, info=None, deterministic=False):
        if info is None:
            return self.actor.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), 
                        deterministic=deterministic)
        else:
            obstacles_list = info["obstacles_properties"]
            filled_obstacles = self.process_obstacles_list(obstacles_list)
            
            return self.actor.act(torch.as_tensor(o, dtype=torch.float32).to(self.device),
                                torch.as_tensor(filled_obstacles, dtype=torch.float32).to(self.device),
                                deterministic=deterministic)
                
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
            filled_obstacles = np.zeros((4 + 1, 10), dtype=np.float32)
            filled_obstacles[:obstacle_dim_act, :obstacle_count_act] = obstacles_array
            # Set the mask row: 1 for existing obstacles, 0 for the rest
            filled_obstacles[obstacle_dim_act, :obstacle_count_act] = 1.0
        else:
            filled_obstacles = np.zeros((4 + 1, 10), dtype=np.float32)
        return filled_obstacles
    
    def process_obstacles_properties_to_array(self, input_list):
        """process for mlp with obstacles properties"""
        array_length = 40 #TODO change the number of obstacles
        result_array = np.zeros(array_length, dtype=np.float32)
        
        # 将input_list中的元素顺次填入result_array中
        for i, (x, y, l, d) in enumerate(input_list):
            if i >= 10:
                break
            result_array[i*4:i*4+4] = [x, y, l, d]
        
        return result_array
    
    def test_agent(self, global_steps, evaluate_tasks_list=None):
        """the function to test the bc agent"""
        average_ep_ret = 0.0
        average_ep_len = 0.0
        success_rate = 0.0
        jack_knife_rate = 0.0
        crash_rate = 0.0
        if evaluate_tasks_list is None:
            feasible_seed_number = 0
            finish_episode = 0
            while finish_episode < self.num_test_episodes:
                o, info = self.env.reset(seed=feasible_seed_number)
                while (not planner.check_is_start_feasible(o["achieved_goal"], info["obstacles_info"], info["map_vertices"], self.check_planner_config)) or (not self.env.unwrapped.check_goal_with_using_lidar_detection_one_hot()):
                    feasible_seed_number += 1
                    o, info = self.env.reset(seed=feasible_seed_number)
                finish_episode += 1
                feasible_seed_number += 1
                terminated, truncated, ep_ret, ep_len = False, False, 0, 0
                
                while not(terminated or truncated):
                    # Take deterministic actions at test time
                    obs_list = [o['observation'], o['achieved_goal'], o['desired_goal']]
                    if not self.observation_type.startswith("original"):
                        obs_list.append(o[self.observation_type])
                    if self.observation_type == "original_with_obstacles_info":
                        obs_list.append(self.process_obstacles_properties_to_array(info['obstacles_properties']))
                    action_input = np.concatenate(obs_list)
                    if self.nn_version == "4":
                        action = self.get_action(action_input, deterministic=True)
                    else:
                        action = self.get_action(action_input, info, deterministic=True)
                    o, r, terminated, truncated, info = self.env.step(action)
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
                if self.use_logger:
                    self.logger.store(ep_rew_mean=ep_ret, ep_len_mean=ep_len)
        else:
            self.num_test_episodes = len(evaluate_tasks_list)
            for j in range(len(evaluate_tasks_list)):
                tasks_list = [evaluate_tasks_list[j]]
                self.env.unwrapped.update_task_list(tasks_list)
                o, info = self.env.reset()
                terminated, truncated, ep_ret, ep_len = False, False, 0, 0
                
                while not(terminated or truncated):
                    # Take deterministic actions at test time
                    obs_list = [o['observation'], o['achieved_goal'], o['desired_goal']]
                    if not self.observation_type.startswith("original"):
                        obs_list.append(o[self.observation_type])
                    if self.observation_type == "original_with_obstacles_info":
                        obs_list.append(self.process_obstacles_properties_to_array(info['obstacles_properties']))
                    action_input = np.concatenate(obs_list)
                    if self.nn_version == "4":
                        action = self.get_action(action_input, deterministic=True)
                    else:
                        action = self.get_action(action_input, info, deterministic=True)
                    o, r, terminated, truncated, info = self.env.step(action)
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
                if self.use_logger:
                    self.logger.store(ep_rew_mean=ep_ret, ep_len_mean=ep_len)
                
        jack_knife_rate /= self.num_test_episodes
        success_rate /= self.num_test_episodes  
        crash_rate /= self.num_test_episodes 
        average_ep_ret /= self.num_test_episodes
        average_ep_len /= self.num_test_episodes
        if self.use_logger:
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
        
    def process_datasets(self, results):
        """process the datasets to that you want"""
        encounter_task_list = results["tasks"]
        astar_results = results["results"]
        processed_data = []

        for j in range(len(encounter_task_list)):
            goal_reached = astar_results[j].get("goal_reached")
            if goal_reached:
                obstacles_properties_list = encounter_task_list[j][4]
                filled_obstacles = self.process_obstacles_list(obstacles_properties_list)
                goal = encounter_task_list[j][1]
                astar_state_list = astar_results[j]["state_list"]
                astar_action_list = astar_results[j]["control_list"]

                for i in range(0, len(astar_action_list), 10):
                    state_goal = np.concatenate((astar_state_list[i], astar_state_list[i], goal))
                    action = astar_action_list[i]
                    processed_data.append({
                        "state_goal": state_goal,
                        "action": action,
                        "filled_obstacles": filled_obstacles
                    })

        return processed_data
    
    def process_datasets_perception(self, results):
        encounter_task_list = results["tasks"]
        astar_results = results["results"]
        processed_data = []

        for j in range(len(encounter_task_list)):
            goal_reached = astar_results[j].get("goal_reached")
            if goal_reached:
                obstacles_properties_list = encounter_task_list[j][4]
                
                goal = encounter_task_list[j][1]
                astar_state_list = astar_results[j]["state_list"]
                astar_action_list = astar_results[j]["control_list"]
                astar_perception_list = astar_results[j]["perception_list"]
                for i in range(0, len(astar_action_list), 10):
                    state_goal_with_perception = np.concatenate((astar_state_list[i], astar_state_list[i], goal, astar_perception_list[i]))
                    action = astar_action_list[i]
                    processed_data.append({
                        "state_goal_with_perception": state_goal_with_perception,
                        "action": action,
                    })

        return processed_data
    
    def load_datasets(self, file_dir_list, file_pkl_number_list, file_read_complete_list, file_read_index_list):
        all_processed_results = []        
        if not all(file_read_complete_list):
            for j in range(len(file_dir_list)):
                file_dir = file_dir_list[j]
                file_read_index = file_read_index_list[j]
                file_name = os.path.join(file_dir, "astar_result_lidar_detection_one_hot_triple_" + str(file_read_index) + ".pkl")
                if not file_read_complete_list[j]:
                    with open(file_name, 'rb') as f:
                        results = pickle.load(f)
                    if self.nn_version == "4":
                        process_results = self.process_datasets_perception(results)
                    else:
                        process_results = self.process_datasets(results)
                    all_processed_results.extend(process_results)
                    file_read_index_list[j] += 1
                    if file_read_index_list[j] >= file_pkl_number_list[j]:
                        file_read_complete_list[j] = True 
        # convert to tensor
        datasets = []
        for sample in all_processed_results:
            tensor_sample = {key: torch.tensor(value) for key, value in sample.items()}
            datasets.append(tensor_sample)
        del all_processed_results       
        return datasets, file_read_complete_list, file_read_index_list

    def load_training_datasets(self, file_dir_list, read_number=10):
        all_processed_results = []        
        for j in range(len(file_dir_list)):
            file_dir = file_dir_list[j]
            for file_read_index in range(read_number):
                file_name = os.path.join(file_dir, "astar_result_lidar_detection_one_hot_triple_" + str(file_read_index) + ".pkl")
                with open(file_name, 'rb') as f:
                    results = pickle.load(f)
                if self.nn_version == "4":
                    process_results = self.process_datasets_perception(results)
                else:
                    process_results = self.process_datasets(results)
                all_processed_results.extend(process_results)
        # convert to tensor
        datasets = []
        for sample in all_processed_results:
            tensor_sample = {key: torch.tensor(value) for key, value in sample.items()}
            datasets.append(tensor_sample)
        del all_processed_results       
        return datasets
    
    def load_test_datasets(self, file_path):
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        goal_reached_cases = 0
        planning_results = results["results"]
        tasks = results["tasks"]
        evaluate_tasks_list = []
        real_planning_list = []
        for j in range(len(tasks)):
            goal_reached = planning_results[j].get("goal_reached")
            if goal_reached:
                goal_reached_cases += 1
                task_dict = {
                    "goal": tasks[j][1],
                    "obstacles_info": tasks[j][2],
                }
                evaluate_tasks_list.append(task_dict)
                real_planning_list.append(planning_results[j])
            if goal_reached_cases >= 100:
                break
        return evaluate_tasks_list, real_planning_list
    
    def get_file_info(self):
        """Initialize the file information
        - file_dir_list: the directory of the file
        - file_pkl_number_list: the number of the pkl files in that directory
        - file_read_complete_list: whether the directory of the file is read completely
        - file_read_index_list: the index of the file that needs to be read
        """
        file_dir_list = []
        file_pkl_number_list = []
        if self.whether_fix_number:
            file_read_complete_list = [False]
            file_read_index_list = [0]
            file_dir_list.append(os.path.join(self.astar_dataset_dir,"astar_result_obstacle_"+str(self.number_obstacles)+"_pickle"))
        else: 
            file_read_complete_list = [False for _ in range(self.number_obstacles+1)]  
            file_read_index_list = [0 for _ in range(self.number_obstacles+1)]
            for i in range(self.number_obstacles+1):
                file_dir_list.append(os.path.join(self.astar_dataset_dir,"astar_result_obstacle_"+str(i)+"_pickle"))
        for file_dir in file_dir_list:
            files = os.listdir(file_dir)
            file_pkl_number_list.append(len(files))
                
        return file_dir_list, file_pkl_number_list, file_read_complete_list, file_read_index_list
    
                
    def run(self):
        evaluate_tasks_list, _ = self.load_test_datasets("datasets/data/astar_result_obstacle_10_pickle/astar_result_lidar_detection_one_hot_triple_0.pkl")
        # Add read the astar trajectory datasets
        file_dir_list, file_pkl_number_list, file_read_complete_list, file_read_index_list = self.get_file_info()
        
        epochs = 1
        current_step = 0
        for epoch in range(epochs):
            print("Prepare Datasets")
            # datas, file_read_complete_list, file_read_index_list = self.load_datasets(file_dir_list, file_pkl_number_list, file_read_complete_list, file_read_index_list)
            datas = self.load_training_datasets(file_dir_list)
            datasets = AstarPlanningDataset(datas)
            dataloader = torch.utils.data.DataLoader(datasets, batch_size=self.batch_size, shuffle=True)
            print("Finish Prepare Datasets")
            for step in range(self.steps_per_epoch):
                total_loss = 0
                for batch in dataloader:
                    bc_loss, bc_loss_info = self.compute_bc_loss(batch)
                    total_loss += bc_loss_info["BC_loss"]
                average_loss = total_loss / len(dataloader)
                current_step += 1
                if self.use_logger:
                    self.writer.add_scalar('train/BC_loss', average_loss, global_step=current_step)
                print(f"Epoch {epoch + 1}/{epochs}, Step {step + 1}/{self.steps_per_epoch}, Loss: {average_loss}")
                if current_step % self.save_freq == 0:
                    # Test agent at the end of each epoch
                    print("Start testing")
                    self.test_agent(current_step, evaluate_tasks_list)
                    print("Finish testing")
                    if not os.path.exists(self.save_model_path):
                        os.makedirs(self.save_model_path)
                    self.save_model(self.save_model_path +'/model_' + str(current_step) + '.pth')
        self.save_model(self.save_model_path +'/model_final.pth')
    
    def save_model(self, filename):
        state = {'actor_state_dict': self.actor.state_dict(),
                 'actor_optimizer': self.actor_optimizer.state_dict(),
                 }
        torch.save(state, filename)
        
        return filename
    
    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        return filename