from gymnasium import Env
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Optional, Dict
from collections import OrderedDict
import math
import os
import sys
import pprint
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from itertools import chain
import re
import argparse
from typing import Optional
PI = np.pi
from tractor_trailer_envs.vehicles.vehicle_zoo import (
    SingleTractor,
    OneTrailer,
    TwoTrailer,
    ThreeTrailer
)
from tractor_trailer_envs.map_and_obstacles.settings import (
    MapBound,
    QuadrilateralObstacle,
    EllipticalObstacle
)
import tractor_trailer_envs.map_and_obstacles as map_and_obs

        

class TractorTrailerReachingEnv(Env):

    @classmethod
    def default_config(cls) -> dict:
        return {
            "verbose": False, 
            "vehicle_type": "single_tractor",
            "reward_type": 'sparse_reward',
            "act_limit": 1, 
            "distancematrix": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00], # shape not change but can tune
            "reward_weights": [1, 0.3, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "max_episode_steps": 600,
            # "goal": (0, 0, 0, 0, 0, 0), # give with None (6 tuples)
            "evaluate_mode": False, # whether evaluate
            "allow_backward": True, # whether allow backward
            "sucess_goal_reward_parking": -0.12,
            "sucess_goal_reward_sparse": 0,
            "sucess_goal_reward_others": 100, # success reward
            # "continuous_step": False,
            "simulation_freq": 10,#[hz]
            # "using_stable_baseline": False, # whether using stable baseline
            "diff_distance_threshold": 0.1, 
            "potential_reward_threshold": 0.5,
            "sparse_reward_threshold": 0.5,
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
            "outer_wall_bound": {
                "x_min": -100, #[m]
                "x_max": 100,
                "y_min": -100,
                "y_max": 100,
            },
            "start_region_bound": {
                "x_min": 0, #[m]
                "x_max": 0,
                "y_min": 0,
                "y_max": 0,
            },
            "goal_region_bound": {
                "x_min": -50, #[m]
                "x_max": 50,
                "y_min": -50,
                "y_max": 50,
            },
            "goal_list": None,
        }
        
    @staticmethod
    def pi_2_pi(theta):
        while theta >= PI:
            theta -= 2.0 * PI

        while theta < -PI:
            theta += 2.0 * PI
        
        return theta
    
    def __init__(self, config: Optional[dict] = None) -> None:
        """Tractor Trailer Env 
        - param: config: for the whole seeting of tractor trailer env
        - param: args: for the initialization of vehicle and obstacle
        """
        super().__init__()
        self.config = self.default_config()
        self.configure(config)
        
    def define_map(self, dict: Dict[str, float]) -> 'MapBound':
        """using this function to define walls and start region"""
        
        vertices = [(dict["x_min"], dict["y_min"]), (dict["x_min"], dict["y_max"]), (dict["x_max"], dict["y_min"]), (dict["x_max"], dict["y_max"])]
        return MapBound(vertices)
    
    def check_map(self, outer_wall_dict: Dict[str, float], start_region_dict: Dict[str, float]) -> bool:
        outer_wall_value_list = [a for _, a in outer_wall_dict.items()]
        start_region_value_list = [b for _, b in start_region_dict.items()]
        return all(item2 > item1 if index % 2 == 0 else item2 < item1 
             for index, (item1, item2) in enumerate(zip(outer_wall_value_list, start_region_value_list)))

        
    
    def define_observation_space(self) -> None:
        # observation_space (all set to 6-dim space)
        achieved_goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        desired_goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        
        self.observation_space = spaces.Dict({
            'achieved_goal': achieved_goal_space,
            'desired_goal': desired_goal_space,
            'observation': observation_space
        })
        
    def define_action_space(self):
        # action_space
        action_low = np.array(
            [-self.config["act_limit"], 
             -self.config["act_limit"]], 
            dtype=np.float32)
        action_high = np.array(
            [self.config["act_limit"], 
             self.config["act_limit"]], 
            dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high)
        
    def configure(self, config: Optional[dict]) -> None:
        
        if config:
            self.config.update(config)
        assert self.check_map(self.config["outer_wall_bound"],
                        self.config["start_region_bound"]), "invalid map define"
        assert self.check_map(self.config["outer_wall_bound"],
                        self.config["goal_region_bound"]), "invalid map define"
        if self.config["verbose"]:
            GREEN_BOLD = "\033[1m\033[32m"
            RESET = "\033[0m"
            print(GREEN_BOLD, end="")
            pprint.pprint(config)
            print(RESET, end="")
        self.reward_type = self.config["reward_type"]
        self.start = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.sx, self.sy, self.syaw0, self.syawt1, self.syawt2, self.syawt3 = self.start
        self.map = self.define_map(self.config["outer_wall_bound"])
        self.start_region = self.define_map(self.config["start_region_bound"])
        self.goal_region = self.define_map(self.config["goal_region_bound"])
        self.vehicle_type = self.config["vehicle_type"]
        # give all the parameters through C_three_trailer
        if self.vehicle_type == "single_tractor":
            self.controlled_vehicle = SingleTractor(self.config["controlled_vehicle_config"])
        elif self.vehicle_type == "one_trailer":
            self.controlled_vehicle = OneTrailer(self.config["controlled_vehicle_config"])
        elif self.vehicle_type == "two_trailer":
            self.controlled_vehicle = TwoTrailer(self.config["controlled_vehicle_config"])
        else:
            self.controlled_vehicle = ThreeTrailer(self.config["controlled_vehicle_config"])
        self.yawmax = np.pi
        self.define_observation_space()
        self.define_action_space()
        
        
        # Optional Parameters
        self.distancematrix = np.diag(self.config["distancematrix"])
        # self.goal = self.config["goal"] # 6-tuples (can't use it to calculate reward)
        # self.gx, self.gy, self.gyaw0, self.gyawt1, self.gyawt2, self.gyawt3 = self.goal
        
        self.dt = 1 / self.config["simulation_freq"]
        self.act_limit = self.config["act_limit"]
        self.evaluate_mode = self.config['evaluate_mode']
             
                      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def update_goal_list(self, goal_list):
        self.config["goal_list"] = goal_list
        
    def sample_from_space(self, **kwargs):
        if 'seed' in kwargs:
            self.seed(kwargs['seed'])
            np.random.seed(kwargs['seed'])
        x_coordinates = self.np_random.uniform(self.config["goal_region_bound"]["x_min"], self.config["goal_region_bound"]["x_max"])
        y_coordinates = self.np_random.uniform(self.config["goal_region_bound"]["y_min"], self.config["goal_region_bound"]["y_max"])
        yaw_state = self.np_random.uniform(-self.yawmax, self.yawmax)
        vehicle = copy.deepcopy(self.controlled_vehicle)
        vehicle.reset_equilibrium(x_coordinates, y_coordinates,yaw_state)
        sample_goal = tuple(vehicle.observe())
        return sample_goal
    
    
    def reset(self, **kwargs):
        # 6-dim
        if 'seed' in kwargs:
            self.seed(kwargs['seed'])
            np.random.seed(kwargs['seed'])
        if self.config["goal_list"] is not None:
            number_goals = len(self.config["goal_list"])
            selected_index = np.random.randint(0, number_goals)
            self.goal = self.config["goal_list"][selected_index]
        else:
            # random sample a goal
            x_coordinates = self.np_random.uniform(self.config["goal_region_bound"]["x_min"], self.config["goal_region_bound"]["x_max"])
            y_coordinates = self.np_random.uniform(self.config["goal_region_bound"]["y_min"], self.config["goal_region_bound"]["y_max"])
            yaw_state = self.np_random.uniform(-self.yawmax, self.yawmax)
            self.controlled_vehicle.reset_equilibrium(x_coordinates, y_coordinates,yaw_state)
            self.goal = tuple(self.controlled_vehicle.observe())
        # self.goal = [x_coordinates, y_coordinates]
        self.gx, self.gy, self.gyaw0, self.gyawt1, self.gyawt2, self.gyawt3 = self.goal
        # for _ in range(4):
        #     yaw_state = self.np_random.uniform(-self.yawmax, self.yawmax)
        #     if yaw_state == self.yawmax:
        #         yaw_state = -self.yawmax
        #     self.goal.append(yaw_state)
        
        # shape the self.state to desired dim
        self.controlled_vehicle.reset_equilibrium(self.sx, self.sy, self.syaw0)
        ox, oy = self.map.sample_surface(0.1)
        if self.controlled_vehicle.is_collision(ox, oy):
            # best not meet this case
            self.terminated = True
        else:
            self.terminated = False
        # Reset step counters and other stateful variables if needed
        self.current_step = 0
        self.truncated = False
        if self.evaluate_mode:
            self.state_list = [self.controlled_vehicle.observe()]
            self.action_list = []
            
        obs_dict = OrderedDict([
            ('observation', self.controlled_vehicle.observe()),
            ("achieved_goal", self.controlled_vehicle.observe()),
            ("desired_goal", np.array(self.goal, dtype=np.float64))
        ])
        
        info_dict = {
            "crashed": False,
            "is_success": False,
            "jack_knife": False,
            "action": None,
            "old_state": None,
        }
            
        
        return obs_dict, info_dict
    
    def diff_distance_reward(self, state, state_, goal=None):
        if goal is None:
            goal = np.array([self.goal], dtype=np.float64)
        # broadcast
        state_diff = state - goal
        new_state_diff = state_ - goal
        weighted_distance = np.dot(np.dot(state_diff, self.distancematrix), state_diff.T).item()
        new_weighted_distance = np.dot(np.dot(new_state_diff, self.distancematrix), new_state_diff.T).item()
        
        if new_weighted_distance < self.config["diff_distance_threshold"]:
            reward = 100
        elif new_weighted_distance >= weighted_distance:
            reward = -1
        else:
            reward = 1
            
        return reward
    
    def potential_reward(self, state, state_, goal=None):
        """
        potential reward: the idea is from the paper
        https://ieeexplore.ieee.org/abstract/document/9756640
        """
        # state/ state_: 2-dim np_array
        if goal is None:
            goal = np.array(self.goal, dtype=np.float64)
        state_diff = state - goal
        new_state_diff = state_ - goal
        # using different potential in different reward
        if self.vehicle_type == "single_tractor":
            statePotential = (abs(state_diff[0]) + abs(state_diff[1]) + 
                self.controlled_vehicle.WB * (abs(math.sin(state_diff[2])) + 1 - math.cos(state_diff[2]))
            )
            state_Potential = (abs(new_state_diff[0]) + abs(new_state_diff[1]) + 
                self.controlled_vehicle.WB * (abs(math.sin(new_state_diff[2])) + 1 - math.cos(new_state_diff[2]))
            )
            if state_Potential < self.config["potential_reward_threshold"]:
                return 100
        elif self.vehicle_type == "one_trailer":
            statePotential = (abs(state_diff[0]) + abs(state_diff[1]) + 
                self.controlled_vehicle.WB * (abs(math.sin(state_diff[2])) + 1 - math.cos(state_diff[2])) +
                self.controlled_vehicle.RTR * (abs(math.sin(state_diff[3])) + 1 - math.cos(state_diff[3]))
            )
            state_Potential = (abs(new_state_diff[0]) + abs(new_state_diff[1]) + 
                self.controlled_vehicle.WB * (abs(math.sin(new_state_diff[2])) + 1 - math.cos(new_state_diff[2])) +
                self.controlled_vehicle.RTR * (abs(math.sin(new_state_diff[3])) + 1 - math.cos(new_state_diff[3]))
            )
            if state_Potential < self.config["potential_reward_threshold"]:
                return 100
        elif self.vehicle_type == "two_trailer":
            statePotential = (abs(state_diff[0]) + abs(state_diff[1]) + 
                self.controlled_vehicle.WB * (abs(math.sin(state_diff[2])) + 1 - math.cos(state_diff[2])) +
                self.controlled_vehicle.RTR * (abs(math.sin(state_diff[3])) + 1 - math.cos(state_diff[3])) +
                self.controlled_vehicle.RTR2 * (abs(math.sin(state_diff[4])) + 1 - math.cos(state_diff[4]))
            )
            state_Potential = (abs(new_state_diff[0, 0]) + abs(new_state_diff[0, 1]) + 
                self.controlled_vehicle.WB * (abs(math.sin(new_state_diff[2])) + 1 - math.cos(new_state_diff[2])) +
                self.controlled_vehicle.RTR * (abs(math.sin(new_state_diff[3])) + 1 - math.cos(new_state_diff[3])) +
                self.controlled_vehicle.RTR2 * (abs(math.sin(new_state_diff[4])) + 1 - math.cos(new_state_diff[4]))
            )
            if state_Potential < self.config["potential_reward_threshold"]:
                return 100
        else:
            statePotential = (abs(state_diff[0]) + abs(state_diff[1]) + 
                self.controlled_vehicle.WB * (abs(math.sin(state_diff[2])) + 1 - math.cos(state_diff[2])) +
                self.controlled_vehicle.RTR * (abs(math.sin(state_diff[3])) + 1 - math.cos(state_diff[3])) +
                self.controlled_vehicle.RTR2 * (abs(math.sin(state_diff[4])) + 1 - math.cos(state_diff[4])) +
                self.controlled_vehicle.RTR3 * (abs(math.sin(state_diff[5])) + 1 - math.cos(state_diff[5]))
            )
            state_Potential = (abs(new_state_diff[0]) + abs(new_state_diff[1]) + 
                self.controlled_vehicle.WB * (abs(math.sin(new_state_diff[2])) + 1 - math.cos(new_state_diff[2])) +
                self.controlled_vehicle.RTR * (abs(math.sin(new_state_diff[3])) + 1 - math.cos(new_state_diff[3])) +
                self.controlled_vehicle.RTR2 * (abs(math.sin(new_state_diff[4])) + 1 - math.cos(new_state_diff[4])) +
                self.controlled_vehicle.RTR3 * (abs(math.sin(new_state_diff[5])) + 1 - math.cos(new_state_diff[5]))
            )
            if state_Potential < self.config["potential_reward_threshold"]:
                return 100
            
        return statePotential - state_Potential
    
    def parking_reward(self, state, state_, p: float = 0.5, goal=None):
        """
        This reward borrows the idea from highway_env parking reward
        the only change is that we take out the crashed penalty
        """
        if goal is None:
            desired_goal = np.array(
                [
                    self.goal[0], 
                    self.goal[1], 
                    math.cos(self.goal[2]), 
                    math.sin(self.goal[2]), 
                    math.cos(self.goal[3]), 
                    math.sin(self.goal[3]), 
                    math.cos(self.goal[4]), 
                    math.sin(self.goal[4]),
                    math.cos(self.goal[5]), 
                    math.sin(self.goal[5]),
                ],
                dtype=np.float64
            )
        else:
            desired_goal = np.array(
                [
                    goal[0], 
                    goal[1], 
                    math.cos(goal[2]), 
                    math.sin(goal[2]), 
                    math.cos(goal[3]), 
                    math.sin(goal[3]), 
                    math.cos(goal[4]), 
                    math.sin(goal[4]),
                    math.cos(goal[5]), 
                    math.sin(goal[5]),
                ],
                dtype=np.float64
            )
        achieved_goal = np.array(
            [
                state[0], 
                state[1], 
                math.cos(state[2]), 
                math.sin(state[2]), 
                math.cos(state[3]), 
                math.sin(state[3]),
                math.cos(state[4]), 
                math.sin(state[4]),
                math.cos(state[5]), 
                math.sin(state[5]),
                ],
            dtype=np.float64
        )
        achieved_goal_ = np.array(
            [
                state_[0], 
                state_[1], 
                math.cos(state_[2]), 
                math.sin(state_[2]), 
                math.cos(state_[3]), 
                math.sin(state_[3]),
                math.cos(state_[4]), 
                math.sin(state_[4]),
                math.cos(state_[5]), 
                math.sin(state_[5]),
            ],
            dtype=np.float64
        )
        reward = -np.power(np.dot(np.abs(achieved_goal_ - desired_goal), np.array(self.config["reward_weights"])), p)
        return reward
    
    def sparse_reward(self, state, state_, goal=None):
        if goal is None:
            goal = np.array([self.goal], dtype=np.float64)
        # broadcast
        new_state_diff = state_ - goal
        new_weighted_distance = np.dot(np.dot(new_state_diff, self.distancematrix), new_state_diff.T).item()
        
        if new_weighted_distance < self.config["sparse_reward_threshold"]:
            reward = 0
        else:
            reward = -1
            
        return reward
    
    def reward(self, old_state, state, goal=None):
        if self.reward_type == "diff_distance":
            reward = self.diff_distance_reward(old_state, state, goal)
        elif self.reward_type == "parking_reward":
            reward = self.parking_reward(np.squeeze(old_state), np.squeeze(state), goal)
        elif self.reward_type == "potential_reward":
            reward = self.potential_reward(old_state, state, goal)
        elif self.reward_type == "sparse_reward":
            reward = self.sparse_reward(old_state, state, goal)
        return reward
    
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Recalculate reward for HER replay buffer
        """
        rewards = []
        for j in range(achieved_goal.shape[0]):  
            if self.reward_type == "diff_distance":
                reward = self.diff_distance_reward(info[j]["old_state"], achieved_goal[j], desired_goal[j])
            elif self.reward_type == "parking_reward":
                reward = self.parking_reward(info[j]["old_state"], achieved_goal[j], desired_goal[j])
            elif self.reward_type == "potential_reward":
                reward = self.potential_reward(info[j]["old_state"], achieved_goal[j], desired_goal[j])
            elif self.reward_type == "sparse_reward":
                reward = self.sparse_reward(info[j]["old_state"], achieved_goal[j], desired_goal[j])
            rewards.append(reward)
        
        return np.array(rewards)
    
    
    # implement tt system here
    def step(self, action):
            
        old_state = self.controlled_vehicle.observe()
        # clip action
        action_clipped = np.clip(action, -self.act_limit, self.act_limit)
        self.controlled_vehicle.step(action_clipped, self.dt, self.config["allow_backward"])
        
        # action_direction = 1 if action_clipped[0] >= 0 else -1
            
        state = self.controlled_vehicle.observe()
                            
        # convert state to a 2-dim matrix
        # state = np.array([self.state], dtype=np.float32)
        # choosing how to calculate reward using reward_type
        reward = self.reward(old_state, state)
        
        info_dict = {
            "crashed": False,
            "is_success": False,
            "jack_knife": False,
            "action": action_clipped,
            "old_state": old_state,
        }
        
        # check if success
        if self.reward_type == "parking_reward":
            if reward >= self.config['sucess_goal_reward_parking']:
                self.terminated = True
                info_dict['is_success'] = True
        elif self.reward_type == "sparse_reward":
            if reward >= self.config['sucess_goal_reward_sparse']:
                self.terminated = True
                info_dict['is_success'] = True
        else:
            if reward >= self.config['sucess_goal_reward_others']:
                self.terminated = True
                info_dict['is_success'] = True
                
        ox, oy = self.map.sample_surface(0.1)
        ox, oy = map_and_obs.remove_duplicates(ox, oy)
        
        if self.controlled_vehicle.is_collision(ox, oy):
            self.terminated = True
            info_dict['crashed'] = True
            
        if self.controlled_vehicle._is_jack_knife():
            self.terminated = True
            info_dict['jack_knife'] = True
            
            
        # check truncated
        self.current_step += 1
        if self.current_step >= self.config["max_episode_steps"]:
            self.truncated = True
        
        if self.evaluate_mode:
            self.state_list.append(self.controlled_vehicle.observe())
            self.action_list.append(action_clipped)
            
        obs_dict = OrderedDict([
            ('observation', self.controlled_vehicle.observe()),
            ("achieved_goal", self.controlled_vehicle.observe()),
            ("desired_goal", np.array(self.goal, dtype=np.float64))
        ])
        
        
        # if self.config["using_stable_baseline"]:
        #     return obs_dict, reward, self.terminated or self.truncated, info_dict
        # else:
        return obs_dict, reward, self.terminated, self.truncated, info_dict

    def render(self, mode='human'):
        assert self.evaluate_mode
        plt.cla()
        ax = plt.gca()
        ox, oy = self.map.sample_surface(0.1)
        ox, oy = map_and_obs.remove_duplicates(ox, oy)
        ox_, oy_ = self.goal_region.sample_surface(1)
        ox_, oy_ = map_and_obs.remove_duplicates(ox_, oy_)
        plt.plot(ox, oy, 'sk', markersize=1)
        plt.plot(ox_, oy_, 'sk', markersize=0.5)
        self.controlled_vehicle.plot(ax, self.action_list[-1], 'blue')
        plt.axis('equal')
        
    def plot_exploration(self, i, goals_list):
        plt.cla()
        ax = plt.gca()
        ox, oy = self.map.sample_surface(0.1)
        ox, oy = map_and_obs.remove_duplicates(ox, oy)
        plt.plot(ox, oy, 'sk', markersize=1)
        self.plot_vehicle = copy.deepcopy(self.controlled_vehicle)
        real_dim = len(self.plot_vehicle.state)
        for goal in goals_list:
            goal = goal[:real_dim]
            self.plot_vehicle.reset(*goal)
            self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue')
        start = np.array(self.start)[:real_dim]
        self.plot_vehicle.reset(*start)
        self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        plt.axis('equal')
        if not os.path.exists('curriculum_vis/visit/'):
            os.makedirs('curriculum_vis/visit/')
        plt.savefig('curriculum_vis/visit/visit_region_{}.png'.format(i))
    
    def plot_goal(self, i):
        plt.cla()
        ax = plt.gca()
        ox, oy = self.map.sample_surface(0.1)
        ox, oy = map_and_obs.remove_duplicates(ox, oy)
        plt.plot(ox, oy, 'sk', markersize=1)
        self.plot_vehicle = copy.deepcopy(self.controlled_vehicle)
        real_dim = len(self.plot_vehicle.state)
        for goal in self.config["goal_list"]:
            goal = goal[:real_dim]
            self.plot_vehicle.reset(*goal)
            self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'red')
        start = np.array(self.start)[:real_dim]
        self.plot_vehicle.reset(*start)
        self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        plt.axis('equal')
        if not os.path.exists('curriculum_vis/goals/'):
            os.makedirs('curriculum_vis/goals/')
        plt.savefig('curriculum_vis/goals/training_goals_{}.png'.format(i))        
        
    def run_simulation(self, save_dir=None):
        assert self.evaluate_mode # in case that you use the function not correctly
        
        from matplotlib.animation import FuncAnimation, PillowWriter
        ox, oy = self.map.sample_surface(0.1)
        ox, oy = map_and_obs.remove_duplicates(ox, oy)
        ox_, oy_ = self.goal_region.sample_surface(1)
        ox_, oy_ = map_and_obs.remove_duplicates(ox_, oy_)
        start_state = self.state_list[0]
        gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = self.goal
        if self.vehicle_type == "single_tractor":
            real_dim = 3
            gyawt1, gyawt2, gyawt3 = None, None, None
        elif self.vehicle_type == "one_trailer":
            real_dim = 4
            gyawt2, gyawt3 = None, None
        elif self.vehicle_type == "two_trailer":
            real_dim = 5
            gyawt3 = None
        else:
            real_dim = 6
            
        
        start_state_to_list = list(start_state)
        start_state_to_list[real_dim:] = [None] * (6 - real_dim)
        sx, sy, syaw0, syawt1, syawt2, syawt3 = start_state_to_list
        
        pathx, pathy, pathyaw0 = [], [], []
        pathyawt1, pathyawt2, pathyawt3 = [], [], []
        # action0, action1 = [], []
        for state in self.state_list:
            state_to_list = list(state)
            state_to_list[real_dim:] = [None] * (6 - real_dim)
            x, y, yaw0, yawt1, yawt2, yawt3 = state_to_list
            pathx.append(x)
            pathy.append(y)
            pathyaw0.append(yaw0)
            pathyawt1.append(yawt1)
            pathyawt2.append(yawt2)
            pathyawt3.append(yawt3)
        
        # for action in self.action_list:
        #     action0.append(action[0])
        #     action1.append(action[1])
            
        fig, ax = plt.subplots()
        def update(num):
            ax.clear()
            plt.axis("equal")
            k = num
            # plot env (obstacle)
            # if dash_area.any():
            #     ax.add_patch(rect)
            plt.plot(ox, oy, "sk", markersize=1)
            plt.plot(ox_, oy_, "sk", markersize=0.5)
            self.controlled_vehicle.reset(sx, sy, syaw0, syawt1, syawt2, syawt3)
            
            self.controlled_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue')
            # TODO
            self.controlled_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
            self.controlled_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
            # plot the planning path
            plt.plot(pathx[:k], pathy[:k], linewidth=1.5, color='r')
            self.controlled_vehicle.reset(pathx[k], pathy[k], pathyaw0[k],pathyawt1[k], pathyawt2[k], pathyawt3[k])
            try:
                self.controlled_vehicle.plot(ax, self.action_list[k], 'black')
            except:
                self.controlled_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
            

        ani = FuncAnimation(fig, update, frames=len(pathx), repeat=True)

        # Save the animation
        writer = PillowWriter(fps=20)
        if not save_dir:
            if not os.path.exists("./rl_training/gif/tt"):
                os.makedirs("./rl_training/gif/tt")
                
            base_path = "./rl_training/gif/tt/path_simulation"
            extension = ".gif"
            
            all_files = os.listdir("./rl_training/gif/tt")
            matched_files = [re.match(r'path_simulation(\d+)\.gif', f) for f in all_files]
            numbers = [int(match.group(1)) for match in matched_files if match]
            
            if numbers:
                save_index = max(numbers) + 1
            else:
                save_index = 1
            ani.save(base_path + str(save_index) + extension, writer=writer) 
        else:
            ani.save(save_dir, writer=writer)