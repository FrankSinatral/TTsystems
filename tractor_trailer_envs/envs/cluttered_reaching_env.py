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
from copy import deepcopy
from PIL import Image
import io
        
def is_rectangle_intersect(rect1, rect2):
    # check whether two rectangles intersect with each other
    # Assume that the edge is parallel to the axis
    # format：[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # can take out the asumptions
    x1_min, y1_min = min(pt[0] for pt in rect1), min(pt[1] for pt in rect1)
    x1_max, y1_max = max(pt[0] for pt in rect1), max(pt[1] for pt in rect1)
    x2_min, y2_min = min(pt[0] for pt in rect2), min(pt[1] for pt in rect2)
    x2_max, y2_max = max(pt[0] for pt in rect2), max(pt[1] for pt in rect2)
    
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def cyclic_angle_distance(angle1, angle2):
    """计算两个角度之间的周期性距离"""
    diff = np.abs(angle1 - angle2)
    return min(diff, 2*np.pi - diff)

def mixed_norm(goal, final_state):
    # 计算位置分量的平方差
    position_diff_square = np.sum((goal[:2] - final_state[:2]) ** 2)
    
    # 计算角度分量的周期性平方差
    angle_diff_square = sum([cyclic_angle_distance(goal[i], final_state[i]) ** 2 for i in range(2, 6)])
    
    # 计算总距离的平方根
    total_distance = np.sqrt(position_diff_square + angle_diff_square)
    return total_distance


class TractorTrailerClutteredReachingEnv(Env):

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
            "sucess_goal_reward_sparse": 0,
            # "continuous_step": False,
            "simulation_freq": 10,#[hz]
            # "using_stable_baseline": False, # whether using stable baseline
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
                "x_min": -30, #[m]
                "x_max": 30,
                "y_min": -30,
                "y_max": 30,
            },
            "start_region_bound": {
                "x_min": 0, #[m]
                "x_max": 0,
                "y_min": 0,
                "y_max": 0,
            },
            "goal_region_bound": {
                "x_min": -30, #[m]
                "x_max": 30,
                "y_min": -30,
                "y_max": 30,
            },
            "goal_with_obstacles_info_list": None, # goal must be given together with obstacles_info as a list(each element a dict)
            "start_list": None,
            "N_steps": 10, # number of steps for each action
            "jack_knife_penalty": 0,
            "collision_penalty": 0,
            "use_rgb": True, # whether use rgb image as an observation
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
        achieved_goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        desired_goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # TODO: here we set the dimension of the obstacle info manually, in the future, we can make it more general
        self.observation_space = spaces.Dict({
            'achieved_goal': achieved_goal_space,
            'desired_goal': desired_goal_space,
            'observation': observation_space,
            'obstacles_info': spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32), # fixed two obstacles
            'achieved_rgb_image': spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8),
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
        # assert self.check_map(self.config["outer_wall_bound"],
        #                 self.config["goal_region_bound"]), "invalid map define"
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
        # In this env, goal region is the same as the outer wall
        self.goal_region = self.define_map(self.config["goal_region_bound"])
        
        # pick car type
        self.vehicle_type = self.config["vehicle_type"]
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
        
        self.dt = 1 / self.config["simulation_freq"]
        self.act_limit = self.config["act_limit"]
        self.evaluate_mode = self.config['evaluate_mode']
        # add a white image
        self.white_image = np.full((3, 84, 84), 255, dtype=np.uint8)   
                      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def update_goal_with_obstacles_info_list(self, goal_with_obstales_info_list):
        # the list must be given that the elment of the list is a dict
        self.config["goal_with_obstacles_info_list"] = goal_with_obstales_info_list
        
    def clear_goal_with_obstacles_info_list(self):
        self.config["goal_with_obstacles_info_list"] = None
    
        
    def sample_from_space(self, **kwargs):
        # currently not used
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

    def generate_random_rectangle(self, frame):
        """random generate a rectangle, make sure it is inside the given frame and the two pairs of opposite edges are not equal
        frame format: [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        return: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        x_min, x_max = frame[0][0], frame[2][0]
        y_min, y_max = frame[0][1], frame[1][1]
        
        # Ensure that the generated rectangle has two pairs of opposite edges that are not equal
        while True:
            x1, x2 = sorted([self.np_random.uniform(x_min, x_max) for _ in range(2)])
            if abs(x1 - x2) > 2:
                break

        while True:
            y1, y2 = sorted([self.np_random.uniform(y_min, y_max) for _ in range(2)])
            if abs(y1 - y2) > 2:
                break
        
        return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    
    def generate_fixed_size_random_rectangle(self, frame, existing_rectangles, length_x=11.5, length_y=2):
        """
        generate a fixed size rectange, make sure it is inside the given frame and does not intersect with existing rectangles
        frame format: [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        existing_rectangles format: [[(x1, y1), (x2, y2), (x3, y3), (x4, y4)], ...]
        length_x: the length of the rectangle along the x axis
        length_y: the length of the rectangle along the y axis
        """
        while True:  # find until finding a non-intersecting rectangle
            min_x, max_x = frame[0][0] + length_x / 2, frame[2][0] - length_x / 2
            min_y, max_y = frame[0][1] + length_y / 2, frame[1][1] - length_y / 2

            center_x = self.np_random.uniform(min_x, max_x)
            center_y = self.np_random.uniform(min_y, max_y)

            new_rect = [
                (center_x - length_x / 2, center_y - length_y / 2),
                (center_x - length_x / 2, center_y + length_y / 2),
                (center_x + length_x / 2, center_y + length_y / 2),
                (center_x + length_x / 2, center_y - length_y / 2)
            ]

            # check whether the new rectangle intersects with existing rectangles
            if all(not is_rectangle_intersect(new_rect, rect) for rect in existing_rectangles):
                return new_rect  # if not intersecting, return the new rectangle

    def get_equilbrium_configuration(self, rect):
        # return the corresponding configuration of the 3-tt vehicle
        # contains two configuration of the tractor trailer vehicle
        results = []
        x_min, x_max = min(pt[0] for pt in rect), max(pt[0] for pt in rect)
        y_min, y_max = min(pt[1] for pt in rect), max(pt[1] for pt in rect)
        length_x = x_max - x_min
        length_y = y_max - y_min
        if length_x == 2:
            # shufang
            state_x = (x_max + x_min) / 2
            state_y = y_max - 4.5
            state_yaw = (np.pi) / 2
            # 90
            results.append(np.array([state_x, state_y, state_yaw, state_yaw, state_yaw, state_yaw]))
            state_y = y_min + 4.5
            state_yaw = -(np.pi) / 2
            results.append(np.array([state_x, state_y, state_yaw, state_yaw, state_yaw, state_yaw]))
        else:  
            state_y = (y_max + y_min) / 2
            state_x = x_max - 4.5
            state_yaw = 0
            # 0
            results.append(np.array([state_x, state_y, state_yaw, state_yaw, state_yaw, state_yaw]))
            state_x = x_min + 4.5
            state_yaw = -np.pi
            results.append(np.array([state_x, state_y, state_yaw, state_yaw, state_yaw, state_yaw]))
        return results 
    
    
    def add_two_non_intersecting_rectangles(self, existing_rects, frame):
        rectangles = existing_rects[:]
        for _ in range(2):
            while True:
                new_rect = self.generate_random_rectangle(frame)
                if all(not is_rectangle_intersect(new_rect, rect) for rect in rectangles):
                    rectangles.append(new_rect)
                    break
        return rectangles
    def generate_ox_oy_with_map_bound(self, obstacles_info):
        
        ox, oy = self.map.sample_surface(0.1)
        try:
            for rectangle in obstacles_info:
                obstacle = QuadrilateralObstacle(rectangle)
                ox_obs, oy_obs = obstacle.sample_surface(0.1)
                ox += ox_obs
                oy += oy_obs
        except:
            pass

        ox, oy = map_and_obs.remove_duplicates(ox, oy)
        return ox, oy
    
    def define_random_map_with_2_obstacles(self, angle_type=0):
        # outside frame
        #TODO: change to a more general way
        
        
        map_env = [(self.config["outer_wall_bound"]["x_min"], self.config["outer_wall_bound"]["y_min"]), \
            (self.config["outer_wall_bound"]["x_min"], self.config["outer_wall_bound"]["y_max"]), \
                (self.config["outer_wall_bound"]["x_max"], self.config["outer_wall_bound"]["y_min"]), \
                    (self.config["outer_wall_bound"]["x_max"], self.config["outer_wall_bound"]["y_max"])]
        existing_rectangles = []
        # existing rectangle
        existing_rectangle = [(-7, -1), (-7, 1), (4.5, 1), (4.5, -1)]
        existing_rectangles.append(existing_rectangle)
        
        # this is for goal generation
        if angle_type % 2 == 0:
            # pingfang
            new_fixed_rectangle = self.generate_fixed_size_random_rectangle(map_env, existing_rectangles)
        else:
            # shufang
            new_fixed_rectangle = self.generate_fixed_size_random_rectangle(map_env, existing_rectangles, length_x=2, length_y=11.5)
        results = self.get_equilbrium_configuration(new_fixed_rectangle)
        
        existing_rectangles.append(new_fixed_rectangle)

        ox, oy = self.map.sample_surface(0.1)
        new_rectangles = self.add_two_non_intersecting_rectangles(existing_rectangles, map_env)

        # randomly generate two obstacles
        obstacles = []
        obstacles_info = []
        for rectangle in new_rectangles:
            if rectangle in existing_rectangles:
                continue
            obstacle = QuadrilateralObstacle(rectangle)
            obstacles_info.append(rectangle)
            obstacles.append(obstacle)

        # add all the obstacle surfaces to the ox, oy list
        for obstacle in obstacles:
            ox_obs, oy_obs = obstacle.sample_surface(0.1)
            ox += ox_obs
            oy += oy_obs

        ox, oy = map_and_obs.remove_duplicates(ox, oy)
        return ox, oy, results, obstacles_info
    
    
    def extract_obstacles_info(self, obstacle_info):
        # [(), (), ...] -> [x1, x2, y1, y2, x1, x2, y1, y2, ...
        corners_info = [[min(point[0] for point in obstacle), max(point[0] for point in obstacle), 
                        min(point[1] for point in obstacle), max(point[1] for point in obstacle)] 
                        for obstacle in obstacle_info]
        return np.array(corners_info, dtype=np.float32).flatten()
    
    
    def reset(self, **kwargs):
        
        # 6-dim
        # seed if given
        if 'seed' in kwargs:
            self.seed(kwargs['seed'])
            np.random.seed(kwargs['seed'])
        if self.config["goal_with_obstacles_info_list"] is not None:
            # random choose between a given goal list if it is given
            number_goal_with_obstacles_info = len(self.config["goal_with_obstacles_info_list"])
            selected_index = np.random.randint(0, number_goal_with_obstacles_info)
            goal = self.config["goal_with_obstacles_info_list"][selected_index]["goal"]
            self.controlled_vehicle.reset_equilibrium(goal[0],goal[1],goal[2])
            self.goal = tuple(self.controlled_vehicle.observe())
            obstacles_info = self.config["goal_with_obstacles_info_list"][selected_index]["obstacles_info"]
            ox, oy = self.generate_ox_oy_with_map_bound(obstacles_info)
        else:
            angle_type = np.random.randint(0, 4)
            ox, oy, goal_results, obstacles_info = self.define_random_map_with_2_obstacles(angle_type)
            goal = goal_results[angle_type//2]
            # random sample a equilibrium goal
            self.controlled_vehicle.reset_equilibrium(goal[0],goal[1],goal[2])
            self.goal = tuple(self.controlled_vehicle.observe())
        # self.goal = [x_coordinates, y_coordinates]
        self.gx, self.gy, self.gyaw0, self.gyawt1, self.gyawt2, self.gyawt3 = self.goal
        # fix the env model once we reset, this self.ox, self.oy, self.obstacles_info will not change 
        # during the whole episode
        self.ox, self.oy = ox, oy
        self.obstacles_info = obstacles_info
        
        # currently under this env, we will not use the start_list for initialization
        if self.config["start_list"] is not None:
            # random choose between a given goal list
            number_starts = len(self.config["start_list"])
            selected_index = np.random.randint(0, number_starts)
            self.start = tuple(self.config["start_list"][selected_index])
            self.sx, self.sy, self.syaw0, self.syawt1, self.syawt2, self.syawt3 = self.start
        # shape the self.state to desired dim
        if self.vehicle_type == "single_tractor":
            self.controlled_vehicle.reset(self.sx, self.sy, self.syaw0)
        elif self.vehicle_type == "one_trailer":
            self.controlled_vehicle.reset(self.sx, self.sy, self.syaw0, self.syawt1)
        elif self.vehicle_type == "two_trailer":
            self.controlled_vehicle.reset(self.sx, self.sy, self.syaw0, self.syawt1, self.syawt2)
        else:
            self.controlled_vehicle.reset(self.sx, self.sy, self.syaw0, self.syawt1, self.syawt2, self.syawt3)
        
        if self.controlled_vehicle.is_collision(self.ox, self.oy):
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
        # We will always render the image for cluttered env
        if self.config["use_rgb"]:
            rgb_image = self.render()
        else:
            rgb_image = self.white_image
        obs_dict = OrderedDict([
            ('observation', self.controlled_vehicle.observe().astype(np.float32)),
            ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
            ("desired_goal", np.array(self.goal, dtype=np.float32)),
            ("obstacles_info", self.extract_obstacles_info(obstacles_info)),
            ("achieved_rgb_image", rgb_image)
        ])
        info_dict = {
            "crashed": False,
            "is_success": False,
            "jack_knife": False,
            "action": None,
            "old_state": None,
            "old_rgb_image": None,
        }
        return obs_dict, info_dict
    
    def sparse_reward(self, state, state_, goal=None):
        if goal is None:
            goal = np.array([self.goal], dtype=np.float32)
        # broadcast
        distance = mixed_norm(goal.squeeze(), state_)
        # new_state_diff = state_ - goal
        # new_weighted_distance = np.dot(np.dot(new_state_diff, self.distancematrix), new_state_diff.T).item()
        
        if distance < self.config["sparse_reward_threshold"]:
            reward = self.config['sucess_goal_reward_sparse']
        else:
            reward = -1
            
        return reward
    
    def sparse_reward_mod(self, state, state_, goal=None):
        if goal is None:
            goal = np.array([self.goal], dtype=np.float32)
        # broadcast
        distance = mixed_norm(goal.squeeze(), state_)
        # new_state_diff = state_ - goal
        # new_weighted_distance = np.sqrt(np.dot(np.dot(new_state_diff, self.distancematrix), new_state_diff.T).item())
        
        if distance < self.config["sparse_reward_threshold"]:
            reward = self.config['sucess_goal_reward_sparse']
        else:
            reward = -1
            
        return reward
    
    def reward(self, old_state, state, goal=None):
        if self.reward_type == "sparse_reward":
            reward = self.sparse_reward(old_state, state, goal)
        elif self.reward_type == "sparse_reward_mod":
            reward = self.sparse_reward_mod(old_state, state, goal)
        return reward
    
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Recalculate reward for HER replay buffer
        """
        rewards = []
        for j in range(achieved_goal.shape[0]):  
            if self.reward_type == "sparse_reward":
                reward = self.sparse_reward(info[j]["old_state"], achieved_goal[j], desired_goal[j])
            elif self.reward_type == "sparse_reward_mod":
                reward = self.sparse_reward_mod(info[j]["old_state"], achieved_goal[j], desired_goal[j])
            rewards.append(reward)
        
        return np.array(rewards)
    
    
    # # implement tt system here
    # def step(self, action):
            
    #     old_state = self.controlled_vehicle.observe()
    #     # clip action
    #     action_clipped = np.clip(action, -self.act_limit, self.act_limit)
    #     for _ in range(self.config["N_steps"]):
    #         self.controlled_vehicle.step(action_clipped, self.dt, self.config["allow_backward"])
    #         # Fank: when evaluating, we want every precise timestep state
    #         if self.evaluate_mode:
    #             self.state_list.append(self.controlled_vehicle.observe())
    #             self.action_list.append(action_clipped)    
    #     state = self.controlled_vehicle.observe()
    #     reward = self.reward(old_state, state)
    #     crashed = self.controlled_vehicle.is_collision(self.ox, self.oy)
    #     jack_knife = self.controlled_vehicle._is_jack_knife()
        
    #     if crashed:
    #         self.terminated = True
    #         reward += self.config["collision_penalty"]
    #         info_dict['crashed'] = True
            
    #     if jack_knife:
    #         self.terminated = True
    #         reward += self.config["jack_knife_penalty"]
    #         info_dict['jack_knife'] = True
            
    #     if reward >= self.config['sucess_goal_reward_sparse']:
    #         self.terminated = True
    #     self.current_step += 1
    #     if self.current_step >= self.config["max_episode_steps"]:
    #         self.truncated = True
        
    #     # check if success
    #     if reward >= self.config['sucess_goal_reward_sparse']:
    #         self.terminated = True
    #         info_dict['is_success'] = True
        
    #     obs_dict = OrderedDict([
    #         ('observation', state),
    #         ("achieved_goal", state),
    #         ("desired_goal", np.array(self.goal, dtype=np.float64)),
    #         ("obstacles_info", self.extract_obstacles_info(self.obstacles_info)),
    #     ])

    #     info_dict = {
    #         "crashed": crashed,
    #         "is_success": self.terminated,
    #         "jack_knife": jack_knife,
    #         "action": action_clipped,
    #         "old_state": old_state,
    #     }

    #     return obs_dict, reward, self.terminated, self.truncated, info_dict
    
    def step(self, action):
        old_state = self.controlled_vehicle.observe()
        action_clipped = np.clip(action, -self.act_limit, self.act_limit)
        if self.config["use_rgb"]:
            old_rgb_image = self.render()
        else:
            old_rgb_image = self.white_image

        for _ in range(self.config["N_steps"]):
            self.controlled_vehicle.step(action_clipped, self.dt, self.config["allow_backward"])
            if self.evaluate_mode:
                self.state_list.append(self.controlled_vehicle.observe())
                self.action_list.append(action_clipped)

        state = self.controlled_vehicle.observe()
        reward = self.reward(old_state, state)
        if self.config["use_rgb"]:   
            new_rgb_image = self.render()
        else:
            new_rgb_image = self.white_image
        crashed = self.controlled_vehicle.is_collision(self.ox, self.oy)
        jack_knife = self.controlled_vehicle._is_jack_knife()
        # still need to change reward
        if crashed:
            self.terminated = True
            reward += self.config["collision_penalty"]

        if jack_knife:
            self.terminated = True
            reward += self.config["jack_knife_penalty"]

        if reward >= self.config['sucess_goal_reward_sparse']:
            self.terminated = True
            is_success = True
        else:
            is_success = False

        self.current_step += 1
        if self.current_step >= self.config["max_episode_steps"]:
            self.truncated = True

        obs_dict = OrderedDict([
            ('observation', state.astype(np.float32)),
            ("achieved_goal", state.astype(np.float32)),
            ("desired_goal", np.array(self.goal, dtype=np.float32)),
            ("obstacles_info", self.extract_obstacles_info(self.obstacles_info)),
            ("achieved_rgb_image", new_rgb_image),
        ])

        info_dict = {
            "crashed": crashed,
            "is_success": is_success,
            "jack_knife": jack_knife,
            "action": action_clipped,
            "old_state": old_state,
            "old_rgb_image": old_rgb_image,
        }

        return obs_dict, reward, self.terminated, self.truncated, info_dict

    # def render(self, mode='human'):
    #     assert self.evaluate_mode
    #     plt.cla()
    #     ax = plt.gca()
    #     plt.plot(self.ox, self.oy, 'sk', markersize=1)
    #     # plt.plot(ox_, oy_, 'sk', markersize=0.5)
    #     self.controlled_vehicle.plot(ax, self.action_list[-1], 'blue')
        
    #     # Plot the goal vehicle
    #     gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = self.goal
    #     self.plot_vehicle = deepcopy(self.controlled_vehicle)
    #     self.plot_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
    #     self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
    #     plt.axis('equal')
    #     plt.savefig("runs_rl/tractor_trailer_envs.png")
    #     print(1)
    
    # def render(self, mode='human'):
    #     assert self.evaluate_mode
    #     # plt.cla()
    #     fig, ax = plt.subplots()
    #     # plt.plot(self.ox, self.oy, 'sr', markersize=1)
    #     map_vertices = self.map.vertices + [self.map.vertices[0]]  # Add the first vertex to the end to close the boundary
    #     map_x, map_y = zip(*map_vertices)  # Unpack the coordinates
    #     plt.plot(map_x, map_y, 'r-')  # Plot the boundaries in red

    #     # Plot the controlled vehicle
    #     self.controlled_vehicle.plot(ax, self.action_list[-1], 'blue', is_full=True)

    #     # Plot the goal vehicle
    #     gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = self.goal
    #     self.plot_vehicle = deepcopy(self.controlled_vehicle)
    #     self.plot_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
    #     self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green', is_full=True)

    #     # Plot the obstacles
    #     self.plot_obstacles(ax)

    #     ax.axis('equal')
    #     fig.savefig("runs_rl/tractor_trailer_envs.png")
    #     return fig  # Return the figure object
    

    def render(self):
        # assert self.evaluate_mode
        fig, ax = plt.subplots()  # Set the size of the figure

        map_vertices = self.map.vertices + [self.map.vertices[0]]
        map_x, map_y = zip(*map_vertices)
        plt.plot(map_x, map_y, 'r-')
        # here we full the vehicle so we don't care the exact action
        self.controlled_vehicle.plot(ax, np.array([0.0, 0.0]), 'blue', is_full=True)

        gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = self.goal
        self.plot_vehicle = deepcopy(self.controlled_vehicle)
        self.plot_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
        self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green', is_full=True)

        self.plot_obstacles(ax)

        ax.axis('off')

        # Convert the figure to a PIL Image object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        # Resize the image and convert it to RGB
        resized_img = img.resize((84, 84), Image.LANCZOS)
        rgb_img = resized_img.convert('RGB')

        # Save the image
        # rgb_img.save("runs_rl/tractor_trailer_envs.png")

        buf.close()
        plt.close(fig) # close the current figure window

        return np.array(np.transpose(rgb_img, (2,0,1))).astype(np.uint8)  # Return the PIL Image object
    
    def reconstruct_image_from_observation(self, observation):
        # a helper fuction that helps to reconstruct the image from the full-dim(26) observation
        # [x1, x2, y1, y2, x1, x2, y1, y2, ...] -> image
        start = observation[:6]
        goal = observation[12:18]
        
        obstacles_info_raw = observation[-8:]
        
        # Reconstruct 'obstacles_info' as a list of tuples
        obstacles_info = [[(obstacles_info_raw[i], obstacles_info_raw[i+2]), 
                        (obstacles_info_raw[i], obstacles_info_raw[i+3]), 
                        (obstacles_info_raw[i+1], obstacles_info_raw[i+3]), 
                        (obstacles_info_raw[i+1], obstacles_info_raw[i+2])] 
                        for i in range(0, len(obstacles_info_raw), 4)]
        
        fig, ax = plt.subplots()
        map_vertices = self.map.vertices + [self.map.vertices[0]]
        map_x, map_y = zip(*map_vertices)
        plt.plot(map_x, map_y, 'r-')
        # Notice here we only
        plot_vehicle = deepcopy(self.controlled_vehicle)
        sx, sy, syaw0, syawt1, syawt2, syawt3 = start
        plot_vehicle.reset(sx, sy, syaw0, syawt1, syawt2, syawt3)

        plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue', is_full=True)   
        gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = goal
        plot_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
        plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green', is_full=True)  
        for obstacle in obstacles_info:
            xs, ys = zip(*obstacle)
            ax.fill(xs, ys, 'red')
            
        ax.axis('off')  
        # Convert the figure to a PIL Image object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        # Resize the image and convert it to RGB
        resized_img = img.resize((84, 84), Image.LANCZOS)
        rgb_img = resized_img.convert('RGB')

        # Save the image
        rgb_img.save("runs_rl/tractor_trailer_envs2.png")

        buf.close()
        plt.close(fig) # close the current figure window

        return np.array(np.transpose(rgb_img, (2,0,1))).astype(np.uint8)  # Return the PIL Image object
    
    def plot_obstacles(self, ax):
        for obstacle in self.obstacles_info:
            xs, ys = zip(*obstacle)  # Unpack the coordinates
            ax.fill(xs, ys, 'red')
    
       
    def plot_exploration(self, i, goals_list):
        plt.cla()
        ax = plt.gca()
        # ox, oy = self.map.sample_surface(0.1)
        # ox, oy = map_and_obs.remove_duplicates(ox, oy)
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
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
        # ox, oy = self.map.sample_surface(0.1)
        # ox, oy = map_and_obs.remove_duplicates(ox, oy)
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
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
            plt.plot(self.ox, self.oy, "sk", markersize=1)
            # plt.plot(ox_, oy_, "sk", markersize=0.5)
            self.plot_vehicle = deepcopy(self.controlled_vehicle)
            self.plot_vehicle.reset(sx, sy, syaw0, syawt1, syawt2, syawt3)
            
            self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue')
            # TODO
            self.plot_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
            self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
            # plot the planning path
            plt.plot(pathx[:k], pathy[:k], linewidth=1.5, color='r')
            self.plot_vehicle.reset(pathx[k], pathy[k], pathyaw0[k],pathyawt1[k], pathyawt2[k], pathyawt3[k])
            try:
                self.plot_vehicle.plot(ax, self.action_list[k], 'black')
            except:
                self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
            

        ani = FuncAnimation(fig, update, frames=len(pathx), repeat=True)

        # Save the animation
        writer = PillowWriter(fps=24)
        if not save_dir:
            if not os.path.exists("./rl_training/gif/tt_cluttered_reaching"):
                os.makedirs("./rl_training/gif/tt_cluttered_reaching")
                
            base_path = "./rl_training/gif/tt_cluttered_reaching/path_simulation"
            extension = ".gif"
            
            all_files = os.listdir("./rl_training/gif/tt_cluttered_reaching")
            matched_files = [re.match(r'path_simulation(\d+)\.gif', f) for f in all_files]
            numbers = [int(match.group(1)) for match in matched_files if match]
            
            if numbers:
                save_index = max(numbers) + 1
            else:
                save_index = 1
            ani.save(base_path + str(save_index) + extension, writer=writer) 
        else:
            ani.save(save_dir, writer=writer)
            
        plt.close(fig)