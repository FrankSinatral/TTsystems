from gymnasium import Env
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Optional, Dict, List
from collections import OrderedDict
import os
import pprint
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


def rotate(point, angle, origin):
    """Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    point: (px, py)
    origin: (ox, oy)
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return qx, qy

def calculate_rectangle_corners(params):
    """Calculate the corners of a rectangle given its center, size and yaw angle.
    params: (center_x, center_y, width, height, yaw) or (center_x, center_y, width, height)
    output: [(x, y), (x, y), (x, y), (x, y)]
    """
    if len(params) == 4:
        cx, cy, width, height = params
        yaw = 0
    elif len(params) == 5:
        cx, cy, width, height, yaw = params
    else:
        raise ValueError("Input params must be a 4-tuple or 5-tuple")

    corners = np.array([
        [cx - width / 2, cy - height / 2],
        [cx + width / 2, cy - height / 2],
        [cx + width / 2, cy + height / 2],
        [cx - width / 2, cy + height / 2]
    ])
    return [rotate(corner, yaw, (cx, cy)) for corner in corners]

def is_rectangle_overlap(corners1, corners2):
    """Check if two rectangles overlap using the Separating Axis Theorem.
    input: corners1, corners2: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    output: True if overlap, False if not
    This can be directly use for obstacle and bounding_box detection
    """
    def get_axes(corners):
        axes = []
        for i in range(len(corners)):
            a, b = corners[i], corners[(i + 1) % len(corners)]
            axis_normal = (b[1] - a[1], a[0] - b[0]) #(b[0] - a[0], b[1] - a[1]) previous direction
            axes.append(axis_normal)
        return axes

    def project(corners, axis):
        projections = [np.dot(vertex, axis) for vertex in corners]
        return min(projections), max(projections)

    axes1 = get_axes(corners1)
    axes2 = get_axes(corners2)
    axes = axes1 + axes2

    for axis in axes:
        min_proj1, max_proj1 = project(corners1, axis)
        min_proj2, max_proj2 = project(corners2, axis)

        if max_proj1 < min_proj2 or min_proj1 > max_proj2:
            return False

    return True


def is_point_inside_rectangle(point, corners):
    """Check if a point is inside a rectangle.
    input: point: (x, y), corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    output: True if the point is inside, False if not
    if a point is in an edge of the corners, it is false
    """
    def get_axes(corners):
        axes = []
        for i in range(len(corners)):
            a, b = corners[i], corners[(i + 1) % len(corners)]
            axis_normal = (b[1] - a[1], a[0] - b[0])
            axes.append(axis_normal)
        return axes

    def project(points, axis):
        projections = [np.dot(vertex, axis) for vertex in points]
        return min(projections), max(projections)

    axes = get_axes(corners)

    for axis in axes:
        min_proj_rect, max_proj_rect = project(corners, axis)
        point_proj = np.dot(point, axis)

        if point_proj <= min_proj_rect or point_proj >= max_proj_rect:
            return False

    return True

def is_rectangle_inside(corners1, corners2):
    """Check if rectangle 1 is completely inside rectangle 2.
    input: corners1, corners2: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    output: True if corners1 is completely inside corners2, False if not
    the logic is that if the for all corners of the first rectangle, they are inside the second rectangle
    the result will be true
    this function can be direcly use for obstacle/bounding_box per map_vertices
    """
    for point in corners1:
        if not is_point_inside_rectangle(point, corners2):
            return False

    return True
        
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


def rotate(point, angle, origin):
    """Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    point: (px, py)
    origin: (ox, oy)
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return qx, qy

def rectangle_corners(center, width, height, yaw):
    """Calculate the corners of a rectangle given its center, size and yaw angle.
    center: (cx, cy)
    width: yaw
    height: vertical to yaw
    output: [(x, y), (), (), ()]
    """
    cx, cy = center
    corners = np.array([
        [cx - width / 2, cy - height / 2],
        [cx + width / 2, cy - height / 2],
        [cx + width / 2, cy + height / 2],
        [cx - width / 2, cy + height / 2]
    ])
    return [rotate(corner, yaw, center) for corner in corners]

def separating_axis_theorem(corners1, corners2):
    """Check if two rectangles overlap using the Separating Axis Theorem.
    input: corners1, corners2: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    output: True if overlap, False if not
    """
    for a, b in zip(corners1, corners1[1:] + corners1[:1]):
        axis_normal = (b[1] - a[1], a[0] - b[0])
        projections1 = [np.dot(vertex, axis_normal) for vertex in corners1]
        projections2 = [np.dot(vertex, axis_normal) for vertex in corners2]
        if max(projections1) < min(projections2) or min(projections1) > max(projections2):
            return False
    return True

def check_intersection(obstacle, obstacles_list, start, goal):
    """
    - obstacle: (center_x, center_y, width, height)
    - obstacles_list: [(center_x, center_y, width, height), ...]
    - start/ goal: (center_x, center_y, width, height, yaw)
    """
    ox, oy, width, height = obstacle
    obstacle_corners = [(ox - width / 2, oy - height / 2), (ox + width / 2, oy - height / 2), (ox + width / 2, oy + height / 2), (ox - width / 2, oy + height / 2)]
    for obs in obstacles_list:
        obs_corners = [(obs[0] - obs[2] / 2, obs[1] - obs[3] / 2), (obs[0] + obs[2] / 2, obs[1] - obs[3] / 2), (obs[0] + obs[2] / 2, obs[1] + obs[3] / 2), (obs[0] - obs[2] / 2, obs[1] + obs[3] / 2)]
        if separating_axis_theorem(obstacle_corners, obs_corners):
            return True
    start_corners = rectangle_corners(start[:2], start[2], start[3], start[4])
    if separating_axis_theorem(obstacle_corners, start_corners):
        return True
    goal_corners = rectangle_corners(goal[:2], goal[2], goal[3], goal[4])
    if separating_axis_theorem(obstacle_corners, goal_corners):
        return True
    return False


class TractorTrailerMetaPlanningEnv(Env):
    @classmethod
    def default_config(cls) -> dict:
        return {
            "verbose": False, 
            "vehicle_type": "single_tractor",
            "observation": "original",
            "reward_type": 'sparse_reward_mod',
            "act_limit": 1, 
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
                "safe_metric": 3.0, #[m] the safe metric for calculation
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
            "perception": {
                "one_hot_representation":{
                    "d": 5, # [m]distance
                    "number": 8, # number of regions
                },
                "one_hot_representation_enhanced": {
                    "d": 10,
                    "number": 8,
                },
                "lidar_detection_one_hot": {
                    "d": 5,
                },
                "lida_detection_one_hot_triple": {
                    "d": 5,
                },
            },
            
            "task_list": None, 
            "N_steps": 10, # number of steps for each action
            "jack_knife_penalty": -20,
            "collision_penalty": -20,
            "use_rgb": False, # whether use rgb image as an observation
            "use_gray": False, # whether use gray image as an observation
            "generate_goals_config": {
                "x_min": -30, #[m]
                "x_max": 30,
                "y_min": -30,
                "y_max": 30,
            },
            "generate_obstacles_config": {
                "number_obstacles": 2,
                "max_length": 20,
                "min_length": 1.0, 
                "fixed_number": True,
                "x_min": -30, #[m]
                "x_max": 30,
                "y_min": -30,
                "y_max": 30,
            },

            "with_obstacles_info": False,
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
        
    def define_observation_space(self) -> None:
        """Define our observation space
        self.observation_type: "original", "one_hot_representation", "one_hot_representation_enhanced", "lidar_detection", "lidar_detection_one_hot", "lidar_detection_one_hot_triple"
        """
        self.observation_type = self.config["observation"]
        achieved_goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        desired_goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # TODO: here we set the dimension of the obstacle info manually, in the future, we can make it more general
        if self.observation_type == "original" and (not self.config["use_gray"]):
            self.observation_space = spaces.Dict({
            'achieved_goal': achieved_goal_space,
            'desired_goal': desired_goal_space,
            'observation': observation_space,
        })
        elif self.observation_type == "original" and self.config["use_gray"]:
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'gray_image': spaces.Box(low=0, high=255, shape=(369,369), dtype=np.uint8),
            })
        elif self.observation_type == "original_with_obstacles_info":
            self.observation_space = spaces.Dict({
            'achieved_goal': achieved_goal_space,
            'desired_goal': desired_goal_space,
            'observation': observation_space,
        })
        elif self.observation_type == "one_hot_representation" and (not self.config["use_gray"]):
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'one_hot_representation': spaces.Box(low=0, high=1, shape=(self.config["perception"]["one_hot_representation"]["number"] * self.number_bounding_box,), dtype=np.float32),
            })
        elif self.observation_type == "one_hot_representation" and self.config["use_gray"]:
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'one_hot_representation': spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32),
                'gray_image': spaces.Box(low=0, high=255, shape=(369,369), dtype=np.uint8),
            })
        elif self.observation_type == "one_hot_representation_enhanced":
            # not working
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'one_hot_representation_enhanced': spaces.Box(low=-np.inf, high=np.inf, shape=(2 + (self.number_bounding_box * 3 * self.config["perception"]["one_hot_representation_enhanced"]["number"]),), dtype=np.float32), #TODO: adjust
            })
        elif self.observation_type == "lidar_detection" and (not self.config["use_gray"]):
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'lidar_detection': spaces.Box(low=0, high=100, shape=(32,), dtype=np.float32),
            }) # TODO: this is not always the case
        elif self.observation_type == "lidar_detection" and self.config["use_gray"]:
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'lidar_detection': spaces.Box(low=0, high=100, shape=(32,), dtype=np.float32),
                'gray_image': spaces.Box(low=0, high=255, shape=(369,369), dtype=np.uint8),
            })
        elif self.observation_type == "lidar_detection_one_hot":
            #TODO
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'lidar_detection_one_hot': spaces.Box(low=0, high=100, shape=(9*self.number_bounding_box,), dtype=np.float32),
            })
        elif self.observation_type == "lidar_detection_one_hot_triple":
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'lidar_detection_one_hot_triple': spaces.Box(low=0, high=100, shape=(3*9*self.number_bounding_box,), dtype=np.float32),
            })
        else: 
            self.observation_space = spaces.Dict({
                'achieved_goal': achieved_goal_space,
                'desired_goal': desired_goal_space,
                'observation': observation_space,
                'collision_metric': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
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
        
    def update_task_list(self, task_list):
        self.config["task_list"] = task_list
        
    def clear_task_list(self):
        self.config["task_list"] = None
        
    def configure(self, config: Optional[dict]) -> None:
        
        if config:
            self.config.update(config)
        
        if self.config["verbose"]:
            GREEN_BOLD = "\033[1m\033[32m"
            RESET = "\033[0m"
            print(GREEN_BOLD, end="")
            pprint.pprint(config)
            print(RESET, end="")
        self.reward_type = self.config["reward_type"]
        self.default_start = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.default_map_vertices = [(-50, -50), (-50, 50), (50, 50), (50, -50)]
        # fix start
        self.start = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.sx, self.sy, self.syaw0, self.syawt1, self.syawt2, self.syawt3 = self.start
        # Define outer wall map, start region and goal region
        self.goal_region = self.define_map(self.config["generate_goals_config"])
        
        # pick car type
        self.pick_vehicle()
        self.yawmax = np.pi
        self.define_observation_space()
        self.define_action_space()
        
        # Optional Parameters
        self.dt = 1 / self.config["simulation_freq"]
        self.act_limit = self.config["act_limit"]
        self.evaluate_mode = self.config['evaluate_mode']
        self.fixed_number = self.config["generate_obstacles_config"]["fixed_number"]
        # add a white image
        self.white_image = np.full((3, 84, 84), 255, dtype=np.uint8)   
                      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def pick_vehicle(self):
        self.vehicle_type = self.config["vehicle_type"]
        if self.vehicle_type == "single_tractor":
            self.controlled_vehicle = SingleTractor(self.config["controlled_vehicle_config"])
            self.number_bounding_box = 1
        elif self.vehicle_type == "one_trailer":
            self.controlled_vehicle = OneTrailer(self.config["controlled_vehicle_config"])
            self.number_bounding_box = 2
        elif self.vehicle_type == "two_trailer":
            self.controlled_vehicle = TwoTrailer(self.config["controlled_vehicle_config"])
            self.number_bounding_box = 3
        else:
            self.controlled_vehicle = ThreeTrailer(self.config["controlled_vehicle_config"])
            self.number_bounding_box = 4
    
    def generate_ox_oy_with_map_bound(self, obstacles_info):
        # Note that obstacels_info can be an empty list
        """
        obstacles_info: [[(x1, y1), (x1, y2), (x2, y2), (x2, y1)], [...]]
        obstacles_info: [] also can be None?
        """
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
    
    def generate_fixed_size_random_rectangle_obstacles(self, number_obstacles, min_length=1.0, max_length=10):
        """generate fixed size random rectange obstacles
        - number_obstacles: represent the number of obstacles(0-5)
        - min_length: the minimum length of the obstacle(we fix the minmum length to 1.0)
        - max_length: the maximum length of the obstacle
        
        Note that if we have no obstacles, the obstacles info will be [] instead of None
        """
        obstacles_list = []
        obstacles_info = []
        number = 0
        generate_trail_count = 0

        # Since it is a equilibrium state, we can calculate the configuration of the vehicle as just one bounding box
        # (x, y, l, d, yaw)
        start_configuration = self.controlled_vehicle.calculate_configurations_given_equilibrium(self.start)
        goal_configuration = self.controlled_vehicle.calculate_configurations_given_equilibrium(self.goal)
        start_corners = calculate_rectangle_corners(start_configuration)
        goal_corners = calculate_rectangle_corners(goal_configuration)
        while number < number_obstacles:
            center_x = self.np_random.uniform(self.config["generate_obstacles_config"]["x_min"], self.config["generate_obstacles_config"]["x_max"])
            center_y = self.np_random.uniform(self.config["generate_obstacles_config"]["y_min"], self.config["generate_obstacles_config"]["y_max"])
            length_x = self.np_random.uniform(min_length, max_length)
            length_y = self.np_random.uniform(min_length, max_length)
            new_obstacle = (center_x, center_y, length_x, length_y)
            new_obstacle_corners = calculate_rectangle_corners(new_obstacle)
            if not is_rectangle_overlap(start_corners, new_obstacle_corners) and not is_rectangle_overlap(goal_corners, new_obstacle_corners) and is_rectangle_inside(new_obstacle_corners, self.map.vertices):
                obstacles_list.append(new_obstacle)
                obstacles_info.append(new_obstacle_corners)
                number += 1
            generate_trail_count += 1
            if generate_trail_count >= 100:
                break
            
        return obstacles_info
    
    def generate_random_size_random_rectangle_obstacles(self, max_number_obstacles, min_length=1.0, max_length=10):
        """generate random size random rectange obstacles
        - number_obstacles: represent the number of obstacles(0-5)
        - min_length: the minimum length of the obstacle(we fix the minmum length to 1.0)
        - max_length: the maximum length of the obstacle
        
        Note that if we have no obstacles, the obstacles info will be [] instead of None
        """
        obstacles_list = []
        obstacles_info = []
        number = 0
        number_obstacles = np.random.randint(0, max_number_obstacles + 1)  # 初始化障碍物数量
        generate_trail_count = 0

        # Since it is a equilibrium state, we can calculate the configuration of the vehicle as just one bounding box
        # (x, y, l, d, yaw)
        start_configuration = self.controlled_vehicle.calculate_configurations_given_equilibrium(self.start)
        goal_configuration = self.controlled_vehicle.calculate_configurations_given_equilibrium(self.goal)
        start_corners = calculate_rectangle_corners(start_configuration)
        goal_corners = calculate_rectangle_corners(goal_configuration)
        while number < number_obstacles:
            center_x = self.np_random.uniform(self.config["generate_obstacles_config"]["x_min"], self.config["generate_obstacles_config"]["x_max"])
            center_y = self.np_random.uniform(self.config["generate_obstacles_config"]["y_min"], self.config["generate_obstacles_config"]["y_max"])
            length_x = self.np_random.uniform(min_length, max_length)
            length_y = self.np_random.uniform(min_length, max_length)
            new_obstacle = (center_x, center_y, length_x, length_y)
            new_obstacle_corners = calculate_rectangle_corners(new_obstacle)
            if not is_rectangle_overlap(start_corners, new_obstacle_corners) and not is_rectangle_overlap(goal_corners, new_obstacle_corners) and is_rectangle_inside(new_obstacle_corners, self.map.vertices):
                obstacles_list.append(new_obstacle)
                obstacles_info.append(new_obstacle_corners)
                number += 1
            generate_trail_count += 1
            if generate_trail_count >= 100:
                break
            
        return obstacles_info
    
    def extract_obstacles_properties(self, obstacles_info):
        """Here the obstacles_info will be [] or have something"""
        if not obstacles_info:
            return []

        properties = []
        for obstacle in obstacles_info:
            x_coordinates = [point[0] for point in obstacle]
            y_coordinates = [point[1] for point in obstacle]
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            min_y = min(y_coordinates)
            max_y = max(y_coordinates)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            length = max_x - min_x
            width = max_y - min_y
            properties.append((center_x, center_y, length, width))

        return properties
    
    def extract_map_properties(self):
        x_coordinates = [point[0] for point in self.map.vertices]
        y_coordinates = [point[1] for point in self.map.vertices]
        min_x = min(x_coordinates)
        max_x = max(x_coordinates)
        min_y = min(y_coordinates)
        max_y = max(y_coordinates)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        length = max_x - min_x
        width = max_y - min_y
        return (center_x, center_y, length, width)
    
    def check_bounding_box_list_inside_map(self, bounding_box_list):
        for bounding_box in bounding_box_list:
            bounding_box_corners = calculate_rectangle_corners(bounding_box)
            if not is_rectangle_inside(bounding_box_corners, self.map.vertices):
                return False
        return True
    
    def check_bounding_box_list_not_collide_obstacles(self, bounding_box_list, obstacles_info):
        for bounding_box in bounding_box_list:
            bounding_box_corners = calculate_rectangle_corners(bounding_box)
            for obstacle in obstacles_info:
                if is_rectangle_overlap(bounding_box_corners, obstacle):
                    return False
        return True
            
    def sample_goal(self, obstacles_info=None):
        """
        this function is for our generation of the goal
        - case1: there is no obstacles we just sample our goal
        
        - case2: there is obstacles we have to check whether the goal is legal
        """
        while True:
            x_coordinates = self.np_random.uniform(self.config["generate_goals_config"]["x_min"], self.config["generate_goals_config"]["x_max"])
            y_coordinates = self.np_random.uniform(self.config["generate_goals_config"]["y_min"], self.config["generate_goals_config"]["y_max"])
            yaw_state = self.np_random.uniform(-self.yawmax, self.yawmax)
            current_goal = np.array([x_coordinates, y_coordinates, yaw_state, yaw_state, yaw_state, yaw_state], dtype=np.float32)
            bounding_box_list = self.controlled_vehicle.get_bounding_box_list(current_goal)
            if obstacles_info is None:
                if self.check_bounding_box_list_inside_map(bounding_box_list):
                    return current_goal
            else: 
                if self.check_bounding_box_list_inside_map(bounding_box_list) and self.check_bounding_box_list_not_collide_obstacles(bounding_box_list, obstacles_info):
                    return current_goal
    
    def sample_obstacles(self):
        # sample fixed number obstacles
        """Always after goal generation"""
        if self.fixed_number:
            return self.generate_fixed_size_random_rectangle_obstacles(number_obstacles=self.config["generate_obstacles_config"]["number_obstacles"], min_length=self.config["generate_obstacles_config"]["min_length"], max_length=self.config["generate_obstacles_config"]["max_length"])
        # sample random number obstacles
        else:
            return self.generate_random_size_random_rectangle_obstacles(max_number_obstacles=self.config["generate_obstacles_config"]["number_obstacles"], min_length=self.config["generate_obstacles_config"]["min_length"], max_length=self.config["generate_obstacles_config"]["max_length"])

    def reset(self, **kwargs):
        
        """
        reset function
        this reset function gives a specific tractor-trailer planning task for 
        the downstream process
        """
        if 'seed' in kwargs:
            self.seed(kwargs['seed'])
            np.random.seed(kwargs['seed'])
        
        
        # This blocks given (self.start) (self.sx etc), (self.map)
        # (self.goal) (self.gx etc) (self.obstacles_info)
        # give one task every  
        if self.config["task_list"] is None:
            # Use the default map vertices
            self.map = MapBound(self.default_map_vertices)
            # Use the default start (fixed)
            self.start = self.default_start
            self.sx, self.sy, self.syaw0, self.syawt1, self.syawt2, self.syawt3 = self.start
            # Sample random goal(A procedure)
            goal = self.sample_goal(obstacles_info=None)
            self.goal = tuple(goal)
            self.controlled_vehicle.reset_equilibrium(goal[0], goal[1], goal[2])
            # Generate Obstacles
            obstacles_info = self.sample_obstacles()
        else:
            assert isinstance(self.config["task_list"], List) and len(self.config["task_list"]) > 0, "task_list should be a list with at least one element"
            selected_index = np.random.randint(0, len(self.config["task_list"]))
            task_dict = self.config["task_list"][selected_index]
            map_vertices = task_dict.get("map_vertices", None)
            start = task_dict.get("start", None)
            goal = task_dict.get("goal", None)
            obstacles_info = task_dict.get("obstacles_info", None)
            if map_vertices is None:
                self.map = MapBound(self.default_map_vertices)
            else:
                self.map = MapBound(task_dict["map_vertices"])
            if start is None:
                self.start = self.default_start
            else:
                self.start = tuple(start)
            self.sx, self.sy, self.syaw0, self.syawt1, self.syawt2, self.syawt3 = self.start
            if goal is None:
                goal = self.sample_goal(obstacles_info=obstacles_info)
            self.goal = tuple(goal)
            self.controlled_vehicle.reset_equilibrium(goal[0], goal[1], goal[2])

            if obstacles_info is None:
                obstacles_info = self.sample_obstacles()

        ox, oy = self.generate_ox_oy_with_map_bound(obstacles_info)
        
        # fix the env model once we reset, this self.ox, self.oy, self.obstacles_info will not change 
        # during the whole episode
        # the sample points from the obstacles should be memorized
        self.ox, self.oy = ox, oy
        self.obstacles_info = obstacles_info
        # self.obstacles_info will not None
        # here's a function to shift the obstacles info
        self.obstacles_properties = self.extract_obstacles_properties(self.obstacles_info)
        self.map_properties = self.extract_map_properties()
        
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
            
        if self.config["use_gray"]:
            self.gray_image = self.render_obstacles()
        else:
            self.gray_image = self.white_image
            
        if self.observation_type == "original" and (not self.config["use_gray"]):
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
            ])
        elif self.observation_type == "original" and self.config["use_gray"]:
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("gray_image", self.gray_image)
            ])
        elif self.observation_type == "original_with_obstacles_info":
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
            ])
        elif self.observation_type == "one_hot_representation" and (not self.config["use_gray"]):
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("one_hot_representation", self.controlled_vehicle.one_hot_representation(d=self.config["perception"]["one_hot_representation"]["d"], \
                    number=self.config["perception"]["one_hot_representation"]["number"], ox=self.ox, oy=self.oy))
            ])
        elif self.observation_type == "one_hot_representation" and self.config["use_gray"]:
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("one_hot_representation", self.controlled_vehicle.one_hot_representation(d=self.config["perception"]["one_hot_representation"]["d"], \
                    number=self.config["perception"]["one_hot_representation"]["number"], ox=self.ox, oy=self.oy)),
                ("gray_image", self.gray_image)
            ])
        elif self.observation_type == "one_hot_representation_enhanced":
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("one_hot_representation_enhanced", self.controlled_vehicle.one_hot_representation_enhanced(d=10, number=8, ox=self.ox, oy=self.oy))
            ])
        elif self.observation_type == "lidar_detection" and (not self.config["use_gray"]):
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("lidar_detection", self.controlled_vehicle.lidar_detection(self.ox, self.oy))
            ])
        elif self.observation_type == "lidar_detection" and self.config["use_gray"]:
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("lidar_detection", self.controlled_vehicle.lidar_detection(self.ox, self.oy)),
                ("gray_image", self.gray_image)
            ])
        elif self.observation_type == "lidar_detection_one_hot":
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("lidar_detection_one_hot", self.controlled_vehicle.lidar_detection_one_hot(self.config["perception"]["lidar_detection_one_hot"]["d"], \
                    self.ox, self.oy)),
            ])
        elif self.observation_type == "lidar_detection_one_hot_triple":
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("lidar_detection_one_hot_triple", np.concatenate([self.controlled_vehicle.lidar_detection_one_hot(self.config["perception"]["lidar_detection_one_hot_triple"]["d"], \
                    self.ox, self.oy), self.controlled_vehicle.lidar_detection_one_hot(2*self.config["perception"]["lidar_detection_one_hot_triple"]["d"], \
                    self.ox, self.oy), self.controlled_vehicle.lidar_detection_one_hot(3*self.config["perception"]["lidar_detection_one_hot_triple"]["d"], \
                    self.ox, self.oy)])),
            ])
        elif self.observation_type == "obstacles_image":
            gray_image = self.render_obstacles()
            obs_dict = OrderedDict([
                ('observation', self.controlled_vehicle.observe().astype(np.float32)),
                ("achieved_goal", self.controlled_vehicle.observe().astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("obstacles_image", gray_image)
            ])
        
        info_dict = {
            "crashed": False,
            "is_success": False,
            "jack_knife": False,
            "action": None,
            "old_state": None,
            "old_rgb_image": None,
            "obstacles_info": self.obstacles_info,
            "obstacles_properties": self.obstacles_properties,
            "map_vertices": self.map.vertices,
            "map_properties": self.map_properties,
        }
        return obs_dict, info_dict
    
    def sparse_reward(self, state, state_, goal=None):
        if goal is None:
            goal = np.array([self.goal], dtype=np.float32)
        # broadcast
        distance = mixed_norm(goal.squeeze(), state_)
        # new_state_diff = state_ - goal
        
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
    
    def step(self, action):
        old_state = self.controlled_vehicle.observe()
        action_clipped = np.clip(action, -self.act_limit, self.act_limit)
        if self.config["use_rgb"]:
            old_rgb_image = self.render()
        else:
            old_rgb_image = self.white_image

        crashed, jack_knife = False, False
        for _ in range(self.config["N_steps"]):
            self.controlled_vehicle.step(action_clipped, self.dt, self.config["allow_backward"])
            # Fank: change every little step to check collision
            if self.controlled_vehicle.is_collision(self.ox, self.oy):
                crashed = True
            
            if self.controlled_vehicle._is_jack_knife():
                jack_knife = True
            if self.evaluate_mode:
                self.state_list.append(self.controlled_vehicle.observe())
                self.action_list.append(action_clipped)
            # TODO: I change here     
            if crashed or jack_knife:
                break
        # dtype: float64 np_array
        state = self.controlled_vehicle.observe()
        reward = self.reward(old_state, state)
        if self.config["use_rgb"]:   
            new_rgb_image = self.render()
        else:
            new_rgb_image = self.white_image
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
        
        # Fank: after check success  
        if self.observation_type == "lidar_detection" and not crashed:
            lidar_detection = self.controlled_vehicle.lidar_detection(self.ox, self.oy)
            minLidar = np.min(lidar_detection)
            if minLidar <= 5:
                reward += self.config["collision_penalty"] * ( 1 - minLidar / 5)
        
        # Take this form out       
        # if self.observation_type == "one_hot_representation" and not crashed:
        #     one_hot_representation = self.controlled_vehicle.one_hot_representation(d=3, number=8, ox=self.ox, oy=self.oy)
        #     sumOneHot = np.sum(one_hot_representation)
        #     if sumOneHot > 0:
        #         reward += self.config["collision_penalty"] * (sumOneHot / 32)
            

        self.current_step += 1
        if self.current_step >= self.config["max_episode_steps"]:
            self.truncated = True
        
        if self.observation_type == "original" and (not self.config["use_gray"]):
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
            ])
        elif self.observation_type == "original" and self.config["use_gray"]:
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("gray_image", self.gray_image)
            ])
        elif self.observation_type == "original_with_obstacles_info":
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
            ])
        elif self.observation_type == "one_hot_representation" and (not self.config["use_gray"]):
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("one_hot_representation", self.controlled_vehicle.one_hot_representation(
                    d=self.config["perception"]["one_hot_representation"]["d"], \
                    number=self.config["perception"]["one_hot_representation"]["number"], \
                    ox=self.ox, oy=self.oy))
            ])
        elif self.observation_type == "one_hot_representation" and self.config["use_gray"]:
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("one_hot_representation", self.controlled_vehicle.one_hot_representation(
                    d=self.config["perception"]["one_hot_representation"]["d"], \
                    number=self.config["perception"]["one_hot_representation"]["number"], \
                    ox=self.ox, oy=self.oy)),
                ("gray_image", self.gray_image)
            ])
        elif self.observation_type == "one_hot_representation_enhanced":
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("one_hot_representation_enhanced", self.controlled_vehicle.one_hot_representation_enhanced(
                    d=self.config["perception"]["one_hot_representation_enhanced"]["d"], \
                    number=self.config["perception"]["one_hot_representation_enhanced"]["number"], \
                    ox=self.ox, oy=self.oy))
            ])
        elif self.observation_type == "lidar_detection" and (not self.config["use_gray"]):
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("lidar_detection", self.controlled_vehicle.lidar_detection(self.ox, self.oy))
            ])
        elif self.observation_type == "lidar_detection" and self.config["use_gray"]:
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("lidar_detection", self.controlled_vehicle.lidar_detection(self.ox, self.oy)),
                ("gray_image", self.gray_image)
            ])
        elif self.observation_type == "lidar_detection_one_hot":
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("lidar_detection_one_hot", self.controlled_vehicle.lidar_detection_one_hot(self.config["perception"]["lidar_detection_one_hot"]["d"],\
                    self.ox, self.oy)),
            ])
        elif self.observation_type == "lidar_detection_one_hot_triple":
            obs_dict = OrderedDict([
                ('observation', state.astype(np.float32)),
                ("achieved_goal", state.astype(np.float32)),
                ("desired_goal", np.array(self.goal, dtype=np.float32)),
                ("lidar_detection_one_hot_triple", np.concatenate([self.controlled_vehicle.lidar_detection_one_hot(self.config["perception"]["lidar_detection_one_hot_triple"]["d"],\
                    self.ox, self.oy),self.controlled_vehicle.lidar_detection_one_hot(2*self.config["perception"]["lidar_detection_one_hot_triple"]["d"],\
                    self.ox, self.oy),self.controlled_vehicle.lidar_detection_one_hot(3*self.config["perception"]["lidar_detection_one_hot_triple"]["d"],\
                    self.ox, self.oy)])),
            ])

        info_dict = {
            "crashed": crashed,
            "is_success": is_success,
            "jack_knife": jack_knife,
            "action": action_clipped,
            "old_state": old_state,
            "old_rgb_image": old_rgb_image,
            "obstacles_info": self.obstacles_info,
            "obstacles_properties": self.obstacles_properties,
            "map_vertices": self.map.vertices,
            "map_properties": self.map_properties,
        }

        return obs_dict, reward, self.terminated, self.truncated, info_dict
    
    def real_render(self):
        assert self.evaluate_mode
        plt.cla()
        ax = plt.gca()
        plt.plot(self.ox, self.oy, 'sk', markersize=0.5)
        plot_vehicle = deepcopy(self.controlled_vehicle)
        # plt.plot(ox_, oy_, 'sk', markersize=0.5)
        try:
            plot_vehicle.plot(ax, self.action_list[-1], 'blue')
        except:
            plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue')
        if self.observation_type == "lidar_detection_one_hot":
            plot_vehicle.plot_lidar_detection_one_hot(self.config["perception"]["lidar_detection_one_hot"]["d"])
        if self.observation_type == "lidar_detection_one_hot_triple":
            plot_vehicle.plot_lidar_detection_one_hot(self.config["perception"]["lidar_detection_one_hot_triple"]["d"])
            plot_vehicle.plot_lidar_detection_one_hot(2*self.config["perception"]["lidar_detection_one_hot_triple"]["d"])
            plot_vehicle.plot_lidar_detection_one_hot(3*self.config["perception"]["lidar_detection_one_hot_triple"]["d"])
        # Plot the goal vehicle
        gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = self.goal
        plot_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
        plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        plt.axis('equal')
        
        # frame_dir = "rl_training/evaluation/frames"
        # os.makedirs(frame_dir, exist_ok=True)
        # plt.savefig(os.path.join(frame_dir, f"frame_{frame_idx:04d}.png"))
        # plt.close()
        plt.savefig("rl_training/tractor_trailer_plots.png")
        plt.close()
        
    def real_free_large_render(self):
        assert self.evaluate_mode
        plt.cla()
        ax = plt.gca()
        plt.plot(self.ox, self.oy, 'sk', markersize=0.5)
        plot_vehicle = deepcopy(self.controlled_vehicle)
        # plt.plot(ox_, oy_, 'sk', markersize=0.5)
        try:
            plot_vehicle.plot(ax, self.action_list[-1], 'blue')
        except:
            plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue')
        # if self.observation_type == "lidar_detection_one_hot":
        #     plot_vehicle.plot_lidar_detection_one_hot(5)
        
        plt.axis('equal')
        
        
        plt.savefig("rl_training/tractor_trailer_free_large_plots.png")
        plt.close()
    
    def render(self):
        """when rgb mode used, we will render the image as a rgb image"""
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
        rgb_img.save("runs_rl/meta_tractor_trailer_env_rgb.png")

        buf.close()
        plt.close(fig) # close the current figure window

        return np.array(np.transpose(rgb_img, (2,0,1))).astype(np.uint8)  # Return the PIL Image object
    
    def render_obstacles(self):
        """Render the obstacles as a single-channel image"""
        fig, ax = plt.subplots()  # Set the size of the figure

        map_vertices = self.map.vertices + [self.map.vertices[0]]
        map_x, map_y = zip(*map_vertices)
        margin = 0.1
        min_x, max_x = min(map_x), max(map_x)
        min_y, max_y = min(map_y), max(map_y)
        plt.plot(map_x, map_y, 'r-')

        # Set the limits of the plot to the boundaries of the map
        plt.xlim(min_x - margin, max_x + margin)
        plt.ylim(min_y - margin, max_y + margin)
        
        # Set the aspect of the plot to be equal
        ax.set_aspect('equal')

        self.plot_obstacles(ax)

        ax.axis('off')

        # Convert the figure to a PIL Image object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)

        # Resize the image and convert it to grayscale
        # resized_img = img.resize((84, 84), Image.LANCZOS)
        gray_img = img.convert('L')
        
        # Convert all non-255 values to 0
        np_img = np.array(gray_img)
        # don't know if this works: currently take this form out
        # np_img = np.where(np_img != 255, 0, np_img)

        # # Save the image
        gray_img.save("runs_rl/meta_tractor_trailer_env_obstacles.png")

        buf.close()
        plt.close(fig)  # close the current figure window
        
        # We don't shift the size of the image now

        return np_img.astype(np.uint8)  # Return the PIL Image object
        
    def render_vehicles(self):
        # plot vehicles and its corresponding goal
        """when rgb mode used, this is used for jinyu's render"""
        fig, ax = plt.subplots()  # Set the size of the figure

        map_vertices = self.map.vertices + [self.map.vertices[0]]
        map_x, map_y = zip(*map_vertices)
        margin = 0.1
        min_x, max_x = min(map_x), max(map_x)
        min_y, max_y = min(map_y), max(map_y)
        
        # plt.plot(map_x, map_y, 'r-')
        
        # here we full the vehicle so we don't care the exact action
        self.controlled_vehicle.plot(ax, np.array([0.0, 0.0]), 'blue', is_full=True)
        gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = self.goal
        self.plot_vehicle = deepcopy(self.controlled_vehicle)
        self.plot_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
        self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green', is_full=True)
        # Set the limits of the plot to the boundaries of the map
        plt.xlim(min_x - margin, max_x + margin)
        plt.ylim(min_y - margin, max_y + margin)

        
        # Set the aspect of the plot to be equal
        ax.set_aspect('equal')
        # self.plot_obstacles(ax)

        ax.axis('off')

        # Convert the figure to a PIL Image object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        img_rgb = img.convert('RGB')
        # # Resize the image to 128x128 and convert it to RGB
        # img_resized = img.resize((128, 128)).convert('RGB')
        
        # Convert the resized image to a numpy array
        np_img = np.array(img_rgb)
        
        # Save the resized image
        # img_rgb.save("runs_rl/meta_tractor_trailer_env_vehicles.png")

        buf.close()
        plt.close(fig) # close the current figure window
        
        return np.transpose(np_img, (2, 0, 1))  # Return the numpy array of the resized image
    
    
    def reconstruct_image_from_observation(self, observation):
        # TODO: here we need to change
        
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
        # rgb_img.save("runs_rl/tractor_trailer_envs2.png")
        # rgb_img.save("runs_rl/tractor_trailer_envs4.png")

        buf.close()
        plt.close(fig) # close the current figure window

        return np.array(np.transpose(rgb_img, (2,0,1))).astype(np.uint8)  # Return the PIL Image object
    
    def plot_obstacles(self, ax):
        for obstacle in self.obstacles_info:
            xs, ys = zip(*obstacle)  # Unpack the coordinates
            ax.fill(xs, ys, 'red')
        
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
            if self.observation_type == "lidar_detection_one_hot":
                detection_result = self.plot_vehicle.lidar_detection_one_hot(5, self.ox, self.oy)
                self.plot_vehicle.plot_lidar_detection_one_hot(5)
                detection_text = f"Detection: {detection_result}"
                ax.text(0.05, 0.95, detection_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            try:
                self.plot_vehicle.plot(ax, self.action_list[k], 'black')
            except:
                self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
            

        ani = FuncAnimation(fig, update, frames=len(pathx), repeat=True)

        # Save the animation
        writer = PillowWriter(fps=24)
        if not save_dir:
            if not os.path.exists("./rl_training/gif/tt_planning"):
                os.makedirs("./rl_training/gif/tt_planning")
                
            base_path = "./rl_training/gif/tt_planning/path_simulation"
            extension = ".gif"
            
            all_files = os.listdir("./rl_training/gif/tt_planning")
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
    
    
    def run_simulation_explore(self, save_dir=None):
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
            # plot the planning path
            plt.plot(pathx[:k], pathy[:k], linewidth=1.5, color='r')
            self.plot_vehicle.reset(pathx[k], pathy[k], pathyaw0[k],pathyawt1[k], pathyawt2[k], pathyawt3[k])
            if self.observation_type == "lidar_detection_one_hot":
                detection_result = self.plot_vehicle.lidar_detection_one_hot(5, self.ox, self.oy)
                self.plot_vehicle.plot_lidar_detection_one_hot(5)
                detection_text = f"Detection: {detection_result}"
                ax.text(0.05, 0.95, detection_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            try:
                self.plot_vehicle.plot(ax, self.action_list[k], 'black')
            except:
                self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
            

        ani = FuncAnimation(fig, update, frames=len(pathx), repeat=True)

        # Save the animation
        writer = PillowWriter(fps=24)
        if not save_dir:
            if not os.path.exists("./rl_training/gif/tt_explore"):
                os.makedirs("./rl_training/gif/tt_explore")
                
            base_path = "./rl_training/gif/tt_explore/path_simulation"
            extension = ".gif"
            
            all_files = os.listdir("./rl_training/gif/tt_explore")
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
        
    def save_result(self, save_dir=None):
        """notice here we need to give obstacles info"""
        assert self.evaluate_mode # in case that you use the function not correctly

        # ox, oy = self.map.sample_surface(0.1)
        # if obstacles_info is not None:
        #     for rectangle in obstacles_info:
        #         obstacle = QuadrilateralObstacle(rectangle)
        #         ox_obs, oy_obs = obstacle.sample_surface(0.1)
        #         ox += ox_obs
        #         oy += oy_obs
        # ox, oy = map_and_obs.remove_duplicates(ox, oy)
        # ox_, oy_ = self.goal_region.sample_surface(1)
        # ox_, oy_ = map_and_obs.remove_duplicates(ox_, oy_)
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

        fig, ax = plt.subplots()
        ax.clear()
        plt.axis("equal")
        k = len(pathx) - 1  # Use the last frame
        plt.plot(self.ox, self.oy, "sk", markersize=1)
        # plt.plot(ox_, oy_, "sk", markersize=0.5)
        self.plot_vehicle = deepcopy(self.controlled_vehicle)
        self.plot_vehicle.reset(sx, sy, syaw0, syawt1, syawt2, syawt3)

        self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue')
        self.plot_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
        self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        plt.plot(pathx[:k], pathy[:k], linewidth=1.5, color='r')
        self.plot_vehicle.reset(pathx[k], pathy[k], pathyaw0[k],pathyawt1[k], pathyawt2[k], pathyawt3[k])
        try:
            self.plot_vehicle.plot(ax, self.action_list[k], 'black')
        except:
            self.plot_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')

        # Save the final frame as a PNG image
        if not save_dir:
            if not os.path.exists("./rl_training/png/tt_meta_reaching"):
                os.makedirs("./rl_training/png/tt_meta_reaching")

            base_path = "./rl_training/png/tt_meta_reaching/path_simulation"
            extension = ".png"

            all_files = os.listdir("./rl_training/png/tt_meta_reaching")
            matched_files = [re.match(r'path_simulation(\d+)\.png', f) for f in all_files]
            numbers = [int(match.group(1)) for match in matched_files if match]

            if numbers:
                save_index = max(numbers) + 1
            else:
                save_index = 1
            plt.savefig(base_path + str(save_index) + extension)
        else:
            plt.savefig(save_dir)
        plt.close()