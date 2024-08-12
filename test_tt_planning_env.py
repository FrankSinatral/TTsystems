import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint
import logging

# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs
from copy import deepcopy
from tractor_trailer_envs import register_tt_envs
register_tt_envs()
import numpy as np
import yaml
import pickle
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import planner

# 设置logging配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    with open("configs/envs/free_big_planning.yaml", 'r') as file:
        config = yaml.safe_load(file)
    with open("datasets/goal_with_obstacles_info_list_hz.pickle", 'rb') as file:
        task_list = pickle.load(file)
    env = gym.make("tt-planning-v0", config=config)
    planner_config = {
        "plot_final_path": False,
        "plot_rs_path": False,
        "plot_expand_tree": False,
        "mp_step": 10,  # Important
        "N_steps": 10,  # Important
        "range_steer_set": 20,
        "max_iter": 50,
        "heuristic_type": "traditional",
        "save_final_plot": False,
        "controlled_vehicle_config": {
            "w": 2.0,  # [m] width of vehicle
            "wb": 3.5,  # [m] wheel base: rear to front steer
            "wd": 1.4,  # [m] distance between left-right wheels (0.7 * W)
            "rf": 4.5,  # [m] distance from rear to vehicle front end
            "rb": 1.0,  # [m] distance from rear to vehicle back end
            "tr": 0.5,  # [m] tyre radius
            "tw": 1.0,  # [m] tyre width
            "rtr": 2.0,  # [m] rear to trailer wheel
            "rtf": 1.0,  # [m] distance from rear to trailer front end
            "rtb": 3.0,  # [m] distance from rear to trailer back end
            "rtr2": 2.0,  # [m] rear to second trailer wheel
            "rtf2": 1.0,  # [m] distance from rear to second trailer front end
            "rtb2": 3.0,  # [m] distance from rear to second trailer back end
            "rtr3": 2.0,  # [m] rear to third trailer wheel
            "rtf3": 1.0,  # [m] distance from rear to third trailer front end
            "rtb3": 3.0,  # [m] distance from rear to third trailer back end   
            "max_steer": 0.6,  # [rad] maximum steering angle
            "v_max": 2.0,  # [m/s] maximum velocity 
            "safe_d": 0.0,  # [m] the safe distance from the vehicle to obstacle 
            "safe_metric": 3.0,  # the safe distance from the vehicle to obstacle
            "xi_max": (np.pi) / 4,  # jack-knife constraint  
        },
        "acceptance_error": 0.5,
    }
    # env.unwrapped.update_task_list(task_list)
    task_list = [
        {
            "goal": np.array([-30, 0, 0, 0, 0, 0], dtype=np.float32),
            "obstacles_info": [[(-15, -10), (-15, 10), (-16, 10), (-16, -10)]],
        }
    ]
    env.unwrapped.update_task_list(task_list)
    feasible_count = 0
    scenario_id = 102659
    while feasible_count < 100000:
        logging.info(f"Trying scenario_id: {scenario_id}")
        obs, info = env.reset(seed=scenario_id) 
        while (not planner.check_is_start_feasible(obs["achieved_goal"], info["obstacles_info"], info["map_vertices"], planner_config)) or (not env.unwrapped.check_goal_with_using_lidar_detection_one_hot()):
            logging.info(f"Scenario {scenario_id} is not feasible, trying next scenario.")
            scenario_id += 1
            logging.info(f"Trying scenario_id: {scenario_id}")
            obs, info = env.reset(seed=scenario_id)
        env.unwrapped.real_render()  
        for i in range(200):
            obs, reward, terminated, truncated, info = env.step([1, 0])
            env.unwrapped.real_render()
        feasible_count += 1
        logging.info(f"Scenario {scenario_id} is feasible. Total feasible count: {feasible_count}")
        scenario_id += 1
        # env.unwrapped.real_render()
    print(1)

if __name__ == "__main__":
    main()