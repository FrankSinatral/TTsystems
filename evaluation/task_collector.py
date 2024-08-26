import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs

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
from joblib import Parallel, delayed
from joblib import load

def convert_tuple_to_dict(task_list):
    converted_list = []
    for task in task_list:
        task_dict = {
            "start": task[0],
            "goal": task[1],
            "obstacles_info": task[2],
            "map_vertices": task[3],
            "obstacles_properties": task[4],
            "map_properties": task[5]
        }
        converted_list.append(task_dict)
    return converted_list
    

def main():
    # initiliaze an env for checking dataset quaility
    task_list = []
    with open("configs/envs/generate.yaml", 'r') as file:
        config = yaml.safe_load(file)
    env = gym.make("tt-planning-v0", config=config)
    generate_feasible_number = 0
    feasible_seed = -1
    check_planner_config = {
            "plot_final_path": False,
            "plot_rs_path": False,
            "plot_expand_tree": False,
            "mp_step": 10,
            "N_steps": 10,
            "range_steer_set": 20,
            "max_iter": 20,
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
    while generate_feasible_number < 5000:
        feasible_seed += 1
        print("Start Finding a New Feasible Task from id:", feasible_seed)
        o, info = env.reset(seed=feasible_seed)
        while (
                not planner.check_is_start_feasible(o["achieved_goal"], info["obstacles_info"], info["map_vertices"], check_planner_config)
                or not env.unwrapped.check_goal_with_lidar_detection_one_hot_modified()
                or not env.unwrapped.check_start_with_lidar_detection_one_hot_modifed()
            ):
            feasible_seed += 1
            o, info = env.reset(seed=feasible_seed)
        generate_feasible_number += 1
        print("Finish Finding a New Feasible Task from id:", feasible_seed)
        task_dict = {
            "goal": o["desired_goal"],
            "obstacles_info": info["obstacles_info"], 
        }
        task_list.append(task_dict)
        # env.unwrapped.real_render()
    with open("datasets/10task_list_evaluation_5000_new2.pkl", "wb") as f:
        pickle.dump(task_list, f)
    
    
    

if __name__ == "__main__":
    main()
    
