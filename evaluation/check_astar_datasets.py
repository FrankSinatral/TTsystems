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
from config import get_config
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
    dataset_filename = "datasets/astar_result_obstacle_0_pickle/astar_result_lidar_detection_one_hot_triple_98.pkl"
    
    
    time1 = time.time()
    
    with open(dataset_filename, "rb") as f:
        results = pickle.load(f)
    time2 = time.time()
    print("Time: ", time2 - time1)
    task_list = results["tasks"]
    converted_task_list = convert_tuple_to_dict(task_list)  
    astar_result_list = results["results"]
    with open("configs/envs/tt_planning_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-planning-v0", config=config)
    planner_config = {
        "plot_final_path": True,
        "plot_rs_path": True,
        "plot_expand_tree": True,
        "mp_step": 10, # Important
        "N_steps": 10, # Important
        "range_steer_set": 20,
        "max_iter": 10,
        "heuristic_type": "mix",
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
    
    count = 0
    count_failed = 0
    for j in range(len(converted_task_list)):
        use_task_list = [converted_task_list[j]]
        env.unwrapped.update_task_list(use_task_list)
        obs, info = env.reset()
        
        
        astar_result = astar_result_list[j]
        if astar_result.get("goal_reached", False):
            print("Goal reached")
            state_list = astar_result["state_list"]
            control_list = astar_result["control_list"]
            if len(control_list) > 400:
                count += 1 
                print("len of control list:", len(control_list))
                for i in range(0, len(control_list), 10):
                    control = control_list[i]
                    obs, reward, done, truncted, info = env.step(control)
                    env.unwrapped.real_render()
        else:
            env.unwrapped.real_render()
            print("Goal not reached")
            count_failed += 1
    print("length > 400:", count)
    print("failed counting:", count_failed)
    
    

if __name__ == "__main__":
    main()
    
