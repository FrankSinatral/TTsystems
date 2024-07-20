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
    # file to check
    dataset_filename = "datasets/data/astar_result_obstacle_10_pickle/astar_result_lidar_detection_one_hot_triple_113.pkl"
    
    with open(dataset_filename, "rb") as f:
        results = pickle.load(f)
    
    task_list = results["tasks"]
    converted_task_list = convert_tuple_to_dict(task_list)  
    astar_result_list = results["results"]
    # initiliaze an env for checking dataset quaility
    with open("configs/envs/tt_planning_v0_check_datasets.yaml", 'r') as file:
        config = yaml.safe_load(file)
    env = gym.make("tt-planning-v0", config=config)
    
    count = 0
    count_failed = 0
    for j in range(len(converted_task_list)):
        use_task_list = [converted_task_list[j]]
        env.unwrapped.update_task_list(use_task_list)
        obs, info = env.reset()
        env.unwrapped.real_render()
        
        astar_result = astar_result_list[j]
        if astar_result.get("goal_reached", False):
            print("Goal reached")
            # state_list = astar_result["state_list"]
            control_list = astar_result["control_list"]
            # plot the success and length > 400 case
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
    
