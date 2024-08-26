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
    with open("datasets/10task_list_evaluation_5000_new2.pkl", 'rb') as f:
        task_list = pickle.load(f)
    
    # # file to check
    # dataset_filename = "datasets/data/test_success_on_fixed_tasks/astar_result_lidar_detection_one_hot_triple_7.pkl"
    
    # with open(dataset_filename, "rb") as f:
    #     results = pickle.load(f)
    
    # task_list = results["tasks"]
    # converted_task_list = convert_tuple_to_dict(task_list)  
    # astar_result_list = results["results"]
    # initiliaze an env for checking dataset quaility
    with open("configs/envs/tt_planning_v0_check_datasets.yaml", 'r') as file:
        config = yaml.safe_load(file)
    env = gym.make("tt-planning-v0", config=config)
    for j in range(len(task_list)):
        now_task_list = [task_list[j]]
        env.unwrapped.update_task_list(now_task_list)
        o, info = env.reset()
        env.unwrapped.real_render()
    
    

if __name__ == "__main__":
    main()
    
