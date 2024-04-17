import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import torch.nn as nn
import gymnasium as gym
import pprint


# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs

from tractor_trailer_envs import register_tt_envs
register_tt_envs()
from config import get_config
import numpy as np
import yaml
import pickle
import time

# this script is for generating training data for the slot-based model
def main():
    
    
    with open("configs/envs/meta_reaching_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-meta-reaching-v0", config=config)
    results_list = []
    
    for j in range(1000):
        start_time = time.time()
        obs, info = env.reset(seed=j + 1)
        task_dict = {
            "obstacles_info": info["obstacles_info"],
            "image_sequences": [],
            "action_sequences": [],
        }
        
        # gray_image = env.render_obstacles()
        rgb_image = env.unwrapped.render_jingyu_test()
        task_dict["image_sequences"].append(rgb_image)
        
        terminated, truncated = False, False
        while (not terminated) and (not truncated):
            action = env.action_space.sample()
            task_dict["action_sequences"].append(action)
            # env.unwrapped.real_render()
            obs, reward, terminated, truncated, info = env.step(action)
            rgb_image = env.unwrapped.render_jingyu_test()
            task_dict["image_sequences"].append(rgb_image)
        end_time = time.time()
        print("time:", end_time - start_time)
        results_list.append(task_dict)
    with open("datasets/slotdatasets_only_vehicle.pickle", "wb") as f:
        pickle.dump(results_list, f)
        
    # env.unwrapped.real_render()
    print(1)
    
    

if __name__ == "__main__":
    main()
    
