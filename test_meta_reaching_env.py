import sys
import os
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
def main():
    
    
    with open("configs/envs/meta_reaching_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-meta-reaching-v0", config=config)
    
    
    # for j in range(10000):
    #     t1 = time.time()
    #     obs, _ = env.reset(seed=(40 + j))
    #     env.action_space.seed(seed=(40 + j))
    #     terminated, truncated = False, False
    #     ep_ret = 0.0
    #     while (not terminated) and (not truncated):
    #         action = env.action_space.sample()
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         # env.unwrapped.render()
    #         ep_ret += reward   
    #     # env.unwrapped.run_simulation()
    #     t2 = time.time()
    #     print("Time: ", t2 - t1)
    for j in range(100):
        start_time = time.time()
        obs, _ = env.reset(seed=j + 1)
        # gray_image = env.render_obstacles()
        rgb_image = env.render_jingyu()
        
        terminated, truncated = False, False
        while (not terminated) and (not truncated):
            action = env.action_space.sample()
            # env.unwrapped.real_render()
            obs, reward, terminated, truncated, info = env.step(action)
            rgb_image = env.render_jingyu()
        end_time = time.time()
        print("time:", end_time - start_time)
        
    # env.unwrapped.real_render()
    print(1)
    
    

if __name__ == "__main__":
    main()
    
