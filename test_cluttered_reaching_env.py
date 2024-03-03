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
    # Load from other file
    with open('planner_result/datas_for_random_map/failed_result_0.pkl', 'rb') as f:
        datas = pickle.load(f)
    goal_with_obstacles_info_list = []
    for i in range(len(datas)):
        data = datas[i]
        goal = data["goal"]
        obstacles_info = data["obstacles_info"]
        goal_with_obstacles_info = {
            "goal": goal,
            "obstacles_info": obstacles_info
        }
        goal_with_obstacles_info_list.append(goal_with_obstacles_info)
    
    with open("configs/envs/cluttered_reaching_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-cluttered-reaching-v0", config=config)
    env.unwrapped.update_goal_with_obstacles_info_list(goal_with_obstacles_info_list)
    for j in range(100):
        t1 = time.time()
        obs, _ = env.reset(seed=(40 + j))
        env.action_space.seed(seed=(40 + j))
        terminated, truncated = False, False
        ep_ret = 0.0
        while (not terminated) and (not truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # env.unwrapped.render()
            ep_ret += reward   
        # env.unwrapped.run_simulation()
        t2 = time.time()
        print("Time: ", t2 - t1)
        
    

if __name__ == "__main__":
    main()
    
