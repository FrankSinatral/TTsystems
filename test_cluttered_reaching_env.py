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

def main():
    with open('planner_result/datas_for_random_map/failed_result_0.pkl', 'rb') as f:
        data = pickle.load(f)
    
    data = data[0]
    goal = data["goal"]
    obstacles_info = data["obstacles_info"]
    goal_with_obstacls_info = {
        "goal": goal,
        "obstacles_info": obstacles_info
    }
    goal_with_obstacls_info_list = [goal_with_obstacls_info]
    
    with open("configs/envs/cluttered_reaching_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-cluttered-reaching-v0", config=config)
    env.unwrapped.update_goal_with_obstacles_info_list(goal_with_obstacls_info_list)
    obs, _ = env.reset(seed=40)
    env.action_space.seed(seed=40)
    terminated, truncated = False, False
    ep_ret = 0.0
    while (not terminated) and (not truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.unwrapped.render()
        ep_ret += reward   
    env.unwrapped.run_simulation() 

if __name__ == "__main__":
    main()
    
