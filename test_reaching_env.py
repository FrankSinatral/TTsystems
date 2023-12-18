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



def main():
    
    parser = get_config()
    args = parser.parse_args()
    config = {
            "evaluate_mode": args.evaluate_mode,
        }
    goals_list = [(4.0, 5.0, 0.0, 0.0, 0.0, 0.0),
                  (3.0, 8.0, 0.0, 0.0, 0.0, 0.0),
                  (2.0, 10.0, 0.0, 0.0, 0.0, 0.0)]
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-reaching-v0", config=config)
    obs, _ = env.reset(seed=20, goals=goals_list)
    env.action_space.seed(seed=20)
    terminated, truncated = False, False
    ep_ret = 0.0
    while (not terminated) and (not truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward   
    env.unwrapped.run_simulation() 

if __name__ == "__main__":
    main()
    
