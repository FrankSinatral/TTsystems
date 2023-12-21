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


def main():
    
    parser = get_config()
    args = parser.parse_args()
    config = {
            "evaluate_mode": args.evaluate_mode,
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
                "max_steer": 0.4, #[rad] maximum steering angle
                "v_max": 10.0, #[m/s] maximum velocity 
                "safe_d": 0.0, #[m] the safe distance from the vehicle to obstacle 
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
            "goal_region_bound": {
                "x_min": -10, #[m]
                "x_max": 10,
                "y_min": -10,
                "y_max": 10,
            },
        }
    goals_list = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
                #   (3.0, 8.0, 0.0, 0.0, 0.0, 0.0),
                #   (2.0, 10.0, 0.0, 0.0, 0.0, 0.0)]
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-reaching-v0", config=config)
    obs, _ = env.reset(seed=20)
    goals_list = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                  (3.0, 8.0, 0.0, 0.0, 0.0, 0.0),
                  (2.0, 10.0, 0.0, 0.0, 0.0, 0.0)]
    env.unwrapped.sample_from_space(seed=30)
    env.unwrapped.update_goal_list(goals_list)
    obs, _ = env.reset(seed=20)
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
    
