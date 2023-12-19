import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint

from tractor_trailer_envs import register_tt_envs

# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs

import numpy as np
import rl_agents as rl
    
def gym_env_fn():
    import gymnasium as gym
    from highway_env import register_highway_envs
    register_highway_envs()
    return gym.make("parking-v0", render_mode="rgb_array")


def gym_tt_env_fn(config):
    import gymnasium as gym
    from tractor_trailer_envs import register_tt_envs
    register_tt_envs()
    return gym.make('tt-reaching-v0', config=config)

def main():
    parser = rl.get_config()
    args = parser.parse_args()
    #TODO: change the api
    config = {
        "vehicle_type": args.env_name,
        "reward_type": args.reward_type,
        # "distancematrix": args.distance_weights,
        # "reward_weights": args.reward_weights,
        "max_episode_steps": args.max_episode_steps,
        # "goal": tuple(args.goal),
        "evaluate_mode": args.evaluate_mode,
        "allow_backward": args.allow_backward,
        "sucess_goal_reward_parking": args.sucess_goal_reward_parking,
        "sucess_goal_reward_others": args.sucess_goal_reward_others,
        "sucess_goal_reward_parking": args.sucess_goal_reward_sparse,
        "verbose": args.verbose,
        "outer_wall_bound": {
            "x_min": -50, #[m]
            "x_max": 50,
            "y_min": -50,
            "y_max": 50,
        },
        "start_region_bound": {
            "x_min": 0, #[m]
            "x_max": 0,
            "y_min": 0,
            "y_max": 0,
        },
        "goal_region_bound": {
                "x_min": -10, #[m]
                "x_max": 10,
                "y_min": -10,
                "y_max": 10,
            },
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
                "v_max": 2.0, #[m/s] maximum velocity 
                "safe_d": 0.0, #[m] the safe distance from the vehicle to obstacle 
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
    }
    register_tt_envs()
    env = gym.make("tt-reaching-v0", config=config)
    her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future', copy_info_dict=True)
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3 import SAC as SAC_stable
    # model = SAC_stable('MultiInputPolicy', env, verbose=1, 
    #             tensorboard_log="runs_stable_rl_tt_reaching", 
    #             buffer_size=int(1e6),
    #             learning_rate=1e-3,
    #             learning_starts=1000,
    #             gamma=0.95, batch_size=1024, tau=0.05,
    #             policy_kwargs=dict(net_arch=[512, 512, 512]),
    #             seed=60)
    
    model = SAC_stable('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs, verbose=1, 
            tensorboard_log="runs_stable_rl_tt_reaching", 
            buffer_size=int(1e6),
            learning_rate=1e-3,
            learning_starts=1000,
            gamma=0.95, batch_size=1024, tau=0.05,
            policy_kwargs=dict(net_arch=[512, 512, 512]),
            seed=60)
    
    LEARNING_STEPS = 4e6 # @param {type: "number"}
    model.learn(int(LEARNING_STEPS), tb_log_name="sac")
    # model.save("stable_rl_save_reaching/")
    print(1)

if __name__ == "__main__":
    main()
    
