import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint

from tractor_trailer_envs import register_tt_envs

# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs


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
    return gym.make('tt-parking-v0', config=config)

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
        "verbose": args.verbose,
        "outer_wall_bound": {
            "x_min": -50, #[m]
            "x_max": 50,
            "y_min": -50,
            "y_max": 50,
        },
        "start_region_bound": {
            "x_min": -10, #[m]
            "x_max": 10,
            "y_min": -10,
            "y_max": 10,
        },
    }
    register_tt_envs()
    env = gym.make("tt-parking-v0", config=config)
    her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3 import SAC as SAC_stable
    model = SAC_stable('MultiInputPolicy', env, verbose=1, 
                tensorboard_log="runs_stable_rl_tt", 
                buffer_size=int(1e6),
                learning_rate=1e-3,
                learning_starts=1000,
                gamma=0.95, batch_size=1024, tau=0.05,
                policy_kwargs=dict(net_arch=[512, 512, 512]),
                seed=206)
    LEARNING_STEPS = 4e6 # @param {type: "number"}
    model.learn(int(LEARNING_STEPS), tb_log_name="sac")
    model.save("stable_rl_save/")
    print(1)

if __name__ == "__main__":
    main()
    
