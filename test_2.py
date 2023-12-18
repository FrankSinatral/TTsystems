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
    env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-parking-v1", config=config)
    obs, _ = env.reset(seed=20)
    env.action_space.seed(seed=20)
    terminated, truncated = False, False
    ep_ret = 0.0
    while (not terminated) and (not truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward   
    env.unwrapped.run_simulation() 
    
    # her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')
    # from stable_baselines3 import HerReplayBuffer
    # from stable_baselines3 import SAC as SAC_stable
    # # model = SAC_stable('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
    # #             replay_buffer_kwargs=her_kwargs,
    # #             verbose=1, 
    # #             tensorboard_log="runs_stable_rl_tt_new", 
    # #             buffer_size=int(1e6),
    # #             learning_rate=1e-3,
    # #             learning_starts=500,
    # #             gamma=0.95, batch_size=1024, tau=0.05,
    # #             policy_kwargs=dict(net_arch=[512, 512, 512]))
    # model = SAC_stable('MultiInputPolicy', env,
    #             verbose=1, 
    #             tensorboard_log="runs_stable_rl_tt", 
    #             buffer_size=int(1e6),
    #             learning_rate=1e-3,
    #             gamma=0.95, batch_size=1024, tau=0.05,
    #             policy_kwargs=dict(net_arch=[512, 512, 512]))
    # LEARNING_STEPS = 4e6 # @param {type: "number"}
    # model.learn(int(LEARNING_STEPS), tb_log_name="sac")
    # if args.vehicle_type == "single_tractor":
    #     model.save("stable_baseline_single_tractor_v1/")
    # elif args.vehicle_type == "one_trailer":
    #     model.save("stable_baseline_one_trailer_v1/")
    # elif args.vehicle_type == "two_trailer":
    #     model.save("stable_baseline_two_trailer_v1/")
    # else:
    #     model.save("stable_baseline_three_trailer_v1/")
    # print(1)

if __name__ == "__main__":
    main()
    
