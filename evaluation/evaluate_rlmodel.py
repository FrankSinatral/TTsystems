# The script is for meta RL training rendering
# to see the effects of the trained model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import tractor_trailer_envs as tt_envs
import pprint
import gymnasium as gym
import rl_agents as agents
import yaml
import torch.nn as nn
import numpy as np
from datetime import datetime
import pickle
from tractor_trailer_envs import register_tt_envs
register_tt_envs()
def gym_tt_planning_env_fn(config: dict):
    import gymnasium as gym
    import tractor_trailer_envs as tt_envs
    if not hasattr(gym_tt_planning_env_fn, "envs_registered"):
        tt_envs.register_tt_envs()
        gym_tt_planning_env_fn.envs_registered = True
    return gym.make("tt-planning-v0", config=config)

def get_current_time_format():
    # get current time
    current_time = datetime.now()
    # demo: 20230130_153042
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_time
def main():
    
    
    config_filename = "configs/agents/eval/rl0.yaml"
    with open(config_filename, 'r') as file:
        config_algo = yaml.safe_load(file)
    observation_type = config_algo['env_config']['observation']
    whether_attention = config_algo['env_config']['with_obstacles_info']
    agent = agents.SAC_ASTAR_META_NEW(env_fn=gym_tt_planning_env_fn,
        config=config_algo)
    modelpath = "datasets/models/original_model.pth"
    agent.load(modelpath, whether_load_buffer=False)
    with open("configs/envs/tt_planning_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    env = gym.make("tt-planning-v0", config=config)
    
    failed_cases = []
    failed_number = 0
    for i in range(100):
        o, info = env.reset(seed=i)
        # env.unwrapped.real_render()
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        while not(terminated or truncated):
            # Take deterministic actions at test time
            o, r, terminated, truncated, info = env.step(agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), deterministic=True))
            ep_ret += r
            ep_len += 1
        # env.unwrapped.run_simulation()
        # env.unwrapped.save_result()
        print(f"Episode {i}: Success - {info['is_success']}")
        if not info['is_success']:
            failed_cases.append((o, info))
            failed_number += 1
    print("failed number: ", failed_number)
    #     env.unwrapped.save_result()
    # Save all failed cases to a single file
    # with open('datasets/all_failed_cases_rl0.pkl', 'wb') as f:
    #     pickle.dump(failed_cases, f)
    
    
        
        
if __name__ == "__main__":
    main()
    