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
# from tractor_trailer_envs import register_tt_envs
# register_tt_envs()

# the script is for checking the results of the rl0 model
def gym_meta_reaching_tt_env_fn(config: dict): 
    return gym.make("tt-meta-reaching-v0", config=config)

# Apply our new-tt planning env to this agent for training
def gym_tt_planning_env_fn(config: dict):
    return gym.make("tt-planning-v0", config=config)


def get_current_time_format():
    # get current time
    current_time = datetime.now()
    # demo: 20230130_153042
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_time
def main():
    
    # diayn algo setting  
    with open("configs/agents/eval/explore.yaml", "r") as f:
        # remember here you have to set the same as the training
        config_algo = yaml.safe_load(f)
    
    env = gym.make("tt-planning-v0", config=config_algo["env_config"])
    agent = agents.DIAYN(env_fn=gym_tt_planning_env_fn,
                         config=config_algo)
    
    filename = "runs_rl/planning-v0_diayn_three_trailer_10_20240614_004436/model_final.pth"
    agent.load(filename, whether_load_buffer=False)
    
    
    skill_dim = agent.skill_dim
    skills = np.eye(skill_dim)
    for skill in skills: 
        o, info = env.reset()
        ep_len = 0
        truncated, terminated = False, False
        while not(terminated or truncated):
            # Take deterministic actions at test time
            o, _, _, _, info = env.step(agent.get_action(o['observation'], skill, True))
            ep_len += 1
            terminated = info["crashed"] or info["jack_knife"]
            truncated = ep_len == env.unwrapped.config["max_episode_steps"]
        env.unwrapped.run_simulation_explore()
    
    
        
        
if __name__ == "__main__":
    main()
    