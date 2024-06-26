import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import rl_agents as agents
import tractor_trailer_envs as tt_envs
from tractor_trailer_envs import register_tt_envs
from datetime import datetime
from rl_agents.sac import core

def gym_reaching_tt_env_fn(config: dict): 
    return gym.make("tt-reaching-v0", config=config)

def gym_meta_reaching_tt_env_fn(config: dict):
    return gym.make("tt-meta-reaching-v0", config=config)

# Apply our new-tt planning env to this agent for training
def gym_tt_planning_env_fn(config: dict):
    return gym.make("tt-planning-v0", config=config)

def main():
    with open("configs/agents/training/explore.yaml", 'r') as file:
        config_algo = yaml.safe_load(file)
    agent = agents.DIAYN(env_fn=gym_tt_planning_env_fn,
        config=config_algo)
    agent.run()
        
    print(1)

if __name__ == "__main__":
    main()
    
