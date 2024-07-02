import sys
import os

import yaml
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import rl_agents as agents
# Apply our new-tt planning env to this agent for training
def gym_tt_planning_env_fn(config: dict):
    import gymnasium as gym
    import tractor_trailer_envs as tt_envs
    if not hasattr(gym_tt_planning_env_fn, "envs_registered"):
        tt_envs.register_tt_envs()
        gym_tt_planning_env_fn.envs_registered = True
    return gym.make("tt-planning-v0", config=config)

def main():
    config_filename = "configs/envs/task_runner.yaml"
    with open(config_filename, 'r') as file:
        config_algo = yaml.safe_load(file)
    task_runner = agents.TaskRunner(env_fn=gym_tt_planning_env_fn,
        config=config_algo)
    task_runner.run()   
    print(1)

if __name__ == "__main__":
    main()