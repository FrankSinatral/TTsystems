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
    addtional_astar_dataset = False
    config_filename = "configs/agents/training/rl0.yaml"
    if addtional_astar_dataset:
        task_runner_configfile = "configs/envs/task_runner.yaml"
        with open(task_runner_configfile, 'r') as file:
            taskrunner_config = yaml.safe_load(file)
        task_runner = agents.TaskRunner(env_fn=gym_tt_planning_env_fn,
            config=taskrunner_config)
        tasks, astar_results = task_runner.load_astar_results("datasets/data")
    with open(config_filename, 'r') as file:
        config_algo = yaml.safe_load(file)
    agent = agents.SAC_ASTAR_META_NEW(env_fn=gym_tt_planning_env_fn,
        config=config_algo)
    if addtional_astar_dataset:
        agent.load_astar_results(tasks, astar_results)
    agent.run()
        
    print(1)

if __name__ == "__main__":
    main()
    
