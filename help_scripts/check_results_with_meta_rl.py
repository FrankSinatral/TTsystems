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


def gym_meta_reaching_tt_env_fn(config: dict): 
    return gym.make("tt-meta-reaching-v0", config=config)


def get_current_time_format():
    # get current time
    current_time = datetime.now()
    # demo: 20230130_153042
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_time
def main():
    use_datasets = True
    # the script is for testing and generate datasets
    with open("datasets/goal_with_obstacles_info_list.pickle", "rb") as f:
        datasets = pickle.load(f)
    with open("configs/envs/meta_reaching_v0_eval.yaml", "r") as f:
        # remember here you have to set the same as the training
        config = yaml.safe_load(f)
        
    with open("configs/agents/sac_astar_meta.yaml", "r") as f:
        # remember here you have to set the same as the training
        config_algo = yaml.safe_load(f)
    
        
     
    env_name = config_algo['env_name']
    seed = config_algo['seed']
    vehicle_type = config_algo['env_config']['vehicle_type']
    # vehicle_type = config['vehicle_type']
    exp_name = env_name + '_' + config_algo['algo_name'] + '_' + vehicle_type + '_' + str(seed) + '_' + get_current_time_format()
    logger_kwargs = {
        'output_dir': config_algo['logging_dir'] + exp_name,
        'output_fname': config_algo['output_fname'],
        'exp_name': exp_name,
     }
    if config_algo['activation'] == 'ReLU':
        activation_fn = nn.ReLU
    elif config_algo['activation'] == 'Tanh':
        activation_fn = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation function: {config_algo['activation']}")
    ac_kwargs = {
        "hidden_sizes": tuple(config_algo['hidden_sizes']),
        "activation": activation_fn
    }
    env = gym.make("tt-meta-reaching-v0", config=config)
    agent = agents.SAC_ASTAR_META(env_fn=gym_meta_reaching_tt_env_fn,
                algo=config_algo['algo_name'],
                ac_kwargs=ac_kwargs,
                seed=seed,
                steps_per_epoch=config_algo['sac_steps_per_epoch'],
                epochs=config_algo['sac_epochs'],
                replay_size=config_algo['replay_size'],
                gamma=config_algo['gamma'],
                polyak=config_algo['polyak'],
                lr=config_algo['lr'],
                alpha=config_algo['alpha'],
                batch_size=config_algo['batch_size'],
                start_steps=config_algo['start_steps'],
                update_after=config_algo['update_after'],
                update_every=config_algo['update_every'],
                # missing max_ep_len
                logger_kwargs=logger_kwargs, 
                save_freq=config_algo['save_freq'],
                num_test_episodes=config_algo['num_test_episodes'],
                log_dir=config_algo['log_dir'],
                whether_her=config_algo['whether_her'],
                use_automatic_entropy_tuning=config_algo['use_auto'],
                env_name=config_algo['env_name'],
                pretrained=config_algo['pretrained'],
                pretrained_itr=config_algo['pretrained_itr'],
                pretrained_dir=config_algo['pretrained_dir'],
                whether_astar=config_algo['whether_astar'],
                astar_ablation=config_algo['astar_ablation'],
                config=config_algo['env_config'])
    filename = 'runs_rl/meta-reaching-v0_sac_astar_meta_three_trailer_10_20240401_223636/model_2499999.pth'
    agent.load(filename, whether_load_buffer=False)
    
    if use_datasets:
        goal_with_obstacles_info_list = []
        for data in datasets[:100]:
            goal = data["goal"]
            obstacles_info = data["obstacles_info"]
            task_dict = {
                "goal": goal,
                "obstacles_info": obstacles_info
            }
            goal_with_obstacles_info_list.append(task_dict) 
    
    failed_cases = []
    failed_number = 0
    for i in range(100):
        # goal_list = [goal_with_obstacles_info_list[i]["goal"]]
        # obstacles_info = goal_with_obstacles_info_list[i]["obstacles_info"]
        if use_datasets:
            now_goal_with_obstacles_info_list = [goal_with_obstacles_info_list[i]]
            env.unwrapped.update_goal_with_obstacles_info_list(now_goal_with_obstacles_info_list)
        
        o, info = env.reset(seed=i)
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        while not(terminated or truncated):
            # Take deterministic actions at test time
            o, r, terminated, truncated, info = env.step(agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o["one_hot_representation"]]), True))
            ep_ret += r
            ep_len += 1
        # env.unwrapped.run_simulation()
        print(f"Episode {i}: Success - {info['is_success']}")
        if not info['is_success']:
            failed_cases.append((o, info))
            failed_number += 1
    print("failed number: ", failed_number)
    #     env.unwrapped.save_result()
    # # Save all failed cases to a single file
    # with open('datasets/all_failed_cases.pkl', 'wb') as f:
    #     pickle.dump(failed_cases, f)
    
    
        
        
if __name__ == "__main__":
    main()
    