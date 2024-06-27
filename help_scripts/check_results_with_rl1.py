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

def clear_obstacles(goal_with_obstacles_info_list):
    for goal_with_obstacles_info in goal_with_obstacles_info_list:
        goal_with_obstacles_info["obstacles_info"] = []
    return 

def shift_to_goal_list(goal_with_obstacles_info_list):
    goal_list = []
    for goal_with_obstacles_info in goal_with_obstacles_info_list:
        goal_list.append(goal_with_obstacles_info["goal"])
    return goal_list
    
def main():
    use_datasets = True
    # the script is for evaluating rl1's result
    # you can choose to evaluate the first 100's data on the dataset
    # or choose randomly from the env
    with open("datasets/goal_with_obstacles_info_list_hz.pickle", "rb") as f:
        datasets = pickle.load(f)
    # with open("datasets/all_failed_cases_rl0.pickle", "rb") as f:
    #     datasets = pickle.load(f)
    # with open("datasets/all_failed_cases_rl1_obs2_linear.pickle", "rb") as f:
    #     datasets = pickle.load(f)
    with open("configs/agents/eval/rl1_env.yaml", "r") as f:
        # remember here you have to set the same as the training
        config = yaml.safe_load(f)
        
    # with open("configs/agents/eval/rl1_one_hot.yaml", "r") as f:
    with open("configs/agents/eval/rl1_lidar_detection_one_hot.yaml", "r") as f:
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
    # filename = 'runs_rl/meta-reaching-v0_sac_astar_meta_three_trailer_10_20240401_223636/model_2499999.pth'
    
    # dataset training
    # filename = 'runs_rl/meta-reaching-v0_rl1_one_hot_three_trailer_10_20240410_004916/model_final.pth'
    # filename = "datasets/models/one_hot_model0.pth"
    # filename = "datasets/models/one_hot_model2.pth"
    filename = "datasets/models/lidar_detection_one_hot_model2.pth"
    
    
    # # number_obstacles=1 + linear penalty
    # filename = "runs_rl/meta-reaching-v0_rl1_one_hot_three_trailer_10_20240410_004846/model_1999999.pth"
    
    # # number_obstacles=2 + linear penalty
    # filename = "runs_rl/meta-reaching-v0_rl1_one_hot_three_trailer_10_20240409_190631/model_4499999.pth"
    
    
    
    agent.load(filename, whether_load_buffer=False)
    
    if use_datasets:
        goal_with_obstacles_info_list = []
        for data in datasets:
            goal = data["goal"]
            obstacles_info = data["obstacles_info"]
            task_dict = {
                "goal": goal,
                "obstacles_info": obstacles_info
            }
            goal_with_obstacles_info_list.append(task_dict) 
    
    failed_cases = []
    failed_number = 0
    for i in range(11, 12):
        # goal_list = [goal_with_obstacles_info_list[i]["goal"]]
        # obstacles_info = goal_with_obstacles_info_list[i]["obstacles_info"]
        if use_datasets:
            now_goal_with_obstacles_info_list = [goal_with_obstacles_info_list[11]]
            # now_goal_with_obstacles_info_list = clear_obstacles(now_goal_with_obstacles_info_list)
            goal_list = shift_to_goal_list(now_goal_with_obstacles_info_list)
            env.unwrapped.update_goal_list(goal_list)
            # env.unwrapped.update_goal_with_obstacles_info_list(now_goal_with_obstacles_info_list)
        
        o, info = env.reset()
        # env.real_render()
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        while not(terminated or truncated):
            # Take deterministic actions at test time
            # o, r, terminated, truncated, info = env.step(agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o["one_hot_representation"]]), True)) # one_hot_representation
            o, r, terminated, truncated, info = env.step(agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o["lidar_detection_one_hot"]]), True)) # lidar_detection_one_hot
            ep_ret += r
            ep_len += 1
        env.unwrapped.run_simulation()
        # env.unwrapped.save_result()
        print(f"Episode {i}: Success - {info['is_success']}")
        if not info['is_success']:
            failed_dict = {
                "goal": o["desired_goal"],
                "obstacles_info": info["obstacles_info"],
            }
            failed_cases.append(failed_dict)
            failed_number += 1
    print("failed number: ", failed_number)
    #     env.unwrapped.save_result()
    # # Save all failed cases to a single file
    # with open('datasets/all_failed_cases_rl1_obs2_linear.pickle', 'wb') as f:
    #     pickle.dump(failed_cases, f)
    
    
        
        
if __name__ == "__main__":
    main()
    