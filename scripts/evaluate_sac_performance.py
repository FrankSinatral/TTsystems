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
import numpy as np
import pickle

# from rl_training.config import get_config

def tt_env_fn(config: dict = None, args = None): 
    return tt_envs.TractorTrailerEnv(config, args)

    
def gym_env_fn():
    return gym.make("parking-v0", render_mode="rgb_array")

def gym_reaching_tt_env_fn(config: dict): 
    return gym.make("tt-reaching-v0", config=config)

def get_current_time_format():
    # get current time
    current_time = datetime.now()
    # demo: 20230130_153042
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_time

def main():
    with open("configs/agents/sac_astar.yaml", 'r') as file:
        config_algo = yaml.safe_load(file)
    
    env_name = config_algo['env_name']
    seed = config_algo['seed']
    exp_name = env_name + '_' + config_algo['algo_name'] + '_' + str(seed) + '_' + get_current_time_format()
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
    with open("configs/envs/reaching_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    agent = agents.SAC_ASTAR(env_fn=gym_reaching_tt_env_fn,
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
                config=config)
    agent.load('runs_rl/reaching-v0_sac_astar_three_trailer_50_20240129_230239/model_4949999.pth', whether_load_buffer=False)
    
    # agent.ac_targ.q1()
    
    average_ep_ret = 0.0
    average_ep_len = 0.0
    success_rate = 0.0
    jack_knife_rate = 0.0
    count = 0
    # start_list = []
    # for _ in range(100):
    #     # fix some point
    #     epsilon = 1e-6
    #     random_angle = np.random.uniform(-np.pi/4 + epsilon, np.pi/4 - epsilon)
    #     start = [0.0, 0.0, 0.0, random_angle, 0.0, 0.0]
    #     start_list.append(start)
    # agent.test_env.unwrapped.update_start_list(start_list)  
    goal_list = []
    # for _ in range(100):
    #     epsilon = 1e-6
    #     random_number_x = np.random.uniform(10, 15)
    #     random_number_y = np.random.uniform(-15, 15)
    #     random_angle = np.random.uniform(-np.pi + epsilon, np.pi - epsilon)
    #     goal = [random_number_x, random_number_y, random_angle, random_angle, 0.0, 0.0]
    #     goal_list.append(goal)
    # agent.test_env.unwrapped.update_goal_list(goal_list)
    # failed_o_list = []
    result_list = []
    for j in range(0, 1000):
        count += 1
        # goal_list = 
        # agent.test_env.unwrapped.update_goal_list(goal_list)
        o, info = agent.test_env.reset(seed=j)
        init_o = o
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        while not(terminated or truncated):
            # Take deterministic actions at test time 
            a = agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), True)
            o, r, terminated, truncated, info = agent.test_env.step(a)
            ep_ret += r
            ep_len += 1
        average_ep_ret += ep_ret
        average_ep_len += ep_len
        if info['is_success']:
            success_rate += 1
            print("success:", j)
        if info['jack_knife']:
            jack_knife_rate += 1
            print("jack_knife:", j)
        # if not info['is_success']:
        #     failed_o_list.append(init_o)
        single_dict = {
            "state_list": agent.test_env.unwrapped.state_list,
            "action_list": agent.test_env.unwrapped.action_list,
            "info": info,
        }
        result_list.append(single_dict)
        # agent.test_env.unwrapped.run_simulation()
    if not os.path.exists("rl_training/datas"):
        os.makedirs("rl_training/datas")
    with open("rl_training/datas/file.pickle", 'wb') as f:
        pickle.dump(result_list, f)   
    jack_knife_rate /= count
    success_rate /= count
    print("success_rate:", success_rate)
    print("jack_knife_rate:", jack_knife_rate)    

if __name__ == "__main__":
    main()
    
