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
    # parser = get_config()
    # args = parser.parse_args()
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
    # #TODO: change the api
    # config = {
    #     "env_name": args.env_name,
    #     "reward_type": args.reward_type,
    #     "xmax": args.xmax, 
    #     "ymax": args.ymax, # [m]
    #     "distancematrix": args.distance_weights,
    #     "reward_weights": args.reward_weights,
    #     "max_episode_steps": args.max_episode_steps,
    #     "goal": tuple(args.goal),
    #     "penalty_backward": args.penalty_backward,
    #     "penalty_switch": args.penalty_switch,
    #     "edge": args.edge,
    #     "save_gif": args.save_gif,
    #     "allow_backward": args.allow_backward,
    #     "constraint_coeff": args.constraint_coeff,
    #     "sucess_goal_reward_parking": args.sucess_goal_reward_parking,
    #     "sucess_goal_reward_others": args.sucess_goal_reward_others,
    #     "continuous_step": args.continuous_step,
    # }
    with open("configs/envs/reaching_v0.yaml", 'r') as file:
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
    agent.run()
        
    print(1)

if __name__ == "__main__":
    main()
    