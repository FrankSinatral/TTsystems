# The script is for meta RL training rendering
# to see the effects of the trained model
import tractor_trailer_envs as tt_envs
import pprint
import gymnasium as gym
import rl_agents as agents
import yaml
import torch.nn as nn
import numpy as np
from datetime import datetime
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
    with open("configs/envs/meta_reaching_v0_eval.yaml", "r") as f:
        # remember here you have to set the same as the training
        config = yaml.safe_load(f)
        
    with open("configs/agents/sac_astar_meta_eval.yaml", "r") as f:
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
                astar_mp_steps=config_algo['astar_mp_steps'],
                astar_N_steps=config_algo['astar_N_steps'],
                astar_max_iter=config_algo['astar_max_iter'],
                astar_heuristic_type=config_algo['astar_heuristic_type'],
                config=config_algo['env_config'])
    filename = 'runs_rl/meta-reaching-v0_sac_astar_meta_three_trailer_10_20240325_134532/model_999999.pth'
    agent.load(filename, whether_load_buffer=False)
    
    for i in range(100, 200):
        o, info = env.reset(seed=i)
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        while not(terminated or truncated):
            # Take deterministic actions at test time
            o, r, terminated, truncated, info = env.step(agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['collision_metric']]), True))
            ep_ret += r
            ep_len += 1
        env.unwrapped.run_simulation()
        
        
if __name__ == "__main__":
    main()
# from config import get_config
# parser = get_config()
# args = parser.parse_args()
# config = {
#             "vehicle_type": args.vehicle_type,
#             "reward_type": args.reward_type,
#             "xmax": args.xmax, 
#             "ymax": args.ymax, # [m]
#             "distancematrix": args.distance_weights,
#             "reward_weights": args.reward_weights,
#             "max_episode_steps": args.max_episode_steps,
#             "goal": tuple(args.goal),
#             "evaluate_mode": args.evaluate_mode,
#             "allow_backward": args.allow_backward,
#             "constraint_coeff": args.constraint_coeff,
#             "sucess_goal_reward_parking": args.sucess_goal_reward_parking,
#             "sucess_goal_reward_others": args.sucess_goal_reward_others,
#             "use_stable_baseline": args.use_stable_baseline,
#             "verbose": args.verbose,
#         }
# env = tt_envs.TractorTrailerParkingEnv(config)
# # env.seed(seed=40)
# env.action_space.seed(seed=40)
# for _ in range(3):
#     obs , _ = env.reset()
#     # state_list = [state]
#     reward_list = []
#     terminated = False
#     truncated = False
#     while (not terminated) and (not truncated):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, _ = env.step(action)
#         # env.render()
#         reward_list.append(reward) 
#         # state_list.append(state)
# print(1)
# # env.run_simulation()
    