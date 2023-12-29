import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint

from tractor_trailer_envs import register_tt_envs

# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs
from collections import deque
import numpy as np
import rl_agents as rl
from stable_baselines3 import HerReplayBuffer
from stable_baselines3 import SAC as SAC_stable
import yaml
def find_closed_goal(goals_list, sample_goal):
    """
    计算并返回 goals_list 中离 sample_goal 最近的元素。

    :param goals_list: 包含多个六维度元组的列表或deque。
    :param sample_goal: 一个六维度的元组。
    :return: goals_list 中离 sample_goal 最近的元素。
    """
    # 将 sample_goal 转换为 NumPy 数组
    sample_goal_array = np.array(sample_goal)

    # 初始化最小距离和最接近的元素
    min_distance = float('inf')
    closest_goal = None

    # 遍历 goals_list 中的每个元素
    for goal in goals_list:
        # 将当前目标转换为 NumPy 数组
        goal_array = np.array(goal)

        # 计算欧式距离
        distance = np.linalg.norm(sample_goal_array - goal_array)

        # 检查是否是更接近的目标
        if distance < min_distance:
            min_distance = distance
            closest_goal = goal

    return closest_goal

def get_first_n_divided_point(closest_goal, sample_goal, n):
    """
    在 closest_goal 和 sample_goal 之间找到离 closest_goal 更近的第一个 n 等分点。

    :param closest_goal: 六维元组，表示一个点。
    :param sample_goal: 六维元组，表示另一个点。
    :param n: 整数，表示等分点的数量。
    :return: 第一个 n 等分点的坐标（元组形式）。
    """
    # 将元组转换为 NumPy 数组
    closest_goal_array = np.array(closest_goal)
    sample_goal_array = np.array(sample_goal)

    # 计算第一个 n 等分点
    divided_point = closest_goal_array + (1 / n) * (sample_goal_array - closest_goal_array)

    # 将 NumPy 数组转换回元组
    divided_point_tuple = tuple(divided_point)

    return divided_point_tuple

def goals_list_projection(goals_list, sample_goal, n):
    closest_goal = find_closed_goal(goals_list, sample_goal)
    return get_first_n_divided_point(closest_goal, sample_goal, n)



def main():
    her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future', copy_info_dict=True)
    parser = rl.get_config()
    args = parser.parse_args()
    # #TODO: change the api
    # config = {
    #     "vehicle_type": "one_trailer",
    #     "reward_type": args.reward_type,
    #     # "distancematrix": args.distance_weights,
    #     # "reward_weights": args.reward_weights,
    #     "max_episode_steps": args.max_episode_steps,
    #     # "goal": tuple(args.goal),
    #     "evaluate_mode": args.evaluate_mode,
    #     "allow_backward": args.allow_backward,
    #     "sucess_goal_reward_parking": args.sucess_goal_reward_parking,
    #     "sucess_goal_reward_others": args.sucess_goal_reward_others,
    #     "verbose": args.verbose,
    #     "outer_wall_bound": {
    #         "x_min": -80, #[m]
    #         "x_max": 80,
    #         "y_min": -80,
    #         "y_max": 80,
    #     },
    #     "start_region_bound": {
    #         "x_min": 0, #[m]
    #         "x_max": 0,
    #         "y_min": 0,
    #         "y_max": 0,
    #     },
    #     "goal_region_bound": {
    #             "x_min": -10, #[m]
    #             "x_max": 10,
    #             "y_min": -10,
    #             "y_max": 10,
    #         },
    #     "controlled_vehicle_config": {
    #             "w": 2.0, #[m] width of vehicle
    #             "wb": 3.5, #[m] wheel base: rear to front steer
    #             "wd": 1.4, #[m] distance between left-right wheels (0.7 * W)
    #             "rf": 4.5, #[m] distance from rear to vehicle front end
    #             "rb": 1.0, #[m] distance from rear to vehicle back end
    #             "tr": 0.5, #[m] tyre radius
    #             "tw": 1.0, #[m] tyre width
    #             "rtr": 2.0, #[m] rear to trailer wheel
    #             "rtf": 1.0, #[m] distance from rear to trailer front end
    #             "rtb": 3.0, #[m] distance from rear to trailer back end
    #             "rtr2": 2.0, #[m] rear to second trailer wheel
    #             "rtf2": 1.0, #[m] distance from rear to second trailer front end
    #             "rtb2": 3.0, #[m] distance from rear to second trailer back end
    #             "rtr3": 2.0, #[m] rear to third trailer wheel
    #             "rtf3": 1.0, #[m] distance from rear to third trailer front end
    #             "rtb3": 3.0, #[m] distance from rear to third trailer back end   
    #             "max_steer": 0.4, #[rad] maximum steering angle
    #             "v_max": 2.0, #[m/s] maximum velocity 
    #             "safe_d": 0.0, #[m] the safe distance from the vehicle to obstacle 
    #             "xi_max": (np.pi) / 4, # jack-knife constraint  
    #         },
    # }
    # with open('config.yaml', 'w') as file:
    #     yaml.dump(config, file)
    with open("configs/envs/reaching_v0.yaml", 'r') as file:
        config = yaml.safe_load(file)
    register_tt_envs()
    env = gym.make("tt-reaching-v0", config=config)
    goals_for_training = [(a, 0.0, 0.0, 0.0, 0.0, 0.0) for a in np.arange(-10, -1, 0.1)] + [(a, 0.0, 0.0, 0.0, 0.0, 0.0) for a in np.arange(1, 10, 0.1)]
    model = SAC_stable('MultiInputPolicy', env, verbose=1, 
                tensorboard_log="runs_stable_rl_tt_reaching/", 
                buffer_size=int(1e6),
                learning_rate=1e-3,
                learning_starts=1000,
                gamma=0.95, batch_size=1024, tau=0.05,
                policy_kwargs=dict(net_arch=[512, 512, 512]),
                seed=20)
    
    # model = SAC_stable('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
    #         replay_buffer_kwargs=her_kwargs, verbose=1, 
    #         tensorboard_log="runs_stable_rl_tt", 
    #         buffer_size=int(1e6),
    #         learning_rate=1e-3,
    #         learning_starts=1000,
    #         gamma=0.95, batch_size=1024, tau=0.05,
    #         policy_kwargs=dict(net_arch=[512, 512, 512]),
    #         seed=60)
    LEARNING_STEPS = 5e3 # @param {type: "number"}
    
    # goals_list
    env.unwrapped.update_goal_list(goals_for_training)
    model.learn(int(LEARNING_STEPS), tb_log_name="one_trailer_straight", reset_num_timesteps=True)
    
    model.save("train_straight_sparse/")
    
    print(1)

if __name__ == "__main__":
    main()
    
