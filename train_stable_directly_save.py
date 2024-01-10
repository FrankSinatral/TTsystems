import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint
import pickle

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
    
    with open("configs/envs/reaching_v0.yaml", 'r') as file:
        config = yaml.safe_load(file)
    register_tt_envs()
    env = gym.make("tt-reaching-v0", config=config)
    # goals_for_training = [(a, 0.0, 0.0, 0.0, 0.0, 0.0) for a in np.arange(-10, -1, 0.1)] + [(a, 0.0, 0.0, 0.0, 0.0, 0.0) for a in np.arange(1, 10, 0.1)]
    seed = 80
    model = SAC_stable('MultiInputPolicy', env, verbose=1, 
                tensorboard_log="train_one_shot/", 
                buffer_size=int(1e6),
                learning_rate=1e-3,
                learning_starts=1000,
                gamma=0.99, batch_size=1024, tau=0.05,
                policy_kwargs=dict(net_arch=[512, 512, 512]),
                seed=seed)
    # her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future', copy_info_dict=True)
    # model = SAC_stable('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
    #         replay_buffer_kwargs=her_kwargs, verbose=1, 
    #         tensorboard_log="runs_stable_rl_tt_reaching", 
    #         buffer_size=int(1e6),
    #         learning_rate=1e-3,
    #         learning_starts=1000,
    #         gamma=0.95, batch_size=1024, tau=0.05,
    #         policy_kwargs=dict(net_arch=[512, 512, 512]),
    #         seed=20)
    LEARNING_STEPS = int(2e7) # @param {type: "number"}
    
    # LEARNING_STEPS = int(2e7)  # 总学习步数
    SAVE_INTERVAL = int(1e6)  # save interval
    save_dir = "train_one_shot/" + 'one_trailer_' + str(seed) + '/'   # save_dir

    # make sure the save_dir exist
    os.makedirs(save_dir, exist_ok=True)

    for step in range(0, LEARNING_STEPS, SAVE_INTERVAL):
        # 执行一定数量的学习步骤
        model.learn(SAVE_INTERVAL, tb_log_name='one_trailer', reset_num_timesteps=False)

        # 保存模型和重放缓冲区
        model_file = os.path.join(save_dir, f"model_{step + SAVE_INTERVAL}")
        buffer_file = os.path.join(save_dir, f"buffer_{step + SAVE_INTERVAL}")
        
        model.save(model_file)
        with open(buffer_file + '.pkl', 'wb') as f:
            pickle.dump(model.replay_buffer, f)

        print(f"Saved model and buffer at step {step + SAVE_INTERVAL}")

    print("Training completed.")
    
    # # goals_list
    # # env.unwrapped.update_goal_list(goals_for_training)
    # model.learn(int(LEARNING_STEPS), tb_log_name="one_trailer_directly", reset_num_timesteps=True)
    
    # model.save("train_one_shot/")

if __name__ == "__main__":
    main()
    
