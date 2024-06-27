import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint


# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs

from tractor_trailer_envs import register_tt_envs
register_tt_envs()
from config import get_config
import numpy as np
import yaml
import pickle
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import planner

def save_image_from_array(array, filename):
    # Convert the numpy array to an Image object
    img = Image.fromarray(array.transpose(1, 2, 0))

    # Save the image
    img.save(filename)

def process_and_save_image(rgb_image, filename):
    # Transpose the image
    rgb_image = np.transpose(rgb_image, axes=(1, 2, 0))

    # Convert the list to a numpy array
    rgb_image = np.array(rgb_image, dtype=np.float32)

    # Normalize the image data to the range [-1, 1]
    # rgb_image = rgb_image / 255.0 * 2.0 - 1

    print(rgb_image.shape)

    # Convert the numpy array to an Image object
    img = Image.fromarray((rgb_image * 255).astype(np.uint8))

    # Save the image
    img.save(filename)

    # Plot the image
    plt.imshow(rgb_image)
    plt.savefig("plot.png")

def main():
    
    
    # with open("configs/envs/tt_planning_v0_eval.yaml", 'r') as file:
    #     config = yaml.safe_load(file)
    with open("configs/envs/free_big_planning.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-planning-v0", config=config)
    planner_config = {
        "plot_final_path": False,
        "plot_rs_path": False,
        "plot_expand_tree": False,
        "mp_step": 10, # Important
        "N_steps": 10, # Important
        "range_steer_set": 20,
        "max_iter": 50,
        "heuristic_type": "mix",
        "save_final_plot": False,
        "controlled_vehicle_config": {
            "w": 2.0, #[m] width of vehicle
            "wb": 3.5, #[m] wheel base: rear to front steer
            "wd": 1.4, #[m] distance between left-right wheels (0.7 * W)
            "rf": 4.5, #[m] distance from rear to vehicle front end
            "rb": 1.0, #[m] distance from rear to vehicle back end
            "tr": 0.5, #[m] tyre radius
            "tw": 1.0, #[m] tyre width
            "rtr": 2.0, #[m] rear to trailer wheel
            "rtf": 1.0, #[m] distance from rear to trailer front end
            "rtb": 3.0, #[m] distance from rear to trailer back end
            "rtr2": 2.0, #[m] rear to second trailer wheel
            "rtf2": 1.0, #[m] distance from rear to second trailer front end
            "rtb2": 3.0, #[m] distance from rear to second trailer back end
            "rtr3": 2.0, #[m] rear to third trailer wheel
            "rtf3": 1.0, #[m] distance from rear to third trailer front end
            "rtb3": 3.0, #[m] distance from rear to third trailer back end   
            "max_steer": 0.6, #[rad] maximum steering angle
            "v_max": 2.0, #[m/s] maximum velocity 
            "safe_d": 0.0, #[m] the safe distance from the vehicle to obstacle 
            "safe_metric": 3.0, # the safe distance from the vehicle to obstacle
            "xi_max": (np.pi) / 4, # jack-knife constraint  
        },
        "acceptance_error": 0.5,
    }
    # task_list = []
    # task_dict = {
    #     "start": np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
    #     "goal" : None,
    #     "obstacles_info": [[(20.0, 1.0), (22.0, 1.0), (22.0, 3.0), (20.0, 3.0)]],
    #     "map_vertices": None,
    # }
    # # np.array([0, -10, 0, 0, 0, 0], dtype=np.float32)
    # task_list.append(task_dict)
    # env.configure({
    #     "task_list": task_list,
    # })
    
    
    # for j in range(10000):
    #     t1 = time.time()
    #     obs, _ = env.reset(seed=(40 + j))
    #     env.action_space.seed(seed=(40 + j))
    #     terminated, truncated = False, False
    #     ep_ret = 0.0
    #     while (not terminated) and (not truncated):
    #         action = env.action_space.sample()
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         # env.unwrapped.render()
    #         ep_ret += reward   
    #     # env.unwrapped.run_simulation()
    #     t2 = time.time()
    #     print("Time: ", t2 - t1)
    for j in range(7, 10):
        start_time = time.time()
        obs, info = env.reset(seed=j + 1)
        env.unwrapped.real_render()
        result_dict = planner.find_astar_trajectory(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], planner_config)
        planner.visualize_planner_final_result(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], result_dict)
        # gray_image = env.render_obstacles()
        # rgb_image = env.render_jingyu()
        
        # rgb_image = env.render_obstacles()
        # v_image = env.render_vehicles()
        terminated, truncated = False, False
        while (not terminated) and (not truncated):
            action = env.action_space.sample()
            # env.unwrapped.real_render()
            obs, reward, terminated, truncated, info = env.step(action)
            # rgb_image = env.render_vehicles()
        end_time = time.time()
        print("time:", end_time - start_time)
        env.unwrapped.run_simulation()
        
    # env.unwrapped.real_render()
    print(1)
    
    

if __name__ == "__main__":
    main()
    
