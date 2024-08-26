import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs

from tractor_trailer_envs import register_tt_envs
register_tt_envs()
import numpy as np
import yaml
import pickle
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import planner
from joblib import Parallel, delayed
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
    
def process_task(j, env, planner_config):
    obs, info = env.reset(seed=j + 1)
    result_dict = planner.find_astar_trajectory(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], planner_config)
    if len(result_dict.get('control_list', [])) > 0 and result_dict.get('goal_reached', False):
        task_dict = {
            "goal": obs["desired_goal"],
            "obstacles_info": info["obstacles_info"]
        }
        return task_dict
    else:
        return None
    
def process_task_unsolved(j, env, planner_config):
    obs, info = env.reset(seed=j + 1)
    result_dict = planner.find_astar_trajectory(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], planner_config)
    if len(result_dict.get('control_list', [])) > 0 and result_dict.get('goal_reached', False):
        return None 
    else:
        task_dict = {
            "goal": obs["desired_goal"],
            "obstacles_info": info["obstacles_info"]
        }
        return task_dict
    
def process_task_unsolved_new(j, task_list, env, planner_config):
    use_task_list = [task_list[j]]
    env.unwrapped.update_task_list(use_task_list)
    obs, info = env.reset()
    result_dict = planner.find_astar_trajectory(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], planner_config)
    if len(result_dict.get('control_list', [])) > 0 and result_dict.get('goal_reached', False):
        task_dict = {
            "goal": obs["desired_goal"],
            "obstacles_info": info["obstacles_info"]
        }
        return task_dict 
    else:
        return None 
def convert_tuple_to_dict(task_list):
    converted_list = []
    for task in task_list:
        task_dict = {
            "start": task[0],
            "goal": task[1],
            "obstacles_info": task[2],
            "map_vertices": task[3],
            "obstacles_properties": task[4],
            "map_properties": task[5]
        }
        converted_list.append(task_dict)
    return converted_list   

def main():
    with open("configs/envs/tt_planning_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-planning-v0", config=config)
    planner_config = {
        "plot_final_path": True,
        "plot_rs_path": True,
        "plot_expand_tree": True,
        "mp_step": 10, # Important
        "N_steps": 10, # Important
        "range_steer_set": 20,
        "max_iter": 20,
        "heuristic_type": "mix_original",
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
    
    with open("datasets/10task_list.pkl", "rb") as f:
        task_list = pickle.load(f)
    task_dict = {
        "goal": np.array([0, -30, np.pi/2, np.pi/2, np.pi/2, np.pi/2], dtype=np.float32),
        "obstacles_info": [[[-20, -20], [20, -20], [20, -10], [-20, -10]]]
    }
    
    dataset_filename = "datasets/data/test_success_on_fixed_tasks/astar_result_lidar_detection_one_hot_triple_6.pkl"
    
    with open(dataset_filename, "rb") as f:
        results = pickle.load(f)
    
    task_list = results["tasks"]
    astar_result_list = results["results"]
    converted_task_list = convert_tuple_to_dict(task_list)  
    failed_case = 0
    for j in range(len(converted_task_list)):
        use_task_list = [converted_task_list[j]]
        env.unwrapped.update_task_list(use_task_list)
        obs, info = env.reset()
        astar_result = astar_result_list[j]
        
        if not astar_result.get("goal_reached", False):
            failed_case += 1
            env.unwrapped.real_render()
        # goal_check_dict = planner.check_and_adjust_goal(obs["desired_goal"], 5, 5, env.unwrapped.ox, env.unwrapped.oy)
            if failed_case % 3 == 0:
                result_dict = planner.find_astar_trajectory(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], planner_config)
        # planner.visualize_planner_final_result(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], result_dict)
    # for j in range(100):
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
    # for j in range(10):
    #     obs, info = env.reset(seed=j + 1)
    #     env.unwrapped.real_render()
    # task_list = Parallel(n_jobs=-1)(delayed(process_task_unsolved_new)(j, task_list, env, planner_config) for j in range(len(task_list)))
    # # task_list = Parallel(n_jobs=-1)(delayed(process_task_unsolved)(j, env, planner_config) for j in range(1000))

    # # Remove None values from the list
    # task_list = [task for task in task_list if task is not None]
    # with open("10task_more_solved_list.pkl", "wb") as f:
    #     pickle.dump(task_list, f)
    # for j in range(200):
    #     obs, info = env.reset(seed=j + 1)
    #     # env.unwrapped.real_render()
    #     result_dict = planner.find_astar_trajectory(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], planner_config)
    #     planner.visualize_planner_final_result(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], result_dict)
    
         
        
    # env.unwrapped.real_render()
    print(1)
    
    

if __name__ == "__main__":
    main()
    
