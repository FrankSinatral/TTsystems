import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

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
from joblib import Parallel, delayed

def process_task(j, task_list, env, planner_config, whether_plan):
    use_task_list = [task_list[j]]
    env.unwrapped.update_task_list(use_task_list)
    obs, info = env.reset()
    way_points = use_task_list[0]["way_point"]
    k = len(way_points)
    goal_reached = False
    for i in range(k):
        if goal_reached:
            break
        for yaw in [0, np.pi/2, np.pi, -np.pi/2]:
            env.unwrapped.controlled_vehicle.reset_equilibrium(way_points[i][0], way_points[i][1], yaw)
            waypoint = np.array([way_points[i][0], way_points[i][1], yaw, yaw, yaw, yaw], dtype=np.float32)
            if not planner.check_waypoint_legal(env, waypoint):
                print("illegal waypoint")
            else:
                print("legal waypoint")
                if whether_plan:
                    result_dict = planner.find_astar_trajectory_two_phases(env, obs["achieved_goal"], obs["desired_goal"], waypoint, info["obstacles_info"], info["map_vertices"], planner_config)
                    if result_dict.get("goal_reached", False):
                        planner.visualize_planner_final_result(obs["achieved_goal"], obs["desired_goal"], info["obstacles_info"], info["map_vertices"], result_dict)
                        goal_reached = True
                        break
    if goal_reached:
        return (True, None)
    return (False, task_list[j])

def main():
    with open("configs/envs/tt_planning_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    env = gym.make("tt-planning-v0", config=config)
    planner_config = {
        "plot_final_path": False,
        "plot_rs_path": False,
        "plot_expand_tree": False,
        "mp_step": 10,
        "N_steps": 10,
        "range_steer_set": 20,
        "max_iter": 10,
        "heuristic_type": "mix",
        "save_final_plot": False,
        "controlled_vehicle_config": {
            "w": 2.0,
            "wb": 3.5,
            "wd": 1.4,
            "rf": 4.5,
            "rb": 1.0,
            "tr": 0.5,
            "tw": 1.0,
            "rtr": 2.0,
            "rtf": 1.0,
            "rtb": 3.0,
            "rtr2": 2.0,
            "rtf2": 1.0,
            "rtb2": 3.0,
            "rtr3": 2.0,
            "rtf3": 1.0,
            "rtb3": 3.0,
            "max_steer": 0.6,
            "v_max": 2.0,
            "safe_d": 0.0,
            "safe_metric": 3.0,
            "xi_max": (np.pi) / 4,
        },
        "acceptance_error": 0.5,
    }
    
    with open("datasets/task_list_way_point_new.pkl", "rb") as f:
        task_list = pickle.load(f)
    
    whether_plan = True

    results = Parallel(n_jobs=-1)(delayed(process_task)(j, task_list, env, planner_config, whether_plan) for j in range(len(task_list)))
    
    successful_cases = sum(1 for result in results if result[0])
    failed_cases = [result[1] for result in results if not result[0] and result[1] is not None]

    print(f"Number of successful cases: {successful_cases}")
    print(f"Number of failed cases: {len(failed_cases)}")

    with open("datasets/failed_task_list.pkl", "wb") as f:
        pickle.dump(failed_cases, f)

if __name__ == "__main__":
    main()
