import pickle
import yaml
import gymnasium as gym
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import tractor_trailer_envs as tt_envs
from tractor_trailer_envs import register_tt_envs
register_tt_envs()
from config import get_config
from utils import planner

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

    filtered_task_list = []
    
    for j, task in enumerate(task_list):
        use_task_list = [task]
        env.unwrapped.update_task_list(use_task_list)
        obs, info = env.reset()
        if planner.check_is_start_feasible(obs["achieved_goal"], info["obstacles_info"], info["map_vertices"], planner_config):
            filtered_task_list.append(task)
        else:
            print(f"Task {j} is not feasible and will be removed.")

    with open("datasets/task_list_way_point_new1.pkl", "wb") as f:
        pickle.dump(filtered_task_list, f)

    print(f"Original task list length: {len(task_list)}")
    print(f"Filtered task list length: {len(filtered_task_list)}")

if __name__ == "__main__":
    main()
