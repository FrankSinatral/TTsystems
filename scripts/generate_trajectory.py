import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import planner_zoo as planners

import planner_zoo.hybrid_astar_planner.hybrid_astar_obs_version as alg_obs
from planner_zoo.hybrid_astar_planner.config import get_config as cfg
# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs
import matplotlib.pyplot as plt
import time
import pickle
from multiprocessing import Pool
import logging
import random


def action_recover_from_planner(control_list, simulation_freq, v_max, max_steer):
    # this shift is for rl api
    new_control_list = []
    for control in control_list:
        new_control = np.array([control[0] * simulation_freq / v_max, control[1] / max_steer])
        new_control_list.append(new_control)
    
    return new_control_list

def forward_simulation_one_trailer(input, goal, control_list, simulation_freq):
    config_dict = {
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
        "xi_max": (np.pi) / 4, # jack-knife constraint  
    }
    transition_list = []
    
    
    controlled_vehicle = tt_envs.OneTrailer(config_dict)
    controlled_vehicle.reset(*input)
    state = controlled_vehicle.observe()
    for action_clipped in control_list:
        controlled_vehicle.step(action_clipped, 1 / simulation_freq)
        next_state = controlled_vehicle.observe()
        transition = [state, action_clipped, next_state]
        transition_list.append(transition)
        state = next_state
    final_state = np.array(controlled_vehicle.state)
    distance_error = np.linalg.norm(goal - final_state)
    if distance_error < 0.5:
        print("Accept")
        return transition_list
    else:
        print("Reject")
        return None



def generate_using_hybrid_astar_one_trailer(goal):
   
    input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    
    map_env = [(-80, -80), (80, -80), (-80, 80), (80, 80)]
    Map = tt_envs.MapBound(map_env)
    
    ox_map, oy_map = Map.sample_surface(0.1)
    ox = ox_map
    oy = oy_map
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    config = {
       "plot_final_path": True,
       "plot_rs_path": True,
       "plot_expand_tree": True,
       "mp_step": 10,
       "range_steer_set": 20,
    }
    one_trailer_planner = alg_obs.OneTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    try:
        # t1 = time.time()
        path, control_list, rs_path = one_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=True)
        # t2 = time.time()
        # print("planning time:", t2 - t1)
        control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
        transition_list = forward_simulation_one_trailer(input, goal, control_recover_list, simulation_freq=10)
        return transition_list
    except: 
        return None
    

def random_generate_goal_one_trailer():
    x_coordinates = random.uniform(-10, 10)
    y_coordinates = random.uniform(-10, 10)
    yaw_state = random.uniform(-np.pi, np.pi)
    return np.array([x_coordinates, y_coordinates, yaw_state, yaw_state])

def main_process(index):
    goal = random_generate_goal_one_trailer()
    transition_list = generate_using_hybrid_astar_one_trailer(goal)
    return index, goal, transition_list

def test_single():
    goal = np.array([5,6,0.0,0.0])
    transition_list = generate_using_hybrid_astar_one_trailer(goal)
    print(1)

def save_results(results, batch_size=100):
    save_dir = './trajectory_buffer'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    existing_files = os.listdir(save_dir)
    file_index = 0
    while f'result_{file_index}.pkl' in existing_files:
        file_index += 1
        
    batch = []
    for index, goal, transitions_list in results:
        if transitions_list is not None:
            batch.append({"goal": goal, "transitions_list": transitions_list})
            
            if len(batch) == batch_size:
                with open(os.path.join(save_dir, f'result_{file_index}.pkl'), 'wb') as f:
                    pickle.dump(batch, f)    
                batch = []
                file_index += 1
                while f'result_{file_index}.pkl' in existing_files:
                    file_index += 1
    if batch:
        with open(os.path.join(save_dir, f'result_{file_index}.pkl'), 'wb') as f:
            pickle.dump(batch, f)        

def parallel_execution(num_processes=4, total_runs=1000):
    with Pool(num_processes) as pool:
        results = pool.map(main_process, range(total_runs))
    save_results(results, batch_size=100)
    
if __name__ == "__main__":
    # t1 = time.time()
    # parallel_execution(20)
    # t2 = time.time()
    # print("execution time:", t2 - t1)
    test_single()
    # with open('./trajectory_buffer/result_0.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print(1)