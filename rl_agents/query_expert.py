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
    # Pack every 10 steps to add to buffer
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
    
def forward_simulation_three_trailer(input, goal, control_list, simulation_freq):
    # Pack every 10 steps to add to buffer
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
    
    
    controlled_vehicle = tt_envs.ThreeTrailer(config_dict)
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
    
def pack_transition(transition_list):
    # for rl training
    pack_transition_list = []
    i = 0
    while i < len(transition_list):
        state, action, _ = transition_list[i]
        next_state_index = min(i + 9, len(transition_list) - 1)
        _, _, next_state = transition_list[next_state_index]
        pack_transition_list.append([state, action, next_state])
        i += 10
    return pack_transition_list

def pack_transition_with_reward(goal, transition_list, obstacles_info=None):
    # for rl training
    # add reward every 10 steps
    
    pack_transition_list = []
    i = 0
    while i < len(transition_list):
        state, action, _ = transition_list[i]
        next_state_index = min(i + 9, len(transition_list) - 1)
        _, _, next_state = transition_list[next_state_index]
        if i == len(transition_list) - 10:
            #pack mamually with reward
            if obstacles_info is not None:
                pack_transition_list.append([np.concatenate([state, state, goal, obstacles_info]), action, np.concatenate([next_state, next_state, goal, obstacles_info]), 15, True])
            else:
                pack_transition_list.append([np.concatenate([state, state, goal]), action, np.concatenate([next_state, next_state, goal]), 15, True])
        else:
            if obstacles_info is not None:
                pack_transition_list.append([np.concatenate([state, state, goal, obstacles_info]), action, np.concatenate([next_state, next_state, goal, obstacles_info]), 15, True])
            else:
                pack_transition_list.append([np.concatenate([state, state, goal]), action, np.concatenate([next_state, next_state, goal]), -1 , False])
        i += 10
    return pack_transition_list



def generate_using_hybrid_astar_one_trailer(input, goal):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    map_env = [(-30, -30), (30, -30), (-30, 30), (30, 30)]
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
        t1 = time.time()
        path, control_list, rs_path = one_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=True)
        t2 = time.time()
        print("planning time:", t2 - t1)
    except: 
        return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_one_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list

def generate_using_hybrid_astar_one_trailer_version1(input, goal, x_scale=3, y_scale=3):
    input = input[:4]
    goal = goal[:4]
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    input_x, input_y = input[0], input[1]
    goal_x, goal_y = goal[0], goal[1]
    map_x_width = x_scale * np.abs(input_x - goal_x)
    map_y_width = y_scale * np.abs(input_y - goal_y)
    max_width = int(max(map_x_width, map_y_width))
    # Fank: mp_step only have to be a even number
    if (max_width + 1) % 2 == 0:
        mp_step = max_width + 1
    else:
        mp_step = max_width + 2
    
    
    map_env = [(input_x - map_x_width, input_y - map_y_width), (input_x - map_x_width, input_y + map_y_width), \
               (input_x + map_x_width, input_y - map_y_width), (input_x + map_x_width, input_y + map_y_width)]
    Map = tt_envs.MapBound(map_env)
    
    ox_map, oy_map = Map.sample_surface(0.1)
    ox = ox_map
    oy = oy_map
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    config = {
       "plot_final_path": False,
       "plot_rs_path": False,
       "plot_expand_tree": False,
       "mp_step": mp_step,
       "range_steer_set": 20,
    }
    one_trailer_planner = alg_obs.OneTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    try:
        # t1 = time.time()
        path, control_list, rs_path = one_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=False)
        # t2 = time.time()
        # print("planning time:", t2 - t1)
    except: 
        return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_one_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list

def generate_using_hybrid_astar_one_trailer_version2(input, goal):
    input = input[:4]
    goal = goal[:4]
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    map_env = [(-30, -30), (30, -30), (-30, 30), (30, 30)]
    Map = tt_envs.MapBound(map_env)
    
    ox_map, oy_map = Map.sample_surface(0.1)
    ox = ox_map
    oy = oy_map
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    config = {
       "plot_final_path": False,
       "plot_rs_path": False,
       "plot_expand_tree": False,
       "mp_step": 10,
       "range_steer_set": 20,
    }
    one_trailer_planner = alg_obs.OneTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    try:
        # t1 = time.time()
        path, control_list, rs_path = one_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=False)
        # t2 = time.time()
        # print("planning time:", t2 - t1)
    except: 
        return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_one_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list

def restore_obstacles_info(flattened_info):
    # Reshape the flattened array to have 4 columns (for x_min, x_max, y_min, y_max)
    reshaped_info = flattened_info.reshape(-1, 4)
    
    # Restore the original obstacle_info format
    obstacle_info = [[(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)] 
                     for x_min, x_max, y_min, y_max in reshaped_info]
    return obstacle_info

def generate_using_hybrid_astar_three_trailer(input, goal, obstacles_info=None):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    
    map_env = [(-30, -30), (30, -30), (-30, 30), (30, 30)]
    Map = tt_envs.MapBound(map_env)
    
    ox_map, oy_map = Map.sample_surface(0.1)
    ox = ox_map
    oy = oy_map
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    if obstacles_info is not None:
        obstacles_info = restore_obstacles_info(obstacles_info)
        if obstacles_info is not None:
            for rectangle in obstacles_info:
                obstacle = tt_envs.QuadrilateralObstacle(rectangle)
                ox_obs, oy_obs = obstacle.sample_surface(0.1)
                ox += ox_obs
                oy += oy_obs
            
    config = {
       "plot_final_path": False,
       "plot_rs_path": False,
       "plot_expand_tree": False,
       "mp_step": 10,
       "range_steer_set": 20,
       "max_iter": 50,
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
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
       "acceptance_error": 0.5,
    }
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    try:
        t1 = time.time()
        path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=True)
        t2 = time.time()
        print("planning time:", t2 - t1)
    except: 
        return None
    if control_list is not None:
        control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=config["controlled_vehicle_config"]["v_max"], max_steer=config["controlled_vehicle_config"]["max_steer"])
        transition_list = forward_simulation_three_trailer(input, goal, control_recover_list, simulation_freq=10)
    else:
        return None
    return transition_list 

def random_generate_goal_one_trailer():
    x_coordinates = random.uniform(-10, 10)
    y_coordinates = random.uniform(-10, 10)
    yaw_state = random.uniform(-np.pi, np.pi)
    return np.array([x_coordinates, y_coordinates, yaw_state, yaw_state])


def query_hybrid_astar_one_trailer(input, goal):
    # fixed to 6-dim
    transition_list = generate_using_hybrid_astar_one_trailer_version1(input, goal)
    if transition_list is None:
        transition_list = generate_using_hybrid_astar_one_trailer_version2(input, goal)
    pack_transition_list = pack_transition_with_reward(goal, transition_list)
    return pack_transition_list

def query_hybrid_astar_three_trailer(input, goal, obstacles_info=None):
    # fixed to 6-dim
    transition_list = generate_using_hybrid_astar_three_trailer(input, goal, obstacles_info)
    if transition_list is not None:
        pack_transition_list = pack_transition_with_reward(goal, transition_list, obstacles_info)
    else:
        return None
    return pack_transition_list
