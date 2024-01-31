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

def define_map1():
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    
    map_env = [(0, 40), (40, 0), (40, 40), (0, 0)]
    Map = tt_envs.MapBound(map_env)
    
    ox_map, oy_map = Map.sample_surface(0.1)
    Obstacle1 = tt_envs.QuadrilateralObstacle([(0, 25), (0, 40), (20, 25), (20, 40)])
    Obstacle2 = tt_envs.QuadrilateralObstacle([(0, 12), (20, 12), (20, 0), (0, 0)])
    Obstacle3 = tt_envs.QuadrilateralObstacle([(26, 0), (27, 0), (27, 12), (26, 12)])
    Obstacle4 = tt_envs.QuadrilateralObstacle([(32, 0), (33, 0), (32, 12), (33, 12)])
    ox1, oy1 = Obstacle1.sample_surface(0.1)
    ox2, oy2 = Obstacle2.sample_surface(0.1)
    ox3, oy3 = Obstacle3.sample_surface(0.1)
    ox4, oy4 = Obstacle4.sample_surface(0.1)
    ox = ox_map + ox1 + ox2 + ox3 + ox4
    oy = oy_map + oy1 + oy2 + oy3 + oy4
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    return ox, oy


def define_map2():
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    
    map_env = [(29, 0), (0, 36), (29, 36), (0, 0)]
    Map = tt_envs.MapBound(map_env)
    
    ox_map, oy_map = Map.sample_surface(0.1)
    Obstacle1 = tt_envs.QuadrilateralObstacle([(6, 36), (17, 36), (6, 30), (17, 30)])
    Obstacle2 = tt_envs.QuadrilateralObstacle([(19, 36), (21, 36), (19, 30), (21, 30)])
    Obstacle3 = tt_envs.QuadrilateralObstacle([(27, 36), (29, 36), (27, 24), (29, 24)])
    Obstacle4 = tt_envs.QuadrilateralObstacle([(6, 18), (8, 18), (6, 24), (8, 24)])
    Obstacle5 = tt_envs.QuadrilateralObstacle([(11, 18), (21, 18), (11, 24), (21, 24)])
    Obstacle6 = tt_envs.QuadrilateralObstacle([(6, 12), (17, 12), (6, 18), (17, 18)])
    Obstacle7 = tt_envs.QuadrilateralObstacle([(19, 12), (21, 12), (19, 18), (21, 18)])
    Obstacle8 = tt_envs.QuadrilateralObstacle([(27, 0), (29, 0), (27, 18), (29, 18)])
    Obstacle9 = tt_envs.QuadrilateralObstacle([(6, 0), (11, 0), (6, 6), (11, 6)])
    Obstacle10 = tt_envs.QuadrilateralObstacle([(13, 0), (21, 0), (13, 6), (21, 6)])
    ox1, oy1 = Obstacle1.sample_surface(0.1)
    ox2, oy2 = Obstacle2.sample_surface(0.1)
    ox3, oy3 = Obstacle3.sample_surface(0.1)
    ox4, oy4 = Obstacle4.sample_surface(0.1)
    ox5, oy5 = Obstacle5.sample_surface(0.1)
    ox6, oy6 = Obstacle6.sample_surface(0.1)
    ox7, oy7 = Obstacle7.sample_surface(0.1)
    ox8, oy8 = Obstacle8.sample_surface(0.1)
    ox9, oy9 = Obstacle9.sample_surface(0.1)
    ox10, oy10 = Obstacle10.sample_surface(0.1)
    ox = ox_map + ox1 + ox2 + ox3 + ox4 + ox5 + ox6 + ox7 + ox8 + ox9 + ox10
    oy = oy_map + oy1 + oy2 + oy3 + oy4 + oy5 + oy6 + oy7 + oy8 + oy9 + oy10
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    return ox, oy
    
    
    
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
    # need to be the same
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
        "max_steer": 0.4, #[rad] maximum steering angle
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
    
    
def forward_simulation_two_trailer(input, goal, control_list, simulation_freq):
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
    
    
    controlled_vehicle = tt_envs.TwoTrailer(config_dict)
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
    
def forward_simulation_single_tractor(input, goal, control_list, simulation_freq):
    
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
    controlled_vehicle = tt_envs.SingleTractor(config_dict)
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
    if distance_error < 0.2:
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
       "heuristic_type": 'rl',
    }
    one_trailer_planner = alg_obs.OneTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = one_trailer_planner.plan_version2(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_one_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list

def generate_using_hybrid_astar_one_trailer_modify(input, goal, x_scale=3, y_scale=3):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    input_x, input_y = input[0], input[1]
    goal_x, goal_y = goal[0], goal[1]
    map_x_width = x_scale * np.abs(input_x - goal_x)
    map_y_width = y_scale * np.abs(input_y - goal_y)
    max_width = int(max(map_x_width, map_y_width))
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
       "heuristic_type": "traditional",
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


def generate_using_hybrid_astar_three_trailer(input, goal):
   
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
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = three_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_three_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list


def generate_using_hybrid_astar_three_trailer_map1(test_case=1):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    
    if test_case == 1:
        # obs_version case1
        input = np.array([8.0, 19.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([29.0, 6.0, np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0)]) 
    elif test_case == 2:
        # obs_version case2
        input = np.array([23.0, 29.0, np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0)])
        goal = np.array([38.0, 6.0, np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0)])
    elif test_case == 3:
        # obs_version case3
        input = np.array([36.0, 8.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([23.0, 6.0, np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0)])
    elif test_case == 4:
        # obs_version case4
        input = np.array([8.0, 15.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([32.0, 23.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    else:
        # obs_version case5
        input = np.array([8.0, 19.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([30.0, 34.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
    
    
    ox, oy = define_map1()
    config = {
       "plot_final_path": True,
       "plot_rs_path": True,
       "plot_expand_tree": True,
       "mp_step": 10,
       "range_steer_set": 20,
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
                "max_steer": 0.4, #[rad] maximum steering angle
                "v_max": 2.0, #[m/s] maximum velocity 
                "safe_d": 0.0, #[m] the safe distance from the vehicle to obstacle 
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
       "acceptance_error": 0.2,
       "heuristic_type": "rl",
    }
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=config["controlled_vehicle_config"]["v_max"], max_steer=config["controlled_vehicle_config"]["max_steer"])
    transition_list = forward_simulation_three_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list


def generate_using_hybrid_astar_one_trailer_map1(test_case=1):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    
    if test_case == 1:
        # obs_version case1
        input = np.array([8.0, 19.0, np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([29.0, 6.0, np.deg2rad(-90.0), np.deg2rad(-90.0)]) 
    elif test_case == 2:
        # obs_version case2
        input = np.array([23.0, 29.0, np.deg2rad(-90.0), np.deg2rad(-90.0)])
        goal = np.array([38.0, 6.0, np.deg2rad(-90.0), np.deg2rad(-90.0)])
    elif test_case == 3:
        # obs_version case3
        input = np.array([36.0, 8.0, np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([23.0, 6.0, np.deg2rad(-90.0), np.deg2rad(-90.0)])
    elif test_case == 4:
        # obs_version case4
        input = np.array([8.0, 15.0, np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([32.0, 23.0, np.deg2rad(0.0), np.deg2rad(0.0)])
    else:
        # obs_version case5
        input = np.array([8.0, 19.0, np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([30.0, 34.0, np.deg2rad(90.0), np.deg2rad(90.0)])
    
    
    ox, oy = define_map1()
    config = {
       "plot_final_path": True,
       "plot_rs_path": True,
       "plot_expand_tree": True,
       "mp_step": 10,
       "range_steer_set": 20,
       "heuristic_type": "rl",
    }
    one_trailer_planner = alg_obs.OneTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = one_trailer_planner.plan_version2(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_one_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list


def generate_using_hybrid_astar_three_trailer_map2(test_case=1):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    if test_case == 1:
        # obs_version case7
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([19.0, 26.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    elif test_case == 2:
        # obs_version case8
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([25.0, 29.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
    elif test_case == 3:
        # obs_version case9
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([25.0, 15.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
    elif test_case == 4:
        # obs_version case10
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([24.0, 11.0, np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0)])
    elif test_case == 5:
        # obs_version case11
        input = np.array([2.0, 8.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([19.0, 10.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    
    ox, oy = define_map2()
    config = {
       "plot_final_path": True,
       "plot_rs_path": True,
       "plot_expand_tree": True,
       "mp_step": 10,
       "range_steer_set": 20,
    #    "heuristic_type": "rl",
    }
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = three_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_three_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list


def generate_using_hybrid_astar_one_trailer_map2(test_case=1):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    if test_case == 1:
        # obs_version case7
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([19.0, 26.0, np.deg2rad(0.0), np.deg2rad(0.0)])
    elif test_case == 2:
        # obs_version case8
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([25.0, 29.0, np.deg2rad(90.0), np.deg2rad(90.0)])
    elif test_case == 3:
        # obs_version case9
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([25.0, 15.0, np.deg2rad(90.0), np.deg2rad(90.0)])
    elif test_case == 4:
        # obs_version case10
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([24.0, 11.0, np.deg2rad(-90.0), np.deg2rad(-90.0)])
    elif test_case == 5:
        # obs_version case11
        input = np.array([2.0, 8.0, np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([19.0, 8.0, np.deg2rad(0.0), np.deg2rad(0.0)])
    
    ox, oy = define_map2()
    config = {
       "plot_final_path": True,
       "plot_rs_path": True,
       "plot_expand_tree": True,
       "mp_step": 4,
       "range_steer_set": 20,
       "heuristic_type": "rl",
    }
    one_trailer_planner = alg_obs.OneTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = one_trailer_planner.plan_version2(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_one_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list

def generate_using_hybrid_astar_two_trailer(input, goal):
   
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
    two_trailer_planner = alg_obs.TwoTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = two_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_two_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list


def generate_using_hybrid_astar_single_tractor(input, goal):
   
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
    single_tractor_planner = alg_obs.SingleTractorHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = single_tractor_planner.plan(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = planners.action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_single_tractor(input, goal, control_recover_list, simulation_freq=10)
    return transition_list     
    

def random_generate_goal_one_trailer():
    x_coordinates = random.uniform(-10, 10)
    y_coordinates = random.uniform(-10, 10)
    yaw_state = random.uniform(-np.pi, np.pi)
    return np.array([x_coordinates, y_coordinates, yaw_state, yaw_state])

def main_process(index):
    goal = random_generate_goal_one_trailer()
    input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    transition_list = generate_using_hybrid_astar_one_trailer(input, goal)
    pack_transition_list = pack_transition(transition_list)
    return index, goal, pack_transition_list

def query_hybrid_astar_one_trailer(input, goal):
    transition_list = generate_using_hybrid_astar_one_trailer(input, goal)
    pack_transition_list = pack_transition(transition_list)
    return goal, pack_transition_list


def test_single():
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    # goal = np.array([3,-8,np.deg2rad(-170.0),np.deg2rad(-150.0)])
    # transition_list = generate_using_hybrid_astar_one_trailer_modify(input, goal)
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    # goal = np.array([10, -10, np.deg2rad(80.0),np.deg2rad(80.0), np.deg2rad(80.0),np.deg2rad(80.0)])
    # transition_list = generate_using_hybrid_astar_three_trailer(input, goal)
    
    
    input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    goal = np.array([10, -10, np.deg2rad(160.0),np.deg2rad(160.0), np.deg2rad(160.0)])
    transition_list = generate_using_hybrid_astar_two_trailer(input, goal)
    
    
    
    # transition_list = generate_using_hybrid_astar_one_trailer(goal)
    return transition_list

def test_single_new():
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    # goal = np.array([3,-8,np.deg2rad(-170.0),np.deg2rad(-150.0)])
    # transition_list = generate_using_hybrid_astar_one_trailer_modify(input, goal)
    input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    goal = np.array([20, 20, np.deg2rad(-150.0), np.deg2rad(-150.0)])
    transition_list = generate_using_hybrid_astar_one_trailer(input, goal)
    # transition_list = generate_using_hybrid_astar_single_tractor(input, goal)
    # transition_list = generate_using_hybrid_astar_one_trailer(goal)
    return transition_list

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

def parallel_execution(num_processes=4, total_runs=100):
    with Pool(num_processes) as pool:
        results = pool.map(main_process, range(total_runs))
    save_results(results, batch_size=100)
    
def define_large_map():
    map_size = 120
    tunel_width = 10
    map_env = [(0, map_size), (map_size, 0), (map_size, map_size), (0, 0)]
    Map = tt_envs.MapBound(map_env)

    ox, oy = Map.sample_surface(0.1)

    # 随机生成20个障碍物
    obstacles = []
    

    obstacles = [
        tt_envs.QuadrilateralObstacle([(tunel_width, tunel_width), (tunel_width, map_size - tunel_width), (map_size / 2, tunel_width), (map_size / 2 , map_size - tunel_width)]),
        tt_envs.QuadrilateralObstacle([(map_size / 2 + tunel_width, map_size - tunel_width), (map_size / 2 + tunel_width, tunel_width), (map_size - tunel_width, tunel_width), (map_size - tunel_width, map_size - tunel_width)]),
    ]

    # 将所有障碍物的表面点加入到坐标列表中
    for obstacle in obstacles:
        ox_obs, oy_obs = obstacle.sample_surface(0.1)
        ox += ox_obs
        oy += oy_obs

    ox, oy = tt_envs.remove_duplicates(ox, oy)
    return ox, oy

def define_large_map_with_20_obstacles():
    map_size = 400
    map_env = [(0, map_size), (map_size, 0), (map_size, map_size), (0, 0)]
    Map = tt_envs.MapBound(map_env)

    ox, oy = Map.sample_surface(0.1)

    # 随机生成20个障碍物
    obstacles = []
    for _ in range(20):
        # 随机确定障碍物的左下角和右上角
        x1, y1 = random.randint(0, map_size - 50), random.randint(0, map_size - 50)
        x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(10, 50)

        # 确保障碍物在地图内
        x2, y2 = min(x2, map_size), min(y2, map_size)

        obstacle = tt_envs.QuadrilateralObstacle([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
        obstacles.append(obstacle)

    # 将所有障碍物的表面点加入到坐标列表中
    for obstacle in obstacles:
        ox_obs, oy_obs = obstacle.sample_surface(0.1)
        ox += ox_obs
        oy += oy_obs

    ox, oy = tt_envs.remove_duplicates(ox, oy)
    return ox, oy

def generate_using_hybrid_astar_three_trailer_tunel_map(test_case=1):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    if test_case == 1:
        # obs_version case7
        input = np.array([58.0, 5.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([89.0, 115.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    elif test_case == 2:
        # obs_version case8
        input = np.array([58.0, 5.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([50.0, 115.0, np.deg2rad(180.0), np.deg2rad(180.0), np.deg2rad(180.0), np.deg2rad(180.0)])
    elif test_case == 3:
        # obs_version case9
        input = np.array([58.0, 5.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
        goal = np.array([89.0, 115.0, np.deg2rad(180.0), np.deg2rad(180.0), np.deg2rad(180.0), np.deg2rad(180.0)])
    elif test_case == 4:
        # obs_version case10
        input = np.array([3.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([24.0, 11.0, np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0), np.deg2rad(-90.0)])
    elif test_case == 5:
        # obs_version case11
        input = np.array([2.0, 8.0, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
        goal = np.array([19.0, 10.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    
    ox, oy = define_large_map()
    config = {
       "plot_final_path": True,
       "plot_rs_path": True,
       "plot_expand_tree": True,
       "mp_step": 10,
       "range_steer_set": 20,
    #    "heuristic_type": "rl",
    }
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6)
    transition_list = forward_simulation_three_trailer(input, goal, control_recover_list, simulation_freq=10)
    return transition_list


    
if __name__ == "__main__":
    
    # ox, oy = define_large_map()
    # planners.plot_map(ox, oy)
    # plt.savefig("large_map.png")
    # transition_list = generate_using_hybrid_astar_three_trailer_tunel_map(test_case=3)
    
    
    transition_list = generate_using_hybrid_astar_three_trailer_map1(test_case=3)
    # print("done")
    # pack_transition_list = pack_transition(transition_list)