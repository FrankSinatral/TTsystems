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

def find_expert_trajectory_meta(o, map, mp_step=10, N_steps=10, max_iter=50, heuristic_type="traditional", observation_type="original", whether_obstacles_info=False):
    """only for 3-tt"""
    goal = o[0]["desired_goal"]
    input = o[0]["observation"]
    obstacles_info = o[1]
    if whether_obstacles_info:
        obstacles_properties = o[2]
    # if len(obstacles_info) == 0:
    #     obstacles_info = None  
    config = {
        "plot_final_path": False,
        "plot_rs_path": False,
        "plot_expand_tree": False,
        "mp_step": mp_step,
        "N_steps": N_steps, # Important
        "range_steer_set": 20,
        "max_iter": max_iter,
        "heuristic_type": heuristic_type,
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
            "safe_metric": 3.0, #[m] the safe distance from the vehicle to obstacle
            "xi_max": (np.pi) / 4, # jack-knife constraint  
        },
        "acceptance_error": 0.5,
        "observation": "original",
    }
    
    pack_transition_list = query_hybrid_astar_three_trailer_meta(input, goal, map, obstacles_info, config, observation_type)
    if whether_obstacles_info:
        if pack_transition_list is None:
            pass
        else:
            for transition in pack_transition_list:
                transition.append(obstacles_properties)
        
    return pack_transition_list

def find_expert_trajectory(o, vehicle_type):
    goal = o["desired_goal"]
    input = o["observation"]
    try:
        obstacles_info = o["obstacles_info"]
    except:
        obstacles_info = None
        
    config = {
        "plot_final_path": False,
        "plot_rs_path": False,
        "plot_expand_tree": False,
        "mp_step": 12, # Important
        "N_steps": 30, # Important
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
    if vehicle_type == "one_trailer":
        pack_transition_list = query_hybrid_astar_one_trailer(input, goal)
    elif vehicle_type == "three_trailer":
        pack_transition_list = query_hybrid_astar_three_trailer(input, goal, obstacles_info, config)
    return pack_transition_list

def query_hybrid_astar_three_trailer_meta(input, goal, map, obstacles_info, config=None, observation_type="original"):
    """this function first generate the transition list using hybrid astar, then pack the transition list with reward"""
    # fixed to 6-dim
    transition_list = generate_using_hybrid_astar_three_trailer_meta(input, goal, map, obstacles_info, config, observation_type)
    if transition_list is not None:
        pack_transition_list = pack_transition_with_reward_meta(goal, transition_list, N_steps=config["N_steps"], observation_type=observation_type)
    else:
        return None
    return pack_transition_list

def action_recover_from_planner(control_list, simulation_freq, v_max, max_steer):
    # this shift is for rl api
    new_control_list = []
    for control in control_list:
        new_control = np.array([control[0] * simulation_freq / v_max, control[1] / max_steer])
        new_control_list.append(new_control)
    
    return new_control_list

def cyclic_angle_distance(angle1, angle2):
    return min(abs(angle1 - angle2), 2 * np.pi - abs(angle1 - angle2))

def mixed_norm(goal, final_state):
    # the input is 6-dim np_array
    # calculate position sum of square
    position_diff_square = np.sum((goal[:2] - final_state[:2]) ** 2)
    
    # calculate angle distance
    angle_diff_square = sum([cyclic_angle_distance(goal[i], final_state[i]) ** 2 for i in range(2, 6)])
    
    # combine the two distances
    total_distance = np.sqrt(position_diff_square + angle_diff_square)
    return total_distance


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
    distance_error = mixed_norm(goal, final_state)
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
        "safe_metric": 3.0, #[m] the safe distance from the vehicle to obstacle
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
    distance_error = mixed_norm(goal, final_state)
    # distance_error = np.linalg.norm(goal - final_state)
    if distance_error < 0.5:
        print("Accept")
        return transition_list
    else:
        print("Reject")
        return None
  
def forward_simulation_three_trailer_meta(input, goal, ox, oy, control_list, simulation_freq, observation_type="original"):
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
        "safe_metric": 3.0, #[m] the safe distance from the vehicle to obstacle
        "xi_max": (np.pi) / 4, # jack-knife constraint  
    }
    transition_list = []
    controlled_vehicle = tt_envs.ThreeTrailer(config_dict)
    controlled_vehicle.reset(*input)
    state = controlled_vehicle.observe()
    if observation_type == "lidar_detection":
        state_lidar_detection = controlled_vehicle.lidar_detection(ox, oy)
    elif observation_type == "one_hot_representation":
        # TODO: to align with the env
        state_one_hot_representation = controlled_vehicle.one_hot_representation(d=5, number=8, ox=ox, oy=oy)
    elif observation_type == "lidar_detection_one_hot":
        state_lidar_detection_one_hot = controlled_vehicle.lidar_detection_one_hot(5, ox, oy)
    elif observation_type == "one_hot_representation_enhanced":
        # TODO: to align with the env
        state_one_hot_representation_enhanced = controlled_vehicle.one_hot_representation_enhanced(d=10, number=8, ox=ox, oy=oy)
    
    for action_clipped in control_list:
        controlled_vehicle.step(action_clipped, 1 / simulation_freq)
        # Fank: add collision detection
        if controlled_vehicle.is_collision(ox, oy):
            return None
        next_state = controlled_vehicle.observe()
        if observation_type == "original":
            transition = [state, action_clipped, next_state]
        elif observation_type == "lidar_detection":
            next_state_lidar_detection = controlled_vehicle.lidar_detection(ox, oy)
            transition = [state, state_lidar_detection, action_clipped, next_state, next_state_lidar_detection]
            state_lidar_detection = next_state_lidar_detection
        elif observation_type == "lidar_detection_one_hot":
            next_state_lidar_detection_one_hot = controlled_vehicle.lidar_detection_one_hot(5, ox, oy)
            transition = [state, state_lidar_detection_one_hot, action_clipped, next_state, next_state_lidar_detection_one_hot]
            state_lidar_detection_one_hot = next_state_lidar_detection_one_hot
        elif observation_type == "one_hot_representation":
            # TODO: to align with the env
            next_state_one_hot_representation = controlled_vehicle.one_hot_representation(d=5, number=8, ox=ox, oy=oy)
            transition = [state, state_one_hot_representation, action_clipped, next_state, next_state_one_hot_representation]
            state_one_hot_representation = next_state_one_hot_representation
        elif observation_type == "one_hot_representation_enhanced":
            # TODO: to align with the env
            next_state_one_hot_representation_enhanced = controlled_vehicle.one_hot_representation_enhanced(d=10, number=8, ox=ox, oy=oy)
            transition = [state, state_one_hot_representation_enhanced, action_clipped, next_state, next_state_one_hot_representation_enhanced]
            state_one_hot_representation_enhanced = next_state_one_hot_representation_enhanced
        
        transition_list.append(transition)
        state = next_state
        
    final_state = np.array(controlled_vehicle.state)
    distance_error = mixed_norm(goal, final_state)
    # distance_error = np.linalg.norm(goal - final_state)
    if distance_error < 0.5:
        print("Accept")
        return transition_list
    else:
        print("Reject")
        return None   

    
def pack_transition(transition_list):
    # Currently not used
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

def pack_transition_with_reward(goal, transition_list, obstacles_info=None, N_steps=10):
    # for rl training
    # add reward every 10 steps
    assert len(transition_list) % N_steps == 0, "The length of transition list should be a multiple of N_steps"
    pack_transition_list = []
    i = 0
    while i < len(transition_list):
        state, action, _ = transition_list[i]
        next_state_index = min(i + N_steps - 1, len(transition_list) - 1)
        _, _, next_state = transition_list[next_state_index]
        if i == len(transition_list) - N_steps:
            #pack mamually with reward
            if obstacles_info is not None:
                pack_transition_list.append([np.concatenate([state, state, goal, obstacles_info]), action, np.concatenate([next_state, next_state, goal, obstacles_info]), 15, True])
            else:
                pack_transition_list.append([np.concatenate([state, state, goal]), action, np.concatenate([next_state, next_state, goal]), 15, True])
        else:
            if obstacles_info is not None:
                pack_transition_list.append([np.concatenate([state, state, goal, obstacles_info]), action, np.concatenate([next_state, next_state, goal, obstacles_info]), -1, False]) # fix a bug
            else:
                pack_transition_list.append([np.concatenate([state, state, goal]), action, np.concatenate([next_state, next_state, goal]), -1 , False])
        i += N_steps
    return pack_transition_list

def pack_transition_with_reward_meta(goal, transition_list, N_steps=10, observation_type="original"):
    # for rl training
    # add reward every 10 steps
    # here I have change the api for more diverse observation type
    # different observation type relabel the reward differently
    assert len(transition_list) % N_steps == 0, "The length of transition list should be a multiple of N_steps"
    pack_transition_list = []
    i = 0
    while i < len(transition_list):
        if observation_type == "original":
            """only (s, a, s')"""
            state, action, _ = transition_list[i]
            next_state_index = min(i + N_steps - 1, len(transition_list) - 1)
            _, _, next_state = transition_list[next_state_index]
            if i == len(transition_list) - N_steps:
                #pack mamually with reward
                pack_transition_list.append([np.concatenate([state, state, goal]), action, np.concatenate([next_state, next_state, goal]), 15, True])
            else:
                pack_transition_list.append([np.concatenate([state, state, goal]), action, np.concatenate([next_state, next_state, goal]), -1, False])
                
        elif observation_type == "lidar_detection":
            """(s, s_lidar, a, s', s'_lidar)
            s_lidar is a 32-dim vector
            """
            state, state_lidar_detection, action, _, _ = transition_list[i]
            next_state_index = min(i + N_steps - 1, len(transition_list) - 1)
            _, _, _, next_state, next_state_lidar_detection = transition_list[next_state_index]
            next_state_minLidar = np.min(next_state_lidar_detection)
            if i == len(transition_list) - N_steps:
                #pack mamually with reward
                # TODO: to align with the env
                if next_state_minLidar <= 5:
                    pack_transition_list.append([np.concatenate([state, state, goal, state_lidar_detection]), action, np.concatenate([next_state, next_state, goal, next_state_lidar_detection]), 15 + (-20) * (1 - next_state_minLidar / 5), True])
                else:
                    pack_transition_list.append([np.concatenate([state, state, goal, state_lidar_detection]), action, np.concatenate([next_state, next_state, goal, next_state_lidar_detection]), 15, True])
            else:
                next_state_minLidar = np.min(next_state_lidar_detection)
                if next_state_minLidar <= 5:
                    pack_transition_list.append([np.concatenate([state, state, goal, state_lidar_detection]), action, np.concatenate([next_state, next_state, goal, next_state_lidar_detection]), -1 + (-20) * (1 - next_state_minLidar / 5), False])
                else:
                    pack_transition_list.append([np.concatenate([state, state, goal, state_lidar_detection]), action, np.concatenate([next_state, next_state, goal, next_state_lidar_detection]), -1, False])
        # elif observation_type == "one_hot_representation":
        #     state, state_one_hot_representation, action, _, _ = transition_list[i]
        #     next_state_index = min(i + N_steps - 1, len(transition_list) - 1)
        #     _, _, _, next_state, next_state_one_hot_representation = transition_list[next_state_index]
        #     next_state_sumOneHot = np.sum(next_state_one_hot_representation)
        #     if i == len(transition_list) - N_steps:
        #         #pack mamually with reward
        #         # TODO: to align with the env
        #         if next_state_sumOneHot >= 1:
        #             pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation]), 15 + (-20) * (next_state_sumOneHot/20), True])
        #         else:
        #             pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation]), 15, True])
        #     else:
        #         if next_state_sumOneHot >= 1:
        #             pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation]), -1 + (-20) * (next_state_sumOneHot/20), False])
        #         else:
        #             pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation]), -1, False])
        elif observation_type == "one_hot_representation":
            state, state_one_hot_representation, action, _, _ = transition_list[i]
            next_state_index = min(i + N_steps - 1, len(transition_list) - 1)
            _, _, _, next_state, next_state_one_hot_representation = transition_list[next_state_index]
            next_state_sumOneHot = np.sum(next_state_one_hot_representation)
            if i == len(transition_list) - N_steps:
                #pack mamually with reward
                # TODO: to align with the env
                if next_state_sumOneHot >= 1:
                    pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation]), 15, True])
                else:
                    pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation]), 15, True])
            else:
                if next_state_sumOneHot >= 1:
                    pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation]), -1, False])
                else:
                    pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation]), -1, False])
        elif observation_type == "one_hot_representation_enhanced":
            state, state_one_hot_representation_enhanced, action, _, _ = transition_list[i]
            next_state_index = min(i + N_steps - 1, len(transition_list) - 1)
            _, _, _, next_state, next_state_one_hot_representation_enhanced = transition_list[next_state_index]
            if i == len(transition_list) - N_steps:
                #pack mamually with reward
                # TODO: to align with the env
                pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation_enhanced]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation_enhanced]), 15, True])
            else:
                pack_transition_list.append([np.concatenate([state, state, goal, state_one_hot_representation_enhanced]), action, np.concatenate([next_state, next_state, goal, next_state_one_hot_representation_enhanced]), -1, False])
        elif observation_type == "lidar_detection_one_hot":
            state, state_lidar_detection_one_hot, action, _, _ = transition_list[i]
            next_state_index = min(i + N_steps - 1, len(transition_list) - 1)
            _, _, _, next_state, next_state_lidar_detection_one_hot = transition_list[next_state_index]
            if i == len(transition_list) - N_steps:
                #pack mamually with reward
                # TODO: to align with the env
                pack_transition_list.append([np.concatenate([state, state, goal, state_lidar_detection_one_hot]), action, np.concatenate([next_state, next_state, goal, next_state_lidar_detection_one_hot]), 15, True])
            else:
                pack_transition_list.append([np.concatenate([state, state, goal, state_lidar_detection_one_hot]), action, np.concatenate([next_state, next_state, goal, next_state_lidar_detection_one_hot]), -1, False])
        i += N_steps
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

def generate_using_hybrid_astar_three_trailer(input, goal, obstacles_info=None, config=None):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    edge = 80
    map_env = [(-edge, -edge), (edge, -edge), (-edge, edge), (edge, edge)]
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
    
    assert config is not None, "config should not be None"
    
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # try:
    t1 = time.time()
    path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    if control_list is not None:
        control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=config["controlled_vehicle_config"]["v_max"], max_steer=config["controlled_vehicle_config"]["max_steer"])
        transition_list = forward_simulation_three_trailer_meta(input, goal, ox, oy, control_recover_list, simulation_freq=10)
    else:
        return None
    return transition_list 

def generate_using_hybrid_astar_three_trailer_meta(input, goal, map, obstacles_info, config=None, observation_type="original"):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    
    ox_map, oy_map = map.sample_surface(0.1)
    ox = ox_map
    oy = oy_map
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    try: # In case of []
        for rectangle in obstacles_info:
            obstacle = tt_envs.QuadrilateralObstacle(rectangle)
            ox_obs, oy_obs = obstacle.sample_surface(0.1)
            ox += ox_obs
            oy += oy_obs      
    except:
        pass
    
    assert config is not None, "config should not be None"
    
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    try:
        t1 = time.time()
        path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=True, obstacles_info=obstacles_info)
        t2 = time.time()
        print("planning time:", t2 - t1)
    except: 
        return None
    if control_list is not None:
        control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=config["controlled_vehicle_config"]["v_max"], max_steer=config["controlled_vehicle_config"]["max_steer"])
        transition_list = forward_simulation_three_trailer_meta(input, goal, ox, oy, control_recover_list, simulation_freq=10, observation_type=observation_type)
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

def query_hybrid_astar_three_trailer(input, goal, obstacles_info=None, config=None):
    # fixed to 6-dim
    transition_list = generate_using_hybrid_astar_three_trailer(input, goal, obstacles_info, config)
    if transition_list is not None:
        pack_transition_list = pack_transition_with_reward(goal, transition_list, obstacles_info, N_steps=config["N_steps"])
    else:
        return None
    return pack_transition_list

