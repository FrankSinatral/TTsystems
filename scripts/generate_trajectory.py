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
import multiprocessing
import logging
import random
from joblib import Parallel, delayed


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
    
    map_env = [(-50, -50), (50, -50), (-50, 50), (50, 50)]
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
       "plot_final_path": False,
       "plot_rs_path": False,
       "plot_expand_tree": False,
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
                "max_steer": 0.6, #[rad] maximum steering angle
                "v_max": 2.0, #[m/s] maximum velocity 
                "safe_d": 0.0, #[m] the safe distance from the vehicle to obstacle 
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
       "acceptance_error": 0.5,
       "max_iter": 20,
    }
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=True)
    if control_list is not None:
        print("The action length:", len(control_list))
        control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=config["controlled_vehicle_config"]["max_steer"])
        transition_list = forward_simulation_three_trailer(input, goal, control_recover_list, simulation_freq=10)
    else:
        return None
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
    input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    goal = np.array([10, -10, np.deg2rad(80.0),np.deg2rad(80.0), np.deg2rad(80.0),np.deg2rad(80.0)])
    transition_list = generate_using_hybrid_astar_three_trailer(input, goal)
    
    
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    # goal = np.array([10, -10, np.deg2rad(160.0),np.deg2rad(160.0), np.deg2rad(160.0)])
    # transition_list = generate_using_hybrid_astar_two_trailer(input, goal)
    
    
    
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

# def parallel_execution(num_processes=4, total_runs=100):
#     with Pool(num_processes) as pool:
#         results = pool.map(main_process, range(total_runs))
#     save_results(results, batch_size=100)
    

def process_item(failed_o):
    input = failed_o['observation']
    goal = failed_o['desired_goal']
    transition_list = generate_using_hybrid_astar_three_trailer(input, goal)
    return transition_list  # 返回每个条目的 transition_list


def process_failed_o(failed_o):
    input = failed_o['observation']
    goal = failed_o['desired_goal']
    transition_list = generate_using_hybrid_astar_three_trailer(input, goal)
    return transition_list is None
    
if __name__ == "__main__":
    
    
    # original version
    # start_time = time.time()
    # with open("rl_training/failed/file.pickle", 'rb') as f:
    #     failed_o_list = pickle.load(f)
    # print("total demo:", len(failed_o_list))
    # failed_planning = 0
    # for failed_o in failed_o_list:
    #     t1 = time.time()
    #     input = failed_o['observation']
    #     goal = failed_o['desired_goal']
    #     transition_list = generate_using_hybrid_astar_three_trailer(input, goal)
    #     if transition_list is None:
    #         failed_planning += 1
    #     t2 = time.time()
    #     print("Planner Finish:", t2 - t1)
    # end_time = time.time()
    # print("total time cost:", end_time - start_time)
    # print("total failed demo:", failed_planning)
    
    
    # joblib version
    start_time = time.time()

    with open("rl_training/failed/file.pickle", 'rb') as f:
        failed_o_list = pickle.load(f)
    print("total demo:", len(failed_o_list))

    # 使用 joblib 并行处理
    results = Parallel(n_jobs=-1)(delayed(process_item)(failed_o) for failed_o in failed_o_list)

    # 组合所有的 transition_list
    all_transitions = []
    for result in results:
        if result is not None:
            all_transitions.extend(result)

    end_time = time.time()
    print("total time cost:", end_time - start_time)
    print("total failed demo:", len([res for res in results if res is None]))
    print("Total transitions:", len(all_transitions))
    print(1)

    
    # # multiprocessing version
    # start_time = time.time()

    # with open("rl_training/failed/file.pickle", 'rb') as f:
    #     failed_o_list = pickle.load(f)
    # print("total demo:", len(failed_o_list))

    # # 创建一个进程池并并行处理列表
    # with multiprocessing.Pool() as pool:
    #     results = pool.map(process_failed_o, failed_o_list)

    # # 计算失败的规划数量
    # failed_planning = sum(results)

    # end_time = time.time()
    # print("total time cost:", end_time - start_time)
    # print("total failed demo:", failed_planning)