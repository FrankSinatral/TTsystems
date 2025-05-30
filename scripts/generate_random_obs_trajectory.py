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
from joblib import Parallel, delayed

def plot_configuration(start, goal, ox, oy, path):
    plt.plot(ox, oy, 'sk', markersize=1)
    plt.axis("equal")
    ax = plt.gca() 
    config = {
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
    vehicle = tt_envs.ThreeTrailer(config)
    # change here last plot goal and start
    vehicle.reset(*goal)
    
    vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
    vehicle.reset(*start)
    vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
    if path is not None:
        plt.plot(path.x, path.y, 'blue')
    plt.savefig("reconstruct.png")
    plt.close()


def is_rectangle_intersect(rect1, rect2):
    # 检验两个长方形rect1和rect2是否相交
    # 这里简化处理，假设长方形平行于坐标轴
    # rect1和rect2格式：[(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    x1_min, y1_min = min(rect1[0][0], rect1[3][0]), min(rect1[0][1], rect1[1][1])
    x1_max, y1_max = max(rect1[1][0], rect1[2][0]), max(rect1[2][1], rect1[3][1])
    x2_min, y2_min = min(rect2[0][0], rect2[3][0]), min(rect2[0][1], rect2[1][1])
    x2_max, y2_max = max(rect2[1][0], rect2[2][0]), max(rect2[2][1], rect2[3][1])
    
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


def action_recover_from_planner(control_list, simulation_freq, v_max, max_steer):
    # this shift is for rl api
    new_control_list = []
    for control in control_list:
        new_control = np.array([control[0] * simulation_freq / v_max, control[1] / max_steer])
        new_control_list.append(new_control)
    
    return new_control_list
    
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


def save_results(results):
    save_dir = './planner_result/datas_for_random_map'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    existing_files = os.listdir(save_dir)
    file_index = 0
    while f'result_{file_index}.pkl' in existing_files:
        file_index += 1
    with open(os.path.join(save_dir, f'result_{file_index}.pkl'), 'wb') as f:
        pickle.dump(results, f)

def define_random_map_with_2_obstacles(angle_type=0):
    # outside frame
    map_env = [(-30, -30), (-30, 30), (30, -30), (30, 30)]
    existing_rectangles = []
    # existing rectangle
    existing_rectangle = [(-7, -1), (-7, 1), (4.5, 1), (4.5, -1)]
    existing_rectangles.append(existing_rectangle)
    
    Map = tt_envs.MapBound(map_env)
    if angle_type % 2 == 0:
        # pingfang
        new_fixed_rectangle = generate_fixed_size_random_rectangle(map_env, existing_rectangles)
    else:
        # shufang
        new_fixed_rectangle = generate_fixed_size_random_rectangle(map_env, existing_rectangles, length_x=2, length_y=11.5)
    results = get_equilbrium_configuration(new_fixed_rectangle)
    
    existing_rectangles.append(new_fixed_rectangle)

    ox, oy = Map.sample_surface(0.1)
    new_rectangles = add_two_non_intersecting_rectangles(existing_rectangles, map_env)

    # 随机生成2个障碍物
    obstacles = []
    obstacles_info = []
    for rectangle in new_rectangles:
        if rectangle in existing_rectangles:
            continue
        obstacle = tt_envs.QuadrilateralObstacle(rectangle)
        obstacles_info.append(rectangle)
        obstacles.append(obstacle)

    # 将所有障碍物的表面点加入到坐标列表中
    for obstacle in obstacles:
        ox_obs, oy_obs = obstacle.sample_surface(0.1)
        ox += ox_obs
        oy += oy_obs

    ox, oy = tt_envs.remove_duplicates(ox, oy)
    return ox, oy, results, obstacles_info


def is_rectangle_intersect(rect1, rect2):
    # check whether two rectangles intersect with each other
    # Assume that the edge is parallel to the axis
    # format：[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # can take out the asumptions
    x1_min, y1_min = min(pt[0] for pt in rect1), min(pt[1] for pt in rect1)
    x1_max, y1_max = max(pt[0] for pt in rect1), max(pt[1] for pt in rect1)
    x2_min, y2_min = min(pt[0] for pt in rect2), min(pt[1] for pt in rect2)
    x2_max, y2_max = max(pt[0] for pt in rect2), max(pt[1] for pt in rect2)
    
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def generate_fixed_size_random_rectangle(frame, existing_rectangles, length_x=11.5, length_y=2):
    """
    generate a fixed size rectange, make sure it is inside the given frame and does not intersect with existing rectangles
    frame format: [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    existing_rectangles format: [[(x1, y1), (x2, y2), (x3, y3), (x4, y4)], ...]
    length_x: the length of the rectangle along the x axis
    length_y: the length of the rectangle along the y axis
    """
    while True:  # find until finding a non-intersecting rectangle
        min_x, max_x = frame[0][0] + length_x / 2, frame[2][0] - length_x / 2
        min_y, max_y = frame[0][1] + length_y / 2, frame[1][1] - length_y / 2

        center_x = random.uniform(min_x, max_x)
        center_y = random.uniform(min_y, max_y)

        new_rect = [
            (center_x - length_x / 2, center_y - length_y / 2),
            (center_x - length_x / 2, center_y + length_y / 2),
            (center_x + length_x / 2, center_y + length_y / 2),
            (center_x + length_x / 2, center_y - length_y / 2)
        ]

        # check whether the new rectangle intersects with existing rectangles
        if all(not is_rectangle_intersect(new_rect, rect) for rect in existing_rectangles):
            return new_rect  # if not intersecting, return the new rectangle

def generate_random_rectangle(frame):
    """random generate a rectangle, make sure it is inside the given frame and the two pairs of opposite edges are not equal
    frame format: [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    return: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    x_min, x_max = frame[0][0], frame[2][0]
    y_min, y_max = frame[0][1], frame[1][1]
    
    # Ensure that the generated rectangle has two pairs of opposite edges that are not equal
    while True:
        x1, x2 = sorted([random.uniform(x_min, x_max) for _ in range(2)])
        if abs(x1 - x2) > 2:
            break

    while True:
        y1, y2 = sorted([random.uniform(y_min, y_max) for _ in range(2)])
        if abs(y1 - y2) > 2:
            break
    
    return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]


def add_two_non_intersecting_rectangles(existing_rects, frame):
    rectangles = existing_rects[:]
    for _ in range(2):
        while True:
            new_rect = generate_random_rectangle(frame)
            if all(not is_rectangle_intersect(new_rect, rect) for rect in rectangles):
                rectangles.append(new_rect)
                break
    return rectangles


def get_equilbrium_configuration(rect):
    # return the corresponding configuration of the 3-tt vehicle
    # contains two configuration of the tractor trailer vehicle
    results = []
    x_min, x_max = min(pt[0] for pt in rect), max(pt[0] for pt in rect)
    y_min, y_max = min(pt[1] for pt in rect), max(pt[1] for pt in rect)
    length_x = x_max - x_min
    length_y = y_max - y_min
    if length_x == 2:
        # shufang
        state_x = (x_max + x_min) / 2
        state_y = y_max - 4.5
        state_yaw = (np.pi) / 2
        # 90
        results.append(np.array([state_x, state_y, state_yaw, state_yaw, state_yaw, state_yaw]))
        state_y = y_min + 4.5
        state_yaw = -(np.pi) / 2
        results.append(np.array([state_x, state_y, state_yaw, state_yaw, state_yaw, state_yaw]))
    else:  
        state_y = (y_max + y_min) / 2
        state_x = x_max - 4.5
        state_yaw = 0
        # 0
        results.append(np.array([state_x, state_y, state_yaw, state_yaw, state_yaw, state_yaw]))
        state_x = x_min + 4.5
        state_yaw = -np.pi
        results.append(np.array([state_x, state_y, state_yaw, state_yaw, state_yaw, state_yaw]))
    return results   

def forward_simulation_three_trailer(input, goal, control_list, simulation_freq, vehicle_config):
    # Pack every 10 steps to add to buffer
    # need to be the same
    transition_list = []
    
    
    controlled_vehicle = tt_envs.ThreeTrailer(vehicle_config)
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
    if distance_error <= 0.2:
        print("Accept")
        return transition_list
    else:
        print("Reject", goal[2])
        return None


def generate_using_hybrid_astar_three_trailer_random_2_obstacles_map(angle_type=0):
    # Random Map Test
    # angle_type: 0 -- 0
    #             1 -- 90
    #             2 -- -180
    #             3 -- -90
    # Fank: make sure that the goal result
    input = np.array([0.0, 0.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)])
    ox, oy, goal_results, obstacles_info = define_random_map_with_2_obstacles(angle_type)
    goal = goal_results[angle_type//2]
    config = {
       "plot_final_path": False,
       "plot_rs_path": False,
       "plot_expand_tree": False,
       "plot_failed_path": False,
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
       "acceptance_error": 0.2,
       "max_iter": 1000,
       "heuristic_type": "traditional",
    }
    # Fill up the things that will have
    result_dict = {"transition_list": None,
                   "original_control_list": None,
                   "difficulty": None,
                   "obstacles_info": obstacles_info,
                   "planning_config": config,
                   "start": input,
                   "goal": goal,
                   "path": None,
                   }
    # Running planner and get results
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    # planning api
    try:
        t1 = time.time()
        path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=True)
        t2 = time.time()
        print("planning time:", t2 - t1)
        result_dict["path"] = path
        result_dict["original_control_list"] = control_list
    except:
        print("failed finding path")
    
    if result_dict["path"] is not None:
        control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=config["controlled_vehicle_config"]["v_max"], max_steer=config["controlled_vehicle_config"]["max_steer"])
        transition_list = forward_simulation_three_trailer(input, goal, control_recover_list, simulation_freq=10, vehicle_config=config["controlled_vehicle_config"])
        result_dict["transition_list"] = transition_list
        if transition_list is not None:
            result_dict["difficulty"] = len(control_list)
        else:
            result_dict["difficulty"] = np.inf
    else:
        result_dict["difficulty"] = np.inf
    
    
    return result_dict

def run_simulation(angle_type):
    # This is a placeholder for your actual function.
    # Replace it with the call to your specific function and return its result.
    result_dict = generate_using_hybrid_astar_three_trailer_random_2_obstacles_map(angle_type=angle_type)
    return result_dict 

# def save_results_to_pickle(results, filename):
#     with open(filename, 'wb') as f:
#         pickle.dump(results, f)    

def load_and_visualize(file_path, i=None):
    # Fank: load from the result and visualize
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    
    # vehicle_config = {
    #             "w": 2.0, #[m] width of vehicle
    #             "wb": 3.5, #[m] wheel base: rear to front steer
    #             "wd": 1.4, #[m] distance between left-right wheels (0.7 * W)
    #             "rf": 4.5, #[m] distance from rear to vehicle front end
    #             "rb": 1.0, #[m] distance from rear to vehicle back end
    #             "tr": 0.5, #[m] tyre radius
    #             "tw": 1.0, #[m] tyre width
    #             "rtr": 2.0, #[m] rear to trailer wheel
    #             "rtf": 1.0, #[m] distance from rear to trailer front end
    #             "rtb": 3.0, #[m] distance from rear to trailer back end
    #             "rtr2": 2.0, #[m] rear to second trailer wheel
    #             "rtf2": 1.0, #[m] distance from rear to second trailer front end
    #             "rtb2": 3.0, #[m] distance from rear to second trailer back end
    #             "rtr3": 2.0, #[m] rear to third trailer wheel
    #             "rtf3": 1.0, #[m] distance from rear to third trailer front end
    #             "rtb3": 3.0, #[m] distance from rear to third trailer back end   
    #             "max_steer": 0.6, #[rad] maximum steering angle
    #             "v_max": 2.0, #[m/s] maximum velocity 
    #             "safe_d": 0.0, #[m] the safe distance from the vehicle to obstacle 
    #             "xi_max": (np.pi) / 4, # jack-knife constraint  
    #         }
        
    
    for result_dict in results:
        start = result_dict["start"]
        goal = result_dict["goal"]
        obstacles_info = result_dict["obstacles_info"]
        transition_list = result_dict["transition_list"]
        path = result_dict["path"]
        # control_recover_list = action_recover_from_planner(result_dict["original_control_list"], simulation_freq=10, v_max=2, max_steer=0.6)
        # transition_list = forward_simulation_three_trailer(start, goal, control_recover_list, simulation_freq=10, vehicle_config=vehicle_config)
        map_bound = [(-30, -30), (-30, 30), (30, -30), (30, 30)]
        Map = tt_envs.MapBound(map_bound)
        obstacles = []
        ox, oy = Map.sample_surface(0.1)
        for obstacle_info in obstacles_info:
            obstacle = tt_envs.QuadrilateralObstacle(obstacle_info)
            obstacles.append(obstacle)
        for obstacle in obstacles:
            ox_obs, oy_obs = obstacle.sample_surface(0.1)
            ox += ox_obs
            oy += oy_obs
        # if transition_list is None:
        #     path = None
        plot_configuration(start, goal, ox, oy, path)

def extract_obstacles_info(obstacle_info):
    # Extract the coordinates of the bottom left and top right corners for each obstacle
    corners_info = [[min(point[0] for point in obstacle), max(point[0] for point in obstacle), 
                     min(point[1] for point in obstacle), max(point[1] for point in obstacle)] 
                    for obstacle in obstacle_info]
    return np.array(corners_info).flatten()

def load_results(file_path):
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    return results

def count_none_and_inf(file_path):
    results = load_results(file_path)
    path_none_count = 0
    transition_none_count = 0
    difficulty_inf_count = 0

    for result in results:
        if result['path'] is None:
            path_none_count += 1
        if result['transition_list'] is None:
            transition_none_count += 1
        if result['difficulty'] == float('inf'):
            difficulty_inf_count += 1

    return path_none_count, transition_none_count, difficulty_inf_count

def generate_ox_oy_with_map_bound(obstacles_info):
    map_bound = [(-30, -30), (-30, 30), (30, -30), (30, 30)]
    Map = tt_envs.MapBound(map_bound)
    ox, oy = Map.sample_surface(0.1)
    try:
        for rectangle in obstacles_info:
            obstacle = tt_envs.QuadrilateralObstacle(rectangle)
            ox_obs, oy_obs = obstacle.sample_surface(0.1)
            ox += ox_obs
            oy += oy_obs
    except:
        pass

    ox, oy = tt_envs.remove_duplicates(ox, oy)
    return ox, oy



def save_abnormal_results(file_path):
    # save abnormal results from previous planning results
    results = load_results(file_path)
    abnormal_results = []
    for result in results:
        if result['path'] is not None and result['transition_list'] is None:
            abnormal_results.append(result)

    # Save the abnormal results to a pickle file
    save_dir = './planner_result/datas_for_random_map'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    existing_files = os.listdir(save_dir)
    file_index = 0
    while f'abnormal_result_{file_index}.pkl' in existing_files:
        file_index += 1
    with open(os.path.join(save_dir, f'abnormal_result_{file_index}.pkl'), 'wb') as f:
        pickle.dump(abnormal_results, f)
        
def save_failed_results(file_path):
    # save failed results from previous planning results
    results = load_results(file_path)
    failed_results = []
    for result in results:
        if result['path'] is None:
            failed_results.append(result)

    # Save the failed results to a pickle file
    save_dir = './planner_result/datas_for_random_map'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    existing_files = os.listdir(save_dir)
    file_index = 0
    while f'failed_result_{file_index}.pkl' in existing_files:
        file_index += 1
    with open(os.path.join(save_dir, f'failed_result_{file_index}.pkl'), 'wb') as f:
        pickle.dump(failed_results, f)
        
def analyze_abnormal_results(file_path):
    results = load_results(file_path)
    first_result = results[0]
    obstacles_info = first_result['obstacles_info']
    start = first_result['start']
    goal = first_result['goal']
    path = first_result['path']
    goal[0] = goal[0]
    ox, oy = generate_ox_oy_with_map_bound(None)
    plot_configuration(start, goal, ox, oy, path)
    config = {
       "plot_final_path": True,
       "plot_rs_path": True,
       "plot_expand_tree": True,
       "plot_failed_path": True,
       "mp_step": 6,
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
       "cost_configuration":
                {
                    "scissors_cost": 10.0,
                    "gear_cost": 10.0,
                    "backward_cost": 1.0,
                    "steer_change_cost": 5.0,
                    "h_cost": 1.0,
                    "steer_angle_cost": 1.0,
                }, 
       "acceptance_error": 0.2,
       "max_iter": 1000,
       "heuristic_type": "traditional",
    }
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    path, control_list, rs_path = three_trailer_planner.plan_new_version(start, goal, get_control_sequence=True, verbose=True)
       
def main():
    # # Number of times to run the simulation for each angle_type
    # num_runs = 1000
    # # Number of jobs to run in parallel. Adjust based on your system's capabilities.
    # num_jobs = -1  # Use all available CPUs

    # # Using joblib to run simulations in parallel for each angle_type
    # results = Parallel(n_jobs=num_jobs)(delayed(run_simulation)(angle_type=i) for i in range(4) for _ in range(num_runs))
    # save_results(results)
    # # for _ in range(1000):
    # #     run_simulation(angle_type=3)
    # # print(1)
    file_path = 'planner_result/datas_for_random_map/result_1.pkl'
    # analyze_abnormal_results(file_path)
    # analyze_abnormal_results(file_path)
    load_and_visualize(file_path)
    # path_none_count, transition_none_count, difficulty_inf_count = count_none_and_inf('planner_result/datas_for_random_map/result_1.pkl')
    # print(f"Number of None Paths: {path_none_count}")
    # print(f"Number of None Transitions: {transition_none_count}")
    # print(f"Number of Infinite Difficulty: {difficulty_inf_count}")
    
    
    # # Use the function
    # file_path = 'planner_result/datas_for_random_map/result_1.pkl'
    # save_failed_results(file_path)
if __name__ == "__main__":
    main()
    
    
    
    