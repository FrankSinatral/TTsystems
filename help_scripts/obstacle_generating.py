import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import tractor_trailer_envs as tt_envs


    

def obstacle_generator():  
    config = {
        "max_steer": 0.6,
        "rb": 1.0,
        "rf": 4.5,
        "rtb": 3.0,
        "rtb2": 3.0,
        "rtb3": 3.0,
        "rtf": 1.0,
        "rtf2": 1.0,
        "rtf3": 1.0,
        "rtr": 2.0,
        "rtr2": 2.0,
        "rtr3": 2.0,
        "safe_d": 0.0,
        "tr": 0.5,
        "tw": 1.0,
        "v_max": 2.0,
        "w": 2.0,
        "wb": 3.5,
        "wd": 1.4,
        "xi_max": 0.7853981633974483,
        "safe_metric": 3.0
    }
    controlled_vehicle = tt_envs.ThreeTrailer(config)
    with open("datasets/reaching_results.pickle", "rb") as f:
        datasets = pickle.load(f)
    goal_with_obstacles_info_list = []
    for data in datasets:
        start = data["task"]["achieved_goal"]
        goal = data["task"]["desired_goal"]
        distance = np.linalg.norm(start[:2] - goal[:2])
        if distance < 10.0:
            continue
        state_list = data["state_list"]
        path_length = len(state_list)
        half_path_length = path_length // 2
        half_state = state_list[half_path_length]
        center = half_state[:2]
        length = 2
        obstacle_info = [(center[0] - length/2, center[1] - length/2), (center[0] - length/2, center[1] + length/2), \
            (center[0] + length/2, center[1] + length/2), (center[0] + length/2, center[1] - length/2)]
        
        # check here to ensure that the start and goal will not collide with the obstacle
        obstacle = tt_envs.QuadrilateralObstacle(obstacle_info)
        ox, oy = obstacle.sample_surface(0.1)
        controlled_vehicle.reset(*start)
        collision_start = controlled_vehicle.is_collision(ox, oy)
        controlled_vehicle.reset(*goal)
        collision_goal = controlled_vehicle.is_collision(ox, oy)
        if collision_start or collision_goal:
            continue
                
        data["obstacles_info"] = [obstacle_info]
        single_dict = {
            "goal": goal,
            "obstacles_info": [obstacle_info],
        }
        goal_with_obstacles_info_list.append(single_dict)
        
    with open("datasets/goal_with_obstacles_info_list.pickle", "wb") as f:
        pickle.dump(goal_with_obstacles_info_list, f)
        
obstacle_generator()
    
    
    