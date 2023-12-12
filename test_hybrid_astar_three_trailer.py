import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../../TTsystems_and_PINN/")

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
    
def main():
    
    
    parser = cfg()
    args = parser.parse_args()
    
    # input = np.array([-8, -8, np.deg2rad(-90.0)])
    # goal = np.array([0.0, 0.0, 0.0])
    
    # random_input = np.array([-10.0, 0.0, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)]) 
    
    
    # input = np.array([10, 10, 90, 90, 90, 90])
    # goal = np.array([0, 0, 0, 0, 0, 0])
    
    # single_tractor_trailer_planner = alg_no_obs.SingleTractorHybridAstarPlanner(ox, oy)
    # path, control_list, rs_path = single_tractor_trailer_planner.plan(input, goal, get_control_sequence=True)
    # single_tractor_trailer_planner.visualize_planning(input, goal, path, gif=True)
    # try:
    #     print("path step:", len(path.x))
    #     single_tractor_planner.visualize_planning(input, goal, path, gif=True)
    # except:
    #     print("No path find")
    
    ## the experiment test the reward for rs_path guide
    # input = np.array([8.0, 19.0, np.deg2rad(0.0)])
    # goal = np.array([29.0, 6.0, np.deg2rad(-90.0)])
    
    # input = np.array([23.0, 29.0, np.deg2rad(-90.0)])
    # goal = np.array([38.0, 6.0, np.deg2rad(-90.0)])
    
    # input = np.array([3.0, 10.0, np.deg2rad(90.0)])
    # goal = np.array([25.0, 15.0, np.deg2rad(90.0)])
    # input = np.array([3.0, 10.0, np.deg2rad(90.0)])
    # goal = np.array([25.0, 29.0, np.deg2rad(90.0)])
    
    # input = np.array([3, 15, np.deg2rad(90.0)])
    # goal = np.array([35, 35, np.deg2rad(90.0)])
    # map_env = [(0, 0), (60, 0), (0, 60), (60, 60)]
    # Map = tt_envs.MapBound(map_env)
    # Obstacle1 = tt_envs.QuadrilateralObstacle([(10, 0), (13, 0), (13, 30), (10, 30)])
    # Obstacle2 = tt_envs.QuadrilateralObstacle([(30, 10), (32.5, 60), (32.5, 10), (30, 60)])
    
    
    
    # input = np.array([10, 15, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
    # goal = np.array([115, 110, np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0), np.deg2rad(90.0)])
   
    input = np.array([10, 15, np.deg2rad(90.0), np.deg2rad(90.0)])
    goal = np.array([115, 110, np.deg2rad(90.0), np.deg2rad(90.0)])
    map_env = [(0, 0), (120, 0), (0, 120), (120, 120)]
    Map = tt_envs.MapBound(map_env)
    Obstacle1 = tt_envs.QuadrilateralObstacle([(40, 0), (43, 0), (43, 80), (40, 80)])
    Obstacle2 = tt_envs.QuadrilateralObstacle([(80, 40), (83, 40), (83, 120), (80, 120)])
    
    ox_map, oy_map = Map.sample_surface(0.1)
    ox1, oy1 = Obstacle1.sample_surface(0.1)
    ox2, oy2 = Obstacle2.sample_surface(0.1)
    
    ox = ox_map + ox1 + ox2
    oy = oy_map + oy1 + oy2
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    # ox, oy = obs.map_env2_high_resolution()
    # ox, oy = obs.remove_duplicates(ox, oy)
    # we still not implement single tractor with obs
    # three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy)
    # t1 = time.time()
    # path, control_list, rs_path = three_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=True)
    # t2 = time.time()
    # print("planning time:", t2 - t1)
    # three_trailer_planner.visualize_planning(input, goal, path, gif=True)
    one_trailer_planner = alg_obs.OneTractorTrailerHybridAstarPlanner(ox, oy)
    t1 = time.time()
    path, control_list, rs_path = one_trailer_planner.plan(input, goal, get_control_sequence=True, verbose=True)
    t2 = time.time()
    print("planning time:", t2 - t1)
    one_trailer_planner.visualize_planning(input, goal, path, gif=True)
    
    # print("the selected input is: ", random_input)
    
    # run_astar_algorithm_no_obstacle(random_input,"goal")
    
main()