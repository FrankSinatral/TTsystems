import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../../ttsystems/")
# import VehicleModel.models as md
import tractor_trailer_envs as tt_envs
# import HybridAstarPlanner.planner_base.no_obs_version as alg_no_obs
# import HybridAstarPlanner.planner_base.obs_version as alg_obs
# import HybridAstarPlanner.planner_base.test_version as alg_test
# import rrt_planner.planner_base.obs_version as alg_rrt
import planner_zoo.rrt_planner.rrt_obs_version as alg_rrt
from rrt_planner.planner_base.config import create_parser as cfg
# import rl_training.tt_system_implement as tt_envs

import matplotlib.pyplot as plt
import time
import pickle
import multiprocessing
import logging
import random
    
def main():
    
    
    parser = cfg()
    args = parser.parse_args()
    
    ## the experiment test the reward for rs_path guide
    # input = np.array([10, 15, np.deg2rad(90.0)])
    # goal = np.array([45, 110, np.deg2rad(90.0)])
    # map_env = [(0, 0), (120, 0), (0, 120), (120, 120)]
    # Map = tt_envs.MapBound(map_env)
    # Obstacle1 = tt_envs.QuadrilateralObstacle([(40, 0), (43, 0), (43, 80), (40, 80)])
    # Obstacle2 = tt_envs.QuadrilateralObstacle([(80, 40), (83, 40), (83, 120), (80, 120)])
    input = np.array([3, 15, np.deg2rad(90.0)])
    goal = np.array([35, 35, np.deg2rad(90.0)])
    map_env = [(0, 0), (40, 0), (0, 40), (40, 40)]
    Map = tt_envs.MapBound(map_env)
    Obstacle1 = tt_envs.QuadrilateralObstacle([(10, 0), (13, 0), (13, 30), (10, 30)])
    Obstacle2 = tt_envs.QuadrilateralObstacle([(30, 10), (33, 40), (33, 10), (30, 40)])
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')
    Map.plot(ax, 0.1)
    Obstacle1.plot(ax, 0.1)
    Obstacle2.plot(ax, 0.1)
    ox_map, oy_map = Map.sample_surface(0.1)
    ox1, oy1 = Obstacle1.sample_surface(0.1)
    ox2, oy2 = Obstacle2.sample_surface(0.1)
    ox = ox_map + ox1 + ox2
    oy = oy_map + oy1 + oy2
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    # we still not implement single tractor with obs
    single_tractor_planner = alg_rrt.RRTPlanner(ox, oy)
    single_tractor_planner.seed(10)
    path = single_tractor_planner.plan(ax, input, goal)
    single_tractor_planner.visualize_planning(input, goal, path)
    
main()