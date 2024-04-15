import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import numpy as np
import rl_agents as agents
import pickle
import tractor_trailer_envs as tt_envs
import time
import planner_zoo.hybrid_astar_planner.hybrid_astar_obs_version as alg_obs

def find_astar_trajectory(input, goal, obstacles_info):
    config = {
        "plot_final_path": False,
        "plot_rs_path": False,
        "plot_expand_tree": False,
        "mp_step": 10, # Important
        "N_steps": 10, # Important
        "range_steer_set": 20,
        "max_iter": 50,
        "heuristic_type": "mix",
        "save_final_plot": True,
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
                    "safe_metric": 3.0, # the safe distance from the vehicle to obstacle
                    "xi_max": (np.pi) / 4, # jack-knife constraint  
                },
        "acceptance_error": 0.5,
    }
    generate_using_hybrid_astar_three_trailer(input, goal, obstacles_info, config)

def generate_using_hybrid_astar_three_trailer(input, goal, obstacles_info=None, config=None):
   
    # input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
    edge = 50
    map_env = [(-edge, -edge), (edge, -edge), (-edge, edge), (edge, edge)]
    Map = tt_envs.MapBound(map_env)
    
    ox_map, oy_map = Map.sample_surface(0.1)
    ox = ox_map
    oy = oy_map
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    if obstacles_info is not None:
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
    path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=True, obstacles_info=obstacles_info)
    t2 = time.time()
    print("planning time:", t2 - t1)
    # except: 
    #     return None
    return None
 

def main():
    with open("datasets/reaching_results_with_obstacles.pickle", "rb") as f:
        datasets = pickle.load(f)
    for data in datasets[:100]:
        input = data["task"]['achieved_goal']
        goal = data["task"]["desired_goal"]
        obstacles_info = data["obstacles_info"]
        find_astar_trajectory(input, goal, obstacles_info)
    # obstacles_info_new = [[(obstacles_info[0][0][0], obstacles_info[0][0][1] - 5), (obstacles_info[0][1][0], obstacles_info[0][1][1] + 5) , \
    #     (obstacles_info[0][2][0], obstacles_info[0][2][1] + 5) , (obstacles_info[0][3][0], obstacles_info[0][3][1] - 5)]]
    # find_astar_trajectory(input, goal, obstacles_info_new)
    
if __name__ == "__main__":
    main()
    
        
    