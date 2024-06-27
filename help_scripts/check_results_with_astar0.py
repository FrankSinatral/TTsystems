import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import numpy as np
import rl_agents as agents
import pickle
import tractor_trailer_envs as tt_envs
import time
import planner_zoo.hybrid_astar_planner.hybrid_astar_obs_version as alg_obs
import yaml
import gymnasium as gym
from tractor_trailer_envs import register_tt_envs
register_tt_envs()
from utils.planner import find_astar_trajectory
from joblib import Parallel, delayed
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

def find_astar_trajectory(input, goal, obstacles_info):
    config = {
        "plot_final_path": False,
        "plot_rs_path": False,
        "plot_expand_tree": False,
        "mp_step": 10, # Important
        "N_steps": 10, # Important
        "range_steer_set": 20,
        "max_iter": 50,
        "heuristic_type": "traditional",
        "save_final_plot": False,
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
    is_success = generate_using_hybrid_astar_three_trailer(input, goal, obstacles_info, config)
    return is_success
    
def test_forward_simulation_three_trailer(input, goal, ox, oy, control_list, simulation_freq):
    """note that this is only a validation function"""
    # Now we test the collision
    
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
    controlled_vehicle = tt_envs.ThreeTrailer(config_dict)
    controlled_vehicle.reset(*input)
    
    
    for action_clipped in control_list:
        controlled_vehicle.step(action_clipped, 1 / simulation_freq)
        if controlled_vehicle.is_collision(ox, oy):
            return False
    final_state = np.array(controlled_vehicle.state)
    distance_error = mixed_norm(goal, final_state)
    # distance_error = np.linalg.norm(goal - final_state)
    if distance_error < 0.5:
        return True
    else:
        return False

def generate_using_hybrid_astar_three_trailer(input, goal, obstacles_info=None, config=None):
   
    
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
    # t1 = time.time()
    path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=False, obstacles_info=obstacles_info)
    # t2 = time.time()
    # print("planning time:", t2 - t1)
    # except: 
    #     return None
    if control_list is not None:
        control_recover_list = action_recover_from_planner(control_list, simulation_freq=10, v_max=config["controlled_vehicle_config"]["v_max"], max_steer=config["controlled_vehicle_config"]["max_steer"])
        is_success = test_forward_simulation_three_trailer(input, goal, ox, oy, control_recover_list, simulation_freq=10)
    else:
        return False
    return is_success
 

def main():
    # run astar0(plain astar planner) to see the final test results
    # you can use both the cases given in the datasets or random selected by the env
    use_datasets = True
    # with open("datasets/reaching_results_with_obstacles.pickle", "rb") as f:
    #     datasets = pickle.load(f)  
    with open("datasets/goal_with_obstacles_info_list.pickle", "rb") as f:
        datasets = pickle.load(f)
    with open("configs/agents/eval/planner0_env.yaml", "r") as f:
        config = yaml.safe_load(f)
    env = gym.make("tt-meta-reaching-v0", config=config)
    failed_cases = []
    failed_number = 0
    if use_datasets:
        goal_with_obstacles_info_list = []
        for data in datasets[:100]:
            goal = data["goal"]
            obstacles_info = data["obstacles_info"]
            task_dict = {
                "goal": goal,
                "obstacles_info": obstacles_info
            }
            goal_with_obstacles_info_list.append(task_dict) 
    task_list = []
    for i in range(100):
        if use_datasets:
            now_goal_with_obstacles_info_list = [goal_with_obstacles_info_list[i]]
            env.unwrapped.update_goal_with_obstacles_info_list(now_goal_with_obstacles_info_list)
        o, info = env.reset(seed=i)
        # env.unwrapped.real_render()
        input = o["achieved_goal"]
        goal = o["desired_goal"]
        obstacles_info = info["obstacles_info"]
        task_list.append((input, goal, obstacles_info))
        map_veritices = info["map_vertices"]
        a_config = {
            "plot_final_path": False,
            "plot_rs_path": False,
            "plot_expand_tree": False,
            "mp_step": 10, # Important
            "N_steps": 10, # Important
            "range_steer_set": 20,
            "max_iter": 50,
            "heuristic_type": "traditional",
            "save_final_plot": False,
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
        find_astar_trajectory(input, goal, obstacles_info, a_config, map_veritices)
        # is_success = find_astar_trajectory(input, goal, obstacles_info)
        # print(f"Episode {i}: Success - {is_success}")
        # if not is_success:
        #     failed_cases.append((o, info))
        #     failed_number += 1
    
    astar_results = Parallel(n_jobs=-1)(delayed(find_astar_trajectory)(input, goal, obstacles_info) for input, goal, obstacles_info in task_list)
    print("failed number: ", astar_results.count(False))
    # print("failed number: ", failed_number)
    # Save all failed cases to a single file
    # with open('datasets/all_failed_cases_planner0.pkl', 'wb') as f:
    #     pickle.dump(failed_cases, f)
    
if __name__ == "__main__":
    main()