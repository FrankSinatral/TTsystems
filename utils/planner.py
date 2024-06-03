import numpy as np
import planner_zoo.hybrid_astar_planner.hybrid_astar_obs_version as alg_obs
import utils.common as common
import tractor_trailer_envs as tt_envs
from tractor_trailer_envs import register_tt_envs
register_tt_envs()

def test_forward_simulation_three_trailer(input, goal, ox, oy, control_list, simulation_freq):
    """This is a function that use tt_envs for forward validation
    Given a input configuration, we use the control_list to simulate the vehicle
    We check whether there is a collision with the obstacles
    Also calculate the distance to the goal
    """
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
    state_list = [np.array(controlled_vehicle.state).astype(np.float32)]
    
    for action_clipped in control_list:
        controlled_vehicle.step(action_clipped, 1 / simulation_freq)
        state_list.append(np.array(controlled_vehicle.state).astype(np.float32))
        if controlled_vehicle.is_collision(ox, oy):
            result_dict = {
                "state_list": state_list,
                "collision": True,
                "goal_reached": False,
            }
            return result_dict
    final_state = np.array(controlled_vehicle.state)
    distance_error = common.mixed_norm(goal, final_state)
    # distance_error = np.linalg.norm(goal - final_state)
    if distance_error < 0.5:
        result_dict = {
            "state_list": state_list,
            "collision": False,
            "goal_reached": True,
        }
        return result_dict
    else:
        result_dict = {
            "state_list": state_list,
            "collision": False,
            "goal_reached": False,
        }
        return result_dict

def find_astar_trajectory(input, goal, obstacles_info, map_veritices, config):
    """
    This is a common api for utilizing the astar planner to find the trajectory
    map_vertices: given as a list of the four vertices
    """
    assert config is not None, "config should not be None"
    Map = tt_envs.MapBound(map_veritices)
    ox_map, oy_map = Map.sample_surface(0.1)
    ox = ox_map
    oy = oy_map
    ox, oy = tt_envs.remove_duplicates(ox, oy)
    if (obstacles_info is not None) and len(obstacles_info) > 0:
        for rectangle in obstacles_info:
            obstacle = tt_envs.QuadrilateralObstacle(rectangle)
            ox_obs, oy_obs = obstacle.sample_surface(0.1)
            ox += ox_obs
            oy += oy_obs
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    path, control_list, rs_path = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=False, obstacles_info=obstacles_info)
    if control_list is None:
        result_dict = {
            "state_list": [input],
            "control_list": [],
            "collision": False,
            "goal_reached": False,
        }
    else:
        control_recover_list = common.action_recover_from_planner(control_list, simulation_freq=10, v_max=config["controlled_vehicle_config"]["v_max"], max_steer=config["controlled_vehicle_config"]["max_steer"])
        result_dict = test_forward_simulation_three_trailer(input, goal, ox, oy, control_recover_list, simulation_freq=10)
        result_dict["control_list"] = control_recover_list
    return result_dict