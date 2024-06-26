import numpy as np
import planner_zoo.hybrid_astar_planner.hybrid_astar_obs_version as alg_obs
import utils.common as common
import tractor_trailer_envs as tt_envs
import matplotlib.pyplot as plt
import time
from tractor_trailer_envs import register_tt_envs
register_tt_envs()

def check_finetune_shape(input):
    x, y, yaw, yawt1, yawt2, yawt3 = input
    if abs(yaw - yawt1) <= np.pi/8 and abs(yawt1 - yawt2) <= np.pi/8 and abs(yawt2 - yawt3) <= np.pi/8:
        return True
    else:
        return False

def test_forward_simulation_three_trailer(input, goal, ox, oy, control_list, simulation_freq, perception_required=None):
    """This is a function that use tt_envs for forward validation
    Given a input configuration, we use the control_list to simulate the vehicle
    We check whether there is a collision with the obstacles
    Also calculate the distance to the goal
    perception_required: observation_type("original", "lidar_detection_one_hot", "lidar_detection_one_hot_triple")
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
    if perception_required == "original" or perception_required is None:
        perception_list = []
    elif perception_required == "lidar_detection_one_hot":
        perception_list = [controlled_vehicle.lidar_detection_one_hot(5, ox, oy)]
    elif perception_required == "lidar_detection_one_hot_triple":
        perception_list = [np.concatenate([controlled_vehicle.lidar_detection_one_hot(5, ox, oy), controlled_vehicle.lidar_detection_one_hot(10, ox, oy),\
            controlled_vehicle.lidar_detection_one_hot(15, ox, oy)])]
    for action_clipped in control_list:
        controlled_vehicle.step(action_clipped, 1 / simulation_freq)
        state_list.append(np.array(controlled_vehicle.state).astype(np.float32))
        if perception_required == "lidar_detection_one_hot":
            perception_list.append(controlled_vehicle.lidar_detection_one_hot(5, ox, oy))
        elif perception_required == "lidar_detection_one_hot_triple":
            perception_list.append(np.concatenate([controlled_vehicle.lidar_detection_one_hot(5, ox, oy), controlled_vehicle.lidar_detection_one_hot(10, ox, oy),\
            controlled_vehicle.lidar_detection_one_hot(15, ox, oy)]))
        if controlled_vehicle._is_jack_knife():
            result_dict = {
                "state_list": state_list,
                "jack_knife": True,
                "collision": False,
                "goal_reached": False,
                "perception_list": perception_list,
            }
            return result_dict
        if controlled_vehicle.is_collision(ox, oy):
            result_dict = {
                "state_list": state_list,
                "jack_knife": False,
                "collision": True,
                "goal_reached": False,
                "perception_list": perception_list,
            }
            return result_dict
    final_state = np.array(controlled_vehicle.state)
    distance_error = common.mixed_norm(goal, final_state)
    # distance_error = np.linalg.norm(goal - final_state)
    if distance_error < 0.5:
        result_dict = {
            "state_list": state_list,
            "jack_knife": False,
            "collision": False,
            "goal_reached": True,
            "perception_list": perception_list,
            "final_state": final_state,
        }
        return result_dict
    else:
        result_dict = {
            "state_list": state_list,
            "jack_knife": False,
            "collision": False,
            "goal_reached": False,
            "perception_list": perception_list,
        }
        return result_dict
    
def generate_ox_oy_from_obstacles_info(obstacles_info, map_vertices):
    """get ox, oy directly using the obstacles_info and map_vertices"""
    Map = tt_envs.MapBound(map_vertices)
    ox_map, oy_map = Map.sample_surface(0.1)
    ox = ox_map
    oy = oy_map
    if (obstacles_info is not None) and len(obstacles_info) > 0:
        for rectangle in obstacles_info:
            obstacle = tt_envs.QuadrilateralObstacle(rectangle)
            ox_obs, oy_obs = obstacle.sample_surface(0.1)
            ox += ox_obs
            oy += oy_obs
    return ox, oy

def check_is_start_feasible(input, obstacles_info, map_vertices, config):
    """this api is for checking whether the start is feasible"""
    ox, oy = generate_ox_oy_from_obstacles_info(obstacles_info, map_vertices)
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    is_start_feasible = three_trailer_planner.check_start_mp_feasible(input)
    return is_start_feasible

def find_astar_trajectory(input, goal, obstacles_info, map_vertices, config, perception_required=None):
    """
    This is a common api for utilizing the astar planner to find the trajectory
    map_vertices: given as a list of the four vertices
    return result_dict
    """
    assert config is not None, "planner config should be initialized"
    if config["heuristic_type"] == "traditional":
        planner_version = '0'
    elif config["heuristic_type"] == "mix":
        planner_version = '1'
    elif config["heuristic_type"] == "mix_lidar_detection_one_hot":
        planner_version = '2'
    ox, oy = generate_ox_oy_from_obstacles_info(obstacles_info, map_vertices)
    three_trailer_planner = alg_obs.ThreeTractorTrailerHybridAstarPlanner(ox, oy, config=config)
    start_planning_time = time.time()
    path, control_list, rs_path, expand_node_number = three_trailer_planner.plan_new_version(input, goal, get_control_sequence=True, verbose=False, obstacles_info=obstacles_info, map_vertices=map_vertices)
    end_planning_time = time.time()
    if control_list is None:
        result_dict = {
            "state_list": [input],
            "control_list": [],
            "jack_knife": False, 
            "collision": False,
            "goal_reached": False,
            "planning_time": end_planning_time - start_planning_time,
            "expand_node_number": expand_node_number,
            "planner_version": planner_version,
        }
    else:
        control_recover_list = common.action_recover_from_planner(control_list, simulation_freq=10, v_max=config["controlled_vehicle_config"]["v_max"], max_steer=config["controlled_vehicle_config"]["max_steer"])
        result_dict = test_forward_simulation_three_trailer(input, goal, ox, oy, control_recover_list, simulation_freq=10, perception_required=perception_required)
        result_dict["control_list"] = control_recover_list
        result_dict["planning_time"] = end_planning_time - start_planning_time
        result_dict["expand_node_number"] = expand_node_number
        result_dict["planner_version"] = planner_version
    return result_dict


def check_waypoint_legal(env, waypoint):
    """The function helps to check whether the waypoint is legal
    - env: the env need to be reset
    - waypoint: np_array
    """
    bounding_box_list = env.unwrapped.controlled_vehicle.get_bounding_box_list(waypoint)
    if env.unwrapped.check_bounding_box_list_inside_map(bounding_box_list) and env.unwrapped.check_bounding_box_list_not_collide_obstacles(bounding_box_list, env.unwrapped.obstacles_info):
        return True
    else:
        return False
    

def find_astar_trajectory_two_phases(env, input, goal, waypoint, obstacles_info, map_vertices, config, perception_required=None):
    """This is a two phases planning version of 
    planning algorithms
    waypoint: 6-dim equilibrium
    """
    assert config is not None, "planner config should be initialized"
    ox, oy = generate_ox_oy_from_obstacles_info(obstacles_info, map_vertices)
    if not check_waypoint_legal(env, waypoint):
        print("waypoint illegal")
        result_dict = {
            "state_list": [input],
            "control_list": [],
        }
        return result_dict

    result_dict1 = find_astar_trajectory(input, waypoint, obstacles_info, map_vertices, config, perception_required)
    if not result_dict1["goal_reached"]:
        print("first phase planning failed")
        result_dict = {
            "state_list": [input],
            "control_list": [],
        }
        return result_dict
    control_list1 = result_dict1.get("control_list")
    final_state1 = result_dict1.get("final_state")
    
    # control_list2, finetune_final_state1 = finetune_trajectory(final_state1, ox, oy)
    # if control_list2 is None:
    #     print("finetune failed")
    #     return
    # result_dict2 = find_astar_trajectory(finetune_final_state1, goal, obstacles_info, map_vertices, config, perception_required)
    result_dict2 = find_astar_trajectory(final_state1, goal, obstacles_info, map_vertices, config, perception_required)
    if not result_dict2["goal_reached"]:
        print("second phase planning failed")
        result_dict = {
            "state_list": [input],
            "control_list": [],
        }
        return result_dict
    control_list3 = result_dict2.get("control_list")
    final_state2 = result_dict2.get("final_state")
    control_list4, finetune_final_state2 = finetune_trajectory(final_state2, ox, oy)
    if control_list4 is None:
        print("finetune failed")
        # control_list = control_list1 + control_list2 + control_list3
        control_list = control_list1 + control_list3
    else:
        # control_list = control_list1 + control_list2 + control_list3 + control_list4
        control_list = control_list1 + control_list3 + control_list4
    result_dict = test_forward_simulation_three_trailer(input, goal, ox, oy, control_list, simulation_freq=10, perception_required=perception_required)
    
    result_dict["control_list"] = control_list
    return result_dict

def finetune_trajectory(state, ox ,oy):
    """Here I want to fullfill a function that can return a finetune policy given the current to fix it to a equilibrium state
    - state: np_array, the current state of the vehicle
    output: control_list, final_state
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
    x, y, yaw, yawt1, yawt2, yawt3 = state
    if not check_finetune_shape(state):
        print("The vehicle shape is not suitable for finetune")
        return None, None
    else:
        controlled_vehicle.reset(x, y, yaw, yawt1, yawt2, yawt3)
        lidar_detection_one_hot = controlled_vehicle.lidar_detection_one_hot(5.3, ox, oy)
        if lidar_detection_one_hot[1] == 1:
            print("Obstacles setting not allowed this finetune")
            return None, None
        else:
            number_1 = 26
            number_2 = 60
            control_list = []
            for i in range(number_2):
                control_list += [np.array([1, 0])] * number_1 + [np.array([-1, 0])] * number_1
            for control in control_list:
                simulation_freq = 10
                controlled_vehicle.step(control, 1 / simulation_freq)
                if controlled_vehicle.is_collision(ox, oy):
                    print("collision occurred when executing the finetune")
                    return None, None
        final_state = np.array(controlled_vehicle.state) 
    return control_list, final_state

def visualize_planner_final_result(input, goal, obstacles_info, map_vertices, result_dict):
    Map = tt_envs.MapBound(map_vertices)
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
    plt.cla()
    ax = plt.gca()
    plt.plot(ox, oy, 'sk', markersize=0.5)
    controlled_vehicle_config = {
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
        "safe_metric": 3.0, #[m] the safe metric for calculation
        "xi_max": (np.pi) / 4, # jack-knife constraint  
    }
    controlled_vehicle = tt_envs.ThreeTrailer(controlled_vehicle_config)
    gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = goal
    controlled_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
    controlled_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
    state_list = result_dict.get("state_list")
    control_list = result_dict.get("control_list")
    if len(control_list) == 0:
        print("failed planning")
        return
    sx, sy, syaw0, syawt1, syawt2, syawt3 = input
    rx = []
    ry = []
    for state in state_list:
        rx.append(state[0])
        ry.append(state[1])
    plt.plot(rx, ry, 'r-', linewidth=1)
    controlled_vehicle.reset(sx, sy, syaw0, syawt1, syawt2, syawt3)
    controlled_vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue')
    
    sx, sy, syaw0, syawt1, syawt2, syawt3 = state_list[-1]
    controlled_vehicle.reset(sx, sy, syaw0, syawt1, syawt2, syawt3)
    controlled_vehicle.plot(ax,control_list[-1], 'blue')
    plt.axis('equal')    
    plt.savefig("rl_training/tractor_trailer_planning.png")
    plt.close()
    
def check_and_adjust_goal(goal, d1, d2, ox, oy):
    """Check if the goal is feasible; if not, adjust the goal based on d2.
    
    -d1: check the last trailer's right behind, d1(2m)
    -d2: check the tractor's front, d2(5m)
    - return: a dict containing the original goal, adjusted goal, and whether the original goal is feasible
    """
    # Vehicle configuration remains unchanged
    controlled_vehicle_config = {
        "w": 2.0,  # [m] width of vehicle
        "wb": 3.5,  # [m] wheel base: rear to front steer
        "wd": 1.4,  # [m] distance between left-right wheels (0.7 * W)
        "rf": 4.5,  # [m] distance from rear to vehicle front end
        "rb": 1.0,  # [m] distance from rear to vehicle back end
        "tr": 0.5,  # [m] tyre radius
        "tw": 1.0,  # [m] tyre width
        "rtr": 2.0,  # [m] rear to trailer wheel
        "rtf": 1.0,  # [m] distance from rear to trailer front end
        "rtb": 3.0,  # [m] distance from rear to trailer back end
        "rtr2": 2.0,  # [m] rear to second trailer wheel
        "rtf2": 1.0,  # [m] distance from rear to second trailer front end
        "rtb2": 3.0,  # [m] distance from rear to second trailer back end
        "rtr3": 2.0,  # [m] rear to third trailer wheel
        "rtf3": 1.0,  # [m] distance from rear to third trailer front end
        "rtb3": 3.0,  # [m] distance from rear to third trailer back end
        "max_steer": 0.6,  # [rad] maximum steering angle
        "v_max": 2.0,  # [m/s] maximum velocity
        "safe_d": 0.0,  # [m] the safe distance from the vehicle to obstacle
        "safe_metric": 3.0,  # [m] the safe metric for calculation
        "xi_max": (np.pi) / 4,  # jack-knife constraint
    }
    result_dict = {
        "original_goal": goal,
        "adjusted_goal": goal,
        "is_original_feasible": False,
        "is_adjusted_feasible": False,
    }
    controlled_vehicle = tt_envs.ThreeTrailer(controlled_vehicle_config)
    gx, gy, gyaw0, gyawt1, gyawt2, gyawt3 = goal
    controlled_vehicle.reset(gx, gy, gyaw0, gyawt1, gyawt2, gyawt3)
    lidar_detection_one_hot = controlled_vehicle.lidar_detection_one_hot(d1, ox, oy)

    if lidar_detection_one_hot[32] == 0:
        # Goal is feasible as is
        result_dict["is_original_feasible"] = True
        return result_dict
    else:
        # Adjust goal based on d2 and recheck
        result_dict["is_original_feasible"] = False
        lidar_detection_one_hot2 = controlled_vehicle.lidar_detection_one_hot(d2, ox, oy)
        if lidar_detection_one_hot2[1] == 0:
            result_dict["is_adjusted_feasible"] = True
            
            # Front is clear, adjust gx by d2 and return new goal
            adjusted_goal = np.array([gx + d2, gy, gyaw0, gyawt1, gyawt2, gyawt3], dtype=np.float32)
            result_dict["adjusted_goal"] = adjusted_goal
            return result_dict
        else:
            # Goal remains unfeasible
            return result_dict