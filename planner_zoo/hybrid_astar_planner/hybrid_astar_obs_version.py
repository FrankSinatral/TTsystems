import os
import sys
import math
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
import re
from heapdict import heapdict
# import scipy.spatial.kdtree as kd
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional
import tractor_trailer_envs as tt_envs
import pickle
import torch

import gymnasium as gym
import rl_agents as agents
import yaml
import torch.nn as nn
import curves_generator

import planner_zoo.hybrid_astar_planner.hybrid_astar as hyastar
   
def plot_rs_path(rspath, ox, oy):
    plt.axis("equal")
    plt.plot(ox, oy, 'sk', markersize=1)
    xlist = rspath.x
    ylist = rspath.y
    plt.plot(xlist, ylist, 'b')
    
def plot_rl_path(rlpath, ox, oy):
    plt.axis("equal")
    plt.plot(ox, oy, 'sk', markersize=1)
    xlist = rlpath.x
    ylist = rlpath.y
    plt.plot(xlist, ylist, 'C1')
    
def plot_map(ox, oy):
    plt.axis("equal")
    plt.plot(ox, oy, 'sk', markersize=1)
    
def gym_reaching_tt_env_fn(config: dict): 
    return gym.make("tt-reaching-v0", config=config)


# Apply our new-tt planning env to this agent for training
def gym_tt_planning_env_fn(config: dict):
    import gymnasium as gym
    import tractor_trailer_envs as tt_envs
    if not hasattr(gym_tt_planning_env_fn, "envs_registered"):
        tt_envs.register_tt_envs()
        gym_tt_planning_env_fn.envs_registered = True
    return gym.make("tt-planning-v0", config=config)

def convert_obstacles_to_local(obstacles_info, n_curr_x, n_curr_y, n_curr_yaw):
    local_obstacles_info = []
    for obstacle in obstacles_info:
        local_obstacle = []
        for (x, y) in obstacle:
            # global->local
            x_local = np.cos(n_curr_yaw) * (x - n_curr_x) + np.sin(n_curr_yaw) * (y - n_curr_y)
            y_local = -np.sin(n_curr_yaw) * (x - n_curr_x) + np.cos(n_curr_yaw) * (y - n_curr_y)
            local_obstacle.append((x_local, y_local))
        local_obstacles_info.append(local_obstacle)
    return local_obstacles_info

def convert_map_vertices_to_local(map_vertices, n_curr_x, n_curr_y, n_curr_yaw):
    local_map_vertices = []
    
    for (x, y) in map_vertices:
        # global->local
        x_local = np.cos(n_curr_yaw) * (x - n_curr_x) + np.sin(n_curr_yaw) * (y - n_curr_y)
        y_local = -np.sin(n_curr_yaw) * (x - n_curr_x) + np.cos(n_curr_yaw) * (y - n_curr_y)
        local_map_vertices.append((x_local, y_local))
    
    return local_map_vertices
    
def extract_rs_path_control(rspath, max_steer, maxc, N_step=10, max_step_size=0.2):
    """extract rs path control from a given rs path
    Note that this rs path will not be scaling back
    Note that this is still not the proper action for rl env action_clipped
    N_step: how many steps to pack
    max_step_size: the maximun step_size taken
    Note that this is irrelevant to vehicle type agnostic
    """
    # TODO: here steps is very important
    rscontrol_list = []
    for ctype, length in zip(rspath.ctypes, rspath.lengths):
        if ctype == 'S':
            steer = 0
        elif ctype == 'WB':
            steer = max_steer
        else:
            steer = -max_steer
        step_number = math.floor((np.abs(length / maxc) / (N_step * max_step_size))) + 1
        
        action_step_size = (length / maxc) / (step_number * N_step)
        rscontrol_list += [np.array([action_step_size, steer])] * (step_number * N_step)
        
    return rscontrol_list

def action_recover_from_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6):
    # this shift is for rl api
    # transform the action to [-1, 1]
    new_control_list = []
    for control in control_list:
        new_control = np.array([control[0] * simulation_freq / v_max, control[1] / max_steer])
        new_control_list.append(new_control)
    
    return new_control_list

def action_recover_to_planner(control_list, simulation_freq=10, v_max=2, max_steer=0.6):
    # change action from rl to planner
    # note here max_steer is slightly different run in rl
    new_control_list = []
    for control in control_list:
        new_control = np.array([control[0] / (simulation_freq / v_max), control[1] * max_steer])
        new_control_list.append(new_control)
    
    return new_control_list

def cyclic_angle_distance(angle1, angle2):
    """calculate the cyclic distance between two angles"""
    diff = np.abs(angle1 - angle2)
    return min(diff, 2*np.pi - diff)

def mixed_norm(goal, final_state):
    # calculate the position component of the periodic square distance
    position_diff_square = np.sum((goal[:2] - final_state[:2]) ** 2)
    
    # calculate the angle component of the periodic square distance
    angle_diff_square = sum([cyclic_angle_distance(goal[i], final_state[i]) ** 2 for i in range(2, 6)])
    
    # calculate the total periodic square distance
    total_distance = np.sqrt(position_diff_square + angle_diff_square)
    return total_distance

def process_obstacles_properties_to_array(input_list):
    """process for mlp with obstacles properties"""
    array_length = 40
    result_array = np.zeros(array_length, dtype=np.float32)
    
    # 将input_list中的元素顺次填入result_array中
    for i, (x, y, l, d) in enumerate(input_list):
        if i >= 10:
            break
        result_array[i*4:i*4+4] = [x, y, l, d]
    
    return result_array

class RlPath:
    pass

class SingleTractorHybridAstarPlanner(hyastar.BasicHybridAstarPlanner):
    
    @classmethod
    def default_config(cls) -> dict:
        return {
            "verbose": False, 
            "vehicle_type": "single_tractor",
            "act_limit": 1, 
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
            "xy_reso": 2.0,
            "yaw_reso": np.deg2rad(15.0),
            "qp_type": "heapdict",
            "max_iter": 1000, # outer loop number
            "step_size": 0.2, # rs path step size
            "move_step": 0.2, # expand tree how many distance to move
            "mp_step": 40.0, # previous 2.0 * reso
            "is_save_animation": True,
            "is_save_expand_tree": True,
            "visualize_mode": True,
            # "dt": 0.1,
            "heuristic_reso": 2.0,
            "heuristic_rr": 1.0, # 0.5 * heuristic_reso
            "whether_obs": True,
            "safe_d": 0.0,
            "extend_area": 0.0,
            "collision_check_step": 10,
            "cost_configuration":
                {
                    "scissors_cost": 200.0,
                    "gear_cost": 100.0,
                    "backward_cost": 5.0,
                    "steer_change_cost": 5.0,
                    "h_cost": 10.0,
                    "steer_angle_cost": 1.0,
                },
            "n_steer": 20,
            "plot_heuristic_nonholonomic": False,
            "plot_rs_path": True,
            "plot_expand_tree": True,
            "plot_final_path": True,
            "plot_failed_path": False,
            "range_steer_set": 8, #need to set the same as n_steer
            "acceptance_error": 0.2,
        }
    
    def configure(self, config: Optional[dict]):
        if config:
            self.config.update(config)
        self.vehicle = tt_envs.SingleTractor(self.config["controlled_vehicle_config"])
        self.max_iter = self.config["max_iter"] 
        self.xyreso = self.config["xy_reso"]
        self.yawreso = self.config["yaw_reso"]
        self.qp_type = self.config["qp_type"] 
        self.safe_d = self.config["safe_d"]
        self.extend_area = self.config["extend_area"]
        self.obs = self.config['whether_obs']
        self.cost = self.config['cost_configuration']
        self.step_size = self.config["step_size"]
        self.n_steer = self.config["n_steer"]
        if self.obs:
            self.heuristic_reso = self.config["heuristic_reso"]
            self.heuristic_rr = self.config["heuristic_rr"]
        if self.qp_type == "heapdict":
            self.qp = hyastar.NewQueuePrior()
        else:
            self.qp = hyastar.QueuePrior()
    
    def __init__(self, ox, oy, config: Optional[dict] = None):
        self.config = self.default_config()
        self.configure(config)
        # self.vehicle = md.C_single_tractor(args)
        
        super().__init__(ox, oy)
    
    def calc_parameters(self):
        minxm = min(self.ox) - self.extend_area
        minym = min(self.oy) - self.extend_area
        maxxm = max(self.ox) + self.extend_area
        maxym = max(self.oy) + self.extend_area

        self.ox.append(minxm)
        self.oy.append(minym)
        self.ox.append(maxxm)
        self.oy.append(maxym)

        minx = round(minxm / self.xyreso)
        miny = round(minym / self.xyreso)
        maxx = round(maxxm / self.xyreso)
        maxy = round(maxym / self.xyreso)

        xw, yw = maxx - minx + 1, maxy - miny + 1

        minyaw = round(-self.vehicle.PI / self.yawreso)
        maxyaw = round(self.vehicle.PI / self.yawreso)
        yaww = maxyaw - minyaw + 1

        P = hyastar.Para_single_tractor(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, self.xyreso, self.yawreso, self.ox, self.oy, self.kdtree)

        return P
    
    def calc_motion_set(self):
        """
        this is much alike motion primitives
        """
        s = [i for i in np.arange(self.vehicle.MAX_STEER / self.n_steer,
                                 self.config["range_steer_set"] * self.vehicle.MAX_STEER / self.n_steer, self.vehicle.MAX_STEER / self.n_steer)]

        steer = [0.0] + s + [-i for i in s]
        direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
        steer = steer + steer

        return steer, direc

    
    def calc_index(self, node):
        '''
        change the way to calculate node index
        '''
        ind = (node.yawind - self.P.minyaw) * self.P.xw * self.P.yw + \
            (node.yind - self.P.miny) * self.P.xw + \
            (node.xind - self.P.minx)
        return ind
    
    def calc_next_node(self, n, ind, u, d):
        '''
        Using the current node/ind and steer/direction to 
        generate new node
        
        n: current node (Node class)
        ind: node index (calc_index)
        u: steer
        d: direction
        P: parameters
        
        returns:
        a node class
        '''
        step = self.config["mp_step"]

        nlist = math.ceil(step / self.config["move_step"])
        xlist = [n.x[-1] + d * self.config["move_step"] * math.cos(n.yaw[-1])]
        ylist = [n.y[-1] + d * self.config["move_step"] * math.sin(n.yaw[-1])]
        yawlist = [self.pi_2_pi(n.yaw[-1] + d * self.config["move_step"] / self.vehicle.WB * math.tan(u))]
        

        for i in range(nlist - 1):
            xlist.append(xlist[i] + d * self.config["move_step"] * math.cos(yawlist[i]))
            ylist.append(ylist[i] + d * self.config["move_step"] * math.sin(yawlist[i]))
            yawlist.append(self.pi_2_pi(yawlist[i] + d * self.config["move_step"] / self.vehicle.WB * math.tan(u)))

        xind = round(xlist[-1] / self.xyreso)
        yind = round(ylist[-1] / self.xyreso)
        yawind = round(yawlist[-1] / self.yawreso)

        # The following includes the procedure to 
        # calculate the cost of each node
        cost = 0.0

        if d > 0:
            direction = 1.0
            cost += abs(step)
        else:
            direction = -1.0
            cost += abs(step) * self.cost["backward_cost"]

        if direction != n.direction:  # switch back penalty
            cost += self.cost["gear_cost"]

        cost += self.cost["steer_angle_cost"] * abs(u)  # steer penalyty
        cost += self.cost["steer_change_cost"] * abs(n.steer - u)  # steer change penalty
        
        cost = n.cost + cost

        directions = [direction for _ in range(len(xlist))]

        # there is no jack-knife to validate
        try:
            node = hyastar.Node_single_tractor(xind, yind, yawind, direction, xlist, ylist,
                        yawlist, directions, u, cost, ind)
        except:
            return None

        return node
    
    def is_index_ok(self, node, collision_check_step: int) -> bool:
        """
        check if the node is legal for a single tractor system
        - node: calc node (Node class)
        - P: parameters
        returns:
        whether the current node is ok
        """
        # check node index
        # check whether to go outside
        if node.xind <= self.P.minx or \
                node.xind >= self.P.maxx or \
                node.yind <= self.P.miny or \
                node.yind >= self.P.maxy:
            return False

        ind = range(0, len(node.x), collision_check_step)

        x = [node.x[k] for k in ind]
        y = [node.y[k] for k in ind]
        yaw = [node.yaw[k] for k in ind]

        if self.is_collision(x, y, yaw):
            return False

        return True
    
    def is_collision(self, x, y, yaw):
        '''
        check whether there is collision
        Inputs:
        x, y, yaw, yawt1, yawt2, yawt3: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        for ix, iy, iyaw in zip(x, y, yaw):
            # first trailer test collision
            d = self.safe_d
                        
            # check the tractor collision
            deltal = (self.vehicle.RF - self.vehicle.RB) / 2.0
            rc = (self.vehicle.RF + self.vehicle.RB) / 2.0 + d

            cx = ix + deltal * math.cos(iyaw)
            cy = iy + deltal * math.sin(iyaw)

            ids = self.P.kdtree.query_ball_point([cx, cy], rc)

            if ids:
                for i in ids:
                    xo = self.P.ox[i] - cx
                    yo = self.P.oy[i] - cy

                    dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                    dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                    if abs(dx_car) <= rc and \
                            abs(dy_car) <= self.vehicle.W / 2.0 + d:
                        return True

        return False
     
    def is_same_grid(self, node1, node2):
        """
        whether the two nodes are on the same grid
        """
        if node1.xind != node2.xind or \
                node1.yind != node2.yind or \
                node1.yawind != node2.yawind:
            return False

        return True
    
    def is_the_start(self, node1, node2):
        """
        whether the two nodes are all start node
        """
        if len(node1.x) == 1 and len(node2.y) == 1:
            return True
        return False
     
    def rs_gear(self, node, ngoal):
        # Fank: put all rs related tech here
        # for single tractor
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        # I add a new attribute to this function 
        # Using a simplified version of calc_all_paths
        paths = self.calc_all_paths(node, ngoal, maxc)
        
        
        find_feasible = False
        if not paths:
            return find_feasible, None
        pq = hyastar.QueuePrior()
        
        for path in paths:
            if path.info["jack_knife"] == False:
                # find a suitable rs path
                # that is acceptable, no collision and no jack_knife
                find_feasible = True
                return find_feasible, path
            pq.put(path, path.rscost)
        # After put all the rs path into pq, 
        # we use the minimal rspath cost as our heuristic
        #TODO: may have to adjust
        while not pq.empty():
            path = pq.get()
            find_feasible = False
            return find_feasible, path
        
    def forward_simulation_single_tractor(self, input, goal, control_list, simulation_freq=10):
        # Fank: use the rs_path control we extract to forward simulation to 
        # check whether suitable this path
        # control_list: clip to [-1,1]
        # not pack
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
        path_x_list, path_y_list, path_yaw_list = [], [], []
        directions = []
        # implement a new vehicle
        controlled_vehicle = tt_envs.SingleTractor(config_dict)
        # Using input to reset
        controlled_vehicle.reset(*input)
        path_x, path_y, path_yaw = controlled_vehicle.state
        path_x_list.append(path_x)
        path_y_list.append(path_y)
        path_yaw_list.append(path_yaw)
        for action_clipped in control_list:
            if action_clipped[0] > 0:
                directions.append(1)
            else:
                directions.append(-1)
            controlled_vehicle.step(action_clipped, 1 / simulation_freq)
            path_x, path_y, path_yaw = controlled_vehicle.state
            path_x_list.append(path_x)
            path_y_list.append(path_y)
            path_yaw_list.append(path_yaw)
            
        directions.append(directions[-1])
        final_state = np.array(controlled_vehicle.state)
        # distance_error = np.linalg.norm(goal - final_state)
        distance_error = mixed_norm(goal, final_state)
        # Fank: accept(false means not good)
        #       collision(false means no collision)
        #       jack_knife(false means no jack_knife)
        info = {
            "accept": False,
            "collision": None,
            "jack_knife": None,
        }
        if distance_error > self.config["acceptance_error"]:
            info["accept"] = False
        else:
            info["accept"] = True
        
        if info["accept"]:
            # Fank: check whether collision here
            ind = range(0, len(path_x_list), self.config["collision_check_step"])
            pathx = [path_x_list[k] for k in ind]
            pathy = [path_y_list[k] for k in ind]
            pathyaw = [path_yaw_list[k] for k in ind]
            if self.is_collision(pathx, pathy, pathyaw):
                info["collision"] = True
            else:
                # no collision
                info["collision"] = False
        
        return path_x_list, path_y_list, path_yaw_list, directions, info
    
    def calc_all_paths(self, node, ngoal, maxc):
        # Fank: 
        # Input: node - start node
        #        nogal - goal node
        #        maxc - maximum culvature
        # this function adds more information for the rspath we selected
        
        sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
        gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]
        q0 = [sx, sy, syaw]
        q1 = [gx, gy, gyaw]
        input = np.array([sx, sy, syaw])
        goal = np.array([gx, gy, gyaw])

        paths = curves_generator.generate_path(q0, q1, maxc)

        for path in paths:
            rscontrol_list = extract_rs_path_control(path, self.vehicle.MAX_STEER, maxc)
            control_list = action_recover_from_planner(rscontrol_list)
            path.x, path.y, path.yaw, path.directions, path.info = self.forward_simulation_single_tractor(input, goal, control_list)
            path.lengths = [l / maxc for l in path.lengths]
            path.L = path.L / maxc
            # add rscontrollist once search the path
            path.rscontrollist = rscontrol_list
            # put calc_rs_cost_here
            path.rscost = self.calc_rs_path_cost_single_tractor(path)
            # Fank: check here if there is jack_knife
            if path.info["accept"] and (not path.info["collision"]):    
                xind = round(path.x[-1] / self.xyreso)
                yind = round(path.y[-1] / self.xyreso)
                yawind = round(path.yaw[-1] / self.yawreso)
                direction = path.directions[-1]
                fpind =  self.calc_index(node) 
                fcost = node.cost + path.rscost
                fx = path.x[1:]
                fy = path.y[1:]
                fyaw = path.yaw[1:]
                fd = path.directions[1:]
                # for d in path.directions[1:]:
                #     if d >= 0:
                #         fd.append(1.0)
                #     else:
                #         fd.append(-1.0)
                fsteer = 0.0
                try:
                    final_node = hyastar.Node_single_tractor(self.vehicle, xind, yind, yawind, direction,
                        fx, fy, fyaw, fd, fsteer, fcost, fpind)
                    path.info["jack_knife"] = False
                    path.info["final_node"] = final_node
                except:
                    path.info["jack_knife"] = True

        return paths
    
    def calc_rs_path_cost_single_tractor(self, rspath) -> float:
        """
        A newly version that rspath contains all the information
        this function calculate rs path cost based on rspath and yawt
        the calculate will be slightly different from node expansion
        Inputs:
        - rspath: path class
        - yawt: the first trailer yaw
        """
        cost = 0.0

        for lr in rspath.lengths:
            if lr >= 0:
                cost += abs(lr)
            else:
                cost += abs(lr) * self.cost["backward_cost"]

        for i in range(len(rspath.lengths) - 1):
            if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
                cost += self.cost["gear_cost"]

        for ctype in rspath.ctypes:
            if ctype != "S":
                cost += self.cost["steer_angle_cost"] * abs(self.vehicle.MAX_STEER)

        nctypes = len(rspath.ctypes)
        ulist = [0.0 for _ in range(nctypes)]

        for i in range(nctypes):
            if rspath.ctypes[i] == "R":
                ulist[i] = -self.vehicle.MAX_STEER
            elif rspath.ctypes[i] == "WB":
                ulist[i] = self.vehicle.MAX_STEER

        for i in range(nctypes - 1):
            cost += self.cost["steer_change_cost"] * abs(ulist[i + 1] - ulist[i])

        return cost
    
    def calc_hybrid_cost(self, n_curr, n_goal, rscost):
        # Fank: A new implement hybrid cost that does not 
        # need to recalculate rs path cost
        heuristic_non_holonomic = rscost
        heuristic_holonomic_obstacles = self.hmap[n_curr.xind - self.P.minx][n_curr.yind - self.P.miny]
        cost = n_curr.cost + \
             self.cost["h_cost"] * max(heuristic_non_holonomic, heuristic_holonomic_obstacles)

        return cost
    
    
    
    def plan(self, start:np.ndarray, goal:np.ndarray, get_control_sequence:bool, verbose=False, *args, **kwargs):
        """
        new version Main Planning Algorithm for single_tractor systems
        :param start: starting point (np_array)
        :param goal: goal point (np_array)
        - path: all the six-dim state along the way (using extract function)
        - rs_path: contains the rspath and rspath control list
        - control list: rspath control list + expand control list
        """
        
        self.sx, self.sy, self.syaw = start
        self.gx, self.gy, self.gyaw = goal
        self.syaw = self.pi_2_pi(self.syaw)
        self.gyaw = self.pi_2_pi(self.gyaw)
        self.sxr, self.syr = round(self.sx / self.xyreso), round(self.sy / self.xyreso)
        self.gxr, self.gyr = round(self.gx / self.xyreso), round(self.gy / self.xyreso)
        self.syawr = round(self.syaw / self.yawreso)
        self.gyawr = round(self.gyaw / self.yawreso)
        
        # initialize start and goal node class for single_tractor
        nstart = hyastar.Node_single_tractor(self.vehicle, self.sxr, self.syr, self.syawr, 1, \
            [self.sx], [self.sy], [self.pi_2_pi(self.syaw)], [1], 0.0, 0.0, -1)
        ngoal = hyastar.Node_single_tractor(self.vehicle, self.gxr, self.gyr, self.gyawr, 1, \
            [self.gx], [self.gy], [self.pi_2_pi(self.gyaw)], [1], 0.0, 0.0, -1)
        # check whether define outside or collision
        if not self.is_index_ok(nstart, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(ngoal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        # calculate heuristic for obstacle
        if self.obs:
            self.hmap = hyastar.calc_holonomic_heuristic_with_obstacle(ngoal, self.P.ox, self.P.oy, self.heuristic_reso, self.heuristic_rr)
        if self.config["plot_heuristic_nonholonomic"]:
            self.visualize_hmap(self.hmap)
        
        
        steer_set, direc_set = self.calc_motion_set()
        # Initialize open_set and closed_set
        open_set, closed_set = {self.calc_index(nstart): nstart}, {}
        
        # reset qp for next using
        self.qp.reset()
        find_rs_path = False
        count = 0
        update = False
        
        # Before expansion, we first check whether
        # there is a suitable rs path to connect or 
        # else write down the heuristic value
        find_feasible, path = self.rs_gear(nstart, ngoal)
        if find_feasible:
            fnode = path.info["final_node"]
            find_rs_path = True
            update = find_feasible
            rs_path = path
            rs_control_list = path.rscontrollist
            if self.config["plot_expand_tree"]:
                plot_rs_path(rs_path, self.ox, self.oy)
                self.plot_expand_tree(start, goal, closed_set, open_set)
                # plt.close()    
            if verbose:
                print("find path before expansion")
            closed_set[self.calc_index(nstart)] = nstart
        else:
            self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost(nstart, ngoal, path.rscost))
        
        # Main Loop
        while True:
            if update:
                # first check update to break the loop
                break
            # I will try not to use this
            # may need to modify when there's obstacle
            if (not open_set) or self.qp.empty():
                if self.config["plot_failed_path"]:
                    self.extract_failed_path(closed_set, nstart)
                return None
            count += 1
            # add if the loop is too much
            if count > self.max_iter:
                print("waste a long time to find")
                return None, None, None
            # pop the node index we want expand on
            ind = self.qp.get()
            # get current node from the open_set
            n_curr = open_set[ind]
            open_set.pop(ind)
            # put the node in the closed_set
            closed_set[ind] = n_curr
            
            # expand based on motion primitive
            for i in range(len(steer_set)):
                node = self.calc_next_node(n_curr, ind, steer_set[i], direc_set[i])
                if not node:
                    # encounter jack_knife
                    continue
                if not self.is_index_ok(node, self.config["collision_check_step"]):
                    # check go outside or collision
                    continue
                node_ind = self.calc_index(node)
                if node_ind in closed_set:
                    # we will not calculate twice 
                    # Note that this can be a limitation
                    continue
                if node_ind not in open_set:
                    open_set[node_ind] = node
                    find_feasible, path = self.rs_gear(node, ngoal)
                    if find_feasible:
                        fnode = path.info["final_node"]
                        find_rs_path = True
                        update = find_feasible
                        rs_path = path
                        rs_control_list = path.rscontrollist
                        if self.config["plot_expand_tree"]:
                            plot_rs_path(rs_path, self.ox, self.oy)
                            self.plot_expand_tree(start, goal, closed_set, open_set)
                            # plt.close()
                        if verbose:
                            print("final expansion node number:", count)
                        # Here you need to add node to closed set
                        closed_set[node_ind] = node
                        # break the inner expand_tree loop
                        break
                    else:
                        self.qp.put(node_ind, self.calc_hybrid_cost(node, ngoal, path.rscost))
                else:
                    if open_set[node_ind].cost > node.cost:
                        open_set[node_ind] = node
                        if self.qp_type == "heapdict":
                            find_feasible, path = self.rs_gear(node, ngoal)
                            if find_feasible:
                                fnode = path.info["final_node"]
                                find_rs_path = True
                                update = find_feasible
                                rs_path = path
                                rs_control_list = path.rscontrollist
                                if self.config["plot_expand_tree"]:
                                    plot_rs_path(rs_path, self.ox, self.oy)
                                    self.plot_expand_tree(start, goal, closed_set, open_set)
                                    # plt.close()
                                if verbose:
                                    print("final expansion node number:", count)
                                closed_set[node_ind] = node
                                break
                            else:    
                                self.qp.queue[node_ind] = self.calc_hybrid_cost(node, ngoal, path.rscost)
            
        # note that closed set and open_set will always overlap in 
        # the last element
        if verbose:
            print("final expand node: ", len(open_set) + len(closed_set) - 1)
        if get_control_sequence:
            path, expand_control_list = self.extract_path_and_control(closed_set, fnode, nstart,find_rs_path=find_rs_path)
            if find_rs_path:
                all_control_list = expand_control_list + rs_control_list
            else:
                rs_path = None
                all_control_list = all_control_list
            return path, all_control_list, rs_path
        else:
            if find_rs_path: 
                return self.extract_path(closed_set, fnode, nstart), None, rs_path
            else:
                return self.extract_path(closed_set, fnode, nstart), None, None
    
    def visualize_hmap(self, hmap):
        # x = ngoal.x[-1]
        # y = ngoal.y[-1]
        # yaw = ngoal.yaw[-1]
        # yawt1 = ngoal.yawt1[-1]
        # yawt2 = ngoal.yawt2[-1]
        # yawt3 = ngoal.yawt3[-1]
        
        # ox, oy = map_env1()
        # define your map
        hmap = np.where(np.isinf(hmap), np.nan, hmap)


        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        hmap_for_vis = np.flip(np.transpose(hmap), 0)
        
        # cax = ax.imshow(hmap_for_vis, cmap="inferno", extent=[-1, 41, -1, 41], aspect='auto')

        # # 设置轴的范围
        # ax.set_xlim(0, 40)
        # ax.set_ylim(0, 40)
        #differnt map and resolution set differently
        # x_min = 0 - (reso / 2)
        # x_max = 29 + (reso / 2)
        # y_min = 0 - (reso / 2)
        # y_max = 36 + (reso / 2)
        
        x_min = min(self.ox) - (self.heuristic_reso / 2)
        x_max = max(self.oy) + (self.heuristic_reso / 2)
        y_min = min(self.oy) - (self.heuristic_reso / 2)
        y_max = max(self.oy) + (self.heuristic_reso / 2)
        
        # cax = ax.imshow(hmap_for_vis, cmap="inferno", extent=[-1, 29, -5, 37], aspect='auto')
        cax = ax.imshow(hmap_for_vis, cmap="jet", extent=[x_min, x_max, y_min, y_max], aspect='auto')

        # 设置轴的范围
        # ax.set_xlim(0, 29)
        # ax.set_ylim(0, 36)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 添加颜色条
        cbar = fig.colorbar(cax)
        
        plt.plot(self.ox, self.oy, 'sk', markersize=1)

        plt.savefig('hmap_challenge_cases.png')
    
    def plot_expand_tree(self, start, goal, closed_set, open_set):
        plt.axis("equal")
        ax = plt.gca() 
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
        self.vehicle.reset(*goal)
        
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        self.vehicle.reset(*start)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
        for key, value in open_set.items():
            self.plot_node(value, color='gray')
        for key, value in closed_set.items():
            self.plot_node(value, color='red')
    
    def plot_node(self, node, color):
        xlist = node.x
        ylist = node.y
        plt.plot(xlist, ylist, color=color, markersize=1)
    
    def plot_real_path(self, rx, ry):
        plt.plot(rx, ry, color="blue", markersize=1)
    
    def visualize_planning(self, start: np.ndarray, goal: np.ndarray, path, 
                           gif=True, save_dir='./planner_result/gif'):
        """visuliaze the planning result
        : param path: a path class
        : start & goal: cast as np.ndarray
        """
        print("Start Visualize the Result")
        x = path.x
        y = path.y
        yaw = path.yaw
        direction = path.direction
        if gif:
            fig, ax = plt.subplots()

            def update(num):
                ax.clear()
                plt.axis("equal")
                k = num
                # plot env (obstacle)
                
                plt.plot(self.ox, self.oy, "sk", markersize=1)
                self.vehicle.reset(*start)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
                
                # plot the planning path
                plt.plot(x, y, linewidth=1.5, color='r')
                self.vehicle.reset(*goal)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
                self.vehicle.reset(x[k], y[k], yaw[k])
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0
                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')
                plt.axis("equal")

            ani = FuncAnimation(fig, update, frames=len(x), repeat=True)

            # Save the animation
            writer = PillowWriter(fps=20)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # base_path = "./planner_result/gif/rrt_path_plan_single_tractor" 
            base_path = os.path.join(save_dir, 'hybrid_astar_path_plan_single_tractor')
            extension = ".gif"
            
            all_files = os.listdir(save_dir)
            matched_files = [re.match(r'hybrid_astar_path_plan_single_tractor(\d+)\.gif', f) for f in all_files]
            numbers = [int(match.group(1)) for match in matched_files if match]
            
            if numbers:
                save_index = max(numbers) + 1
            else:
                save_index = 1
            ani.save(base_path + str(save_index) + extension, writer=writer)
            print("Done Plotting")
            
        else:
            # this is when your device has display setting
            fig, ax = plt.subplots()
            # this is when your device has display setting
            plt.pause(5)

            for k in range(len(x)):
                ax.clear()
                plt.axis("equal")
                # plot env (obstacle)
                plt.plot(self.ox, self.oy, 'sk', markersize=1)
                # plot the planning path
                plt.plot(x, y, linewidth=1.5, color='r')
                self.vehicle.reset(*start)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
                self.vehicle.reset(*goal)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')

                # calculate every time step
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    # different from a single car
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0
                # draw goal model
                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')
                plt.pause(0.0001)

            plt.show()

    def extract_path_and_control(self, closed, ngoal, nstart, reverse=False, find_rs_path=True):
        """
        extract the path 
        and the action sequence before rs path
        - find_rs_path: whether we find rs path (always yes)
        """
        rx, ry, ryaw, direc = [], [], [], []
        expand_control_list = []
        step = self.config["mp_step"]
        nlist = math.ceil(step / self.config["move_step"])
        node = ngoal
        count = 0
        # cost = ngoal.cost
        
        while True:
            #append the current node state configuration
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            direc += node.directions[::-1]
            # cost += node.cost

            if self.is_the_start(node, nstart) and self.is_same_grid(node, nstart):
                break
            if find_rs_path:
                if count > 0: #which means this is definitely not rs path
                    for i in range(nlist):
                        expand_control_list.append(np.array([node.directions[-1] * self.step_size, node.steer]))       
            else:
                for i in range(nlist):
                    expand_control_list.append(np.array([node.directions[-1] * self.step_size, node.steer
                                                        ]))
            # tracking parent ind
            node = closed[node.pind]
            count += 1
        if not reverse:
            rx = rx[::-1]
            ry = ry[::-1]
            ryaw = ryaw[::-1]
            direc = direc[::-1]
            direc[0] = direc[1]
        
        if self.config["plot_final_path"]:
            self.plot_real_path(rx, ry)
            save_dir = './planner_result/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            files = os.listdir(save_dir)
            
            file_index = 0
            file_name = f"hybrid_expand_tree_single_tractor_{file_index}.png"
            while file_name in files:
                file_index += 1
                file_name = f"hybrid_expand_tree_single_tractor_{file_index}.png"
            plt.savefig(os.path.join(save_dir, file_name))
        plt.close()  
        
            
        path = hyastar.Path_single_tractor(rx, ry, ryaw, direc)
        expand_control_list = expand_control_list[::-1]
        
        return path, expand_control_list
    
    def extract_path(self, closed, ngoal, nstart, reverse=False):
        """
        extract the path
        closed: closed_set (dictionary: key is node_ind, value is node class)
        ngoal: goal node class
        nstart: start node class
        reverse: whether to reverse or not
        
        returns:
        path class
        """
        rx, ry, ryaw, direc = [], [], [], []
        node = ngoal
        
        while True:
            #append the current node state configuration
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            direc += node.directions[::-1]

            if self.is_the_start(node, nstart) and self.is_same_grid(node, nstart):
                break
            # tracking parent ind
            node = closed[node.pind]
        if not reverse:
            rx = rx[::-1]
            ry = ry[::-1]
            ryaw = ryaw[::-1]
            direc = direc[::-1]
            direc[0] = direc[1]
            
        if self.config["plot_final_path"]:
            self.plot_real_path(rx, ry)
            plt.savefig("./planner_zoo/hybrid_expand_tree.png")
            plt.close()
        
        path = hyastar.Path_single_tractor(rx, ry, ryaw, direc)

        return path
    
    def extract_failed_path(self, closed, nstart):
        
        for value in closed.values():
            plt.plot(value.x, value.y,'.', color='grey', markersize=1)
        plt.plot(nstart.x, nstart.y, 'o', color='r', markersize=3)    
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
        # plt.legend()
        plt.axis("equal")
        if not os.path.exists("planner_result/failed_trajectory_single_tractor"):
            os.makedirs("planner_result/failed_trajectory_single_tractor")
            
        base_path = "./planner_result/failed_trajectory_single_tractor/"
        extension = ".png"
            
        all_files = os.listdir("./planner_result/failed_trajectory_single_tractor")
        matched_files = [re.match(r'explored(\d+)\.png', f) for f in all_files]
        numbers = [int(match.group(1)) for match in matched_files if match]
        
        if numbers:
            save_index = max(numbers) + 1
        else:
            save_index = 0
        plt.savefig(base_path + "explored" + str(save_index) + extension)
        plt.close()

class OneTractorTrailerHybridAstarPlanner(hyastar.BasicHybridAstarPlanner):
    @classmethod
    def default_config(cls) -> dict:
        return {
            "verbose": False, 
            "heuristic_type": "traditional",
            "vehicle_type": "one_trailer",
            "act_limit": 1, 
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
            "xy_reso": 2.0,
            "yaw_reso": np.deg2rad(15.0),
            "qp_type": "heapdict",
            "max_iter": 1000, # outer loop number
            "step_size": 0.2, # rs path sample step size
            "move_step": 0.2, # expand node tree
            "n_steer": 20, #how many parts to divide in motion primitives
            "mp_step": 40.0, # previous 2.0 * reso, how many moves in mp step
            "is_save_animation": True,
            "is_save_expand_tree": True,
            "visualize_mode": True,
            # "dt": 0.1,
            "heuristic_reso": 2.0,
            "heuristic_rr": 1.0, # 0.5 * heuristic_reso
            "whether_obs": True,
            "safe_d": 0.0,
            "extend_area": 0.0,
            "collision_check_step": 10,
            "goal_yaw_error": np.deg2rad(3.0),
            "cost_configuration":
                {
                    "scissors_cost": 200.0,
                    "gear_cost": 100.0,
                    "backward_cost": 5.0,
                    "steer_change_cost": 5.0,
                    "h_cost": 10.0,
                    "steer_angle_cost": 1.0,
                }, 
            "plot_heuristic_nonholonomic": False,
            "plot_rs_path": True,
            "plot_expand_tree": True,
            "plot_final_path": True,
            "plot_failed_path": False,
            "range_steer_set": 8, #need to set the same as n_steer
            "acceptance_error": 0.5,
        }
    
    def configure(self, config: Optional[dict]):
        if config:
            self.config.update(config)
        self.vehicle = tt_envs.OneTrailer(self.config["controlled_vehicle_config"])
        self.max_iter = self.config["max_iter"] 
        self.xyreso = self.config["xy_reso"]
        self.yawreso = self.config["yaw_reso"]
        self.qp_type = self.config["qp_type"] 
        self.safe_d = self.config["safe_d"]
        self.extend_area = self.config["extend_area"]
        self.obs = self.config['whether_obs']
        self.cost = self.config['cost_configuration']
        self.step_size = self.config["step_size"]
        self.n_steer = self.config["n_steer"]
        self.heuristic_type = self.config['heuristic_type']
        if self.obs:
            self.heuristic_reso = self.config["heuristic_reso"]
            self.heuristic_rr = self.config["heuristic_rr"]
        if self.qp_type == "heapdict":
            self.qp = hyastar.NewQueuePrior()
        else:
            self.qp = hyastar.QueuePrior()
            
        if self.heuristic_type == 'rl':
            self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with open("configs/agents/sac_astar.yaml", 'r') as file:
                config_algo = yaml.safe_load(file)
            
            env_name = config_algo['env_name']
            seed = config_algo['seed']
            exp_name = env_name + '_' + config_algo['algo_name'] + '_' + str(seed)
            logger_kwargs = {
                'output_dir': config_algo['logging_dir'] + exp_name,
                'output_fname': config_algo['output_fname'],
                'exp_name': exp_name,
            }
            if config_algo['activation'] == 'ReLU':
                activation_fn = nn.ReLU
            elif config_algo['activation'] == 'Tanh':
                activation_fn = nn.Tanh
            else:
                raise ValueError(f"Unsupported activation function: {config_algo['activation']}")
            ac_kwargs = {
                "hidden_sizes": tuple(config_algo['hidden_sizes']),
                "activation": activation_fn
            }
            with open("configs/envs/reaching_v0_eval.yaml", 'r') as file:
                config = yaml.safe_load(file)
            
            self.agent = agents.SAC_ASTAR(env_fn=gym_reaching_tt_env_fn,
                        algo=config_algo['algo_name'],
                        ac_kwargs=ac_kwargs,
                        seed=seed,
                        steps_per_epoch=config_algo['sac_steps_per_epoch'],
                        epochs=config_algo['sac_epochs'],
                        replay_size=config_algo['replay_size'],
                        gamma=config_algo['gamma'],
                        polyak=config_algo['polyak'],
                        lr=config_algo['lr'],
                        alpha=config_algo['alpha'],
                        batch_size=config_algo['batch_size'],
                        start_steps=config_algo['start_steps'],
                        update_after=config_algo['update_after'],
                        update_every=config_algo['update_every'],
                        # missing max_ep_len
                        logger_kwargs=logger_kwargs, 
                        save_freq=config_algo['save_freq'],
                        num_test_episodes=config_algo['num_test_episodes'],
                        log_dir=config_algo['log_dir'],
                        whether_her=config_algo['whether_her'],
                        use_automatic_entropy_tuning=config_algo['use_auto'],
                        env_name=config_algo['env_name'],
                        pretrained=config_algo['pretrained'],
                        pretrained_itr=config_algo['pretrained_itr'],
                        pretrained_dir=config_algo['pretrained_dir'],
                        whether_astar=config_algo['whether_astar'],
                        config=config,
                        device='cpu')
            filename = 'runs_rl/reaching-v0_sac_astar_30_20240116_214349/model_1999999.pth'
            self.agent.load(filename, whether_load_buffer=False)
            # self.clipped_action = self.agent.test_env.unwrapped.act_limit
            # self.agent_steps = self.agent.test_env.unwrapped.config["N_steps"]
            # self.rl_controlled_vehicle = tt_envs.OneTrailer(self.config["controlled_vehicle_config"])
            print("load rl agent planner from", filename)
            
    
    
    def __init__(self, ox, oy, config: Optional[dict] = None) -> None:
        self.config = self.default_config()
        self.configure(config)
        
        super().__init__(ox, oy)
    
    def calc_parameters(self):
        """calculate parameters of the planning problem
        return: para class implemented in hybrid_astar.py
        """
        minxm = min(self.ox) - self.extend_area
        minym = min(self.oy) - self.extend_area
        maxxm = max(self.ox) + self.extend_area
        maxym = max(self.oy) + self.extend_area

        self.ox.append(minxm)
        self.oy.append(minym)
        self.ox.append(maxxm)
        self.oy.append(maxym)

        minx = round(minxm / self.xyreso)
        miny = round(minym / self.xyreso)
        maxx = round(maxxm / self.xyreso)
        maxy = round(maxym / self.xyreso)

        xw, yw = maxx - minx + 1, maxy - miny + 1

        minyaw = round(-self.vehicle.PI / self.yawreso)
        maxyaw = round(self.vehicle.PI / self.yawreso)
        yaww = maxyaw - minyaw + 1

        minyawt1, maxyawt1, yawt1w = minyaw, maxyaw, yaww

        P = hyastar.Para_one_trailer(minx, miny, minyaw, minyawt1, maxx, maxy, maxyaw,
                maxyawt1, xw, yw, yaww, yawt1w, self.xyreso, self.yawreso, self.ox, self.oy, self.kdtree)

        return P
    
    def calc_motion_set(self):
        """
        this is much alike motion primitives
        """
        s = [i for i in np.arange(self.vehicle.MAX_STEER / self.n_steer,
                                self.config["range_steer_set"] * self.vehicle.MAX_STEER / self.n_steer, self.vehicle.MAX_STEER / self.n_steer)]

        steer = [0.0] + s + [-i for i in s]
        direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
        steer = steer + steer

        return steer, direc

    def calc_next_node(self, n, ind, u, d):
        '''
        Using the current node/ind and steer/direction to 
        generate new node
        
        n: current node (Node class)
        ind: node index (calc_index)
        u: steer
        d: direction
        P: parameters
        
        returns:
        a node class
        '''
        step = self.config["mp_step"]
        # step = self.xyreso * 2.0

        nlist = math.ceil(step / self.config["move_step"])
        xlist = [n.x[-1] + d * self.config["move_step"] * math.cos(n.yaw[-1])]
        ylist = [n.y[-1] + d * self.config["move_step"] * math.sin(n.yaw[-1])]
        yawlist = [self.pi_2_pi(n.yaw[-1] + d * self.config["move_step"] / self.vehicle.WB * math.tan(u))]
        yawt1list = [self.pi_2_pi(n.yawt1[-1] +
                            d * self.config["move_step"] / self.vehicle.RTR * math.sin(n.yaw[-1] - n.yawt1[-1]))]
        

        for i in range(nlist - 1):
            xlist.append(xlist[i] + d * self.config["move_step"] * math.cos(yawlist[i]))
            ylist.append(ylist[i] + d * self.config["move_step"] * math.sin(yawlist[i]))
            yawlist.append(self.pi_2_pi(yawlist[i] + d * self.config["move_step"] / self.vehicle.WB * math.tan(u)))
            yawt1list.append(self.pi_2_pi(yawt1list[i] +
                                    d * self.config["move_step"] / self.vehicle.RTR * math.sin(yawlist[i] - yawt1list[i])))

        xind = round(xlist[-1] / self.xyreso)
        yind = round(ylist[-1] / self.xyreso)
        yawind = round(yawlist[-1] / self.yawreso)

        # The following includes the procedure to 
        # calculate the cost of each node
        cost = 0.0

        if d > 0:
            direction = 1.0
            cost += abs(step)
        else:
            direction = -1.0
            cost += abs(step) * self.cost["backward_cost"]

        if direction != n.direction:  # switch back penalty
            cost += self.cost["gear_cost"]

        cost += self.cost["steer_angle_cost"] * abs(u)  # steer penalyty
        cost += self.cost["steer_change_cost"] * abs(n.steer - u)  # steer change penalty
        # may need to cancel this
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(yawlist, yawt1list)])  # jacknif cost
        
        cost = n.cost + cost

        directions = [direction for _ in range(len(xlist))]

        # check whether there is jack-knife state
        try:
            node = hyastar.Node_one_trailer(self.vehicle, xind, yind, yawind, direction, xlist, ylist,
                        yawlist, yawt1list, directions, u, cost, ind)
        except:
            return None

        return node
    
    def calc_index(self, node):
        '''
        change the way to calculate node index
        '''
        ind = (node.yawind - self.P.minyaw) * self.P.xw * self.P.yw + \
            (node.yind - self.P.miny) * self.P.xw + \
            (node.xind - self.P.minx)

        yawt1_ind = round(node.yawt1[-1] / self.P.yawreso)
        ind += (yawt1_ind - self.P.minyawt1) * self.P.xw * self.P.yw * self.P.yaww
        
        return ind
    
    def is_index_ok(self, node, collision_check_step: int) -> bool:
        """
        check if the node is legal for a one trailer system
        - node: calc node (Node class)
        - P: parameters
        returns:
        whether the current node is ok
        """
        # check node index
        # check whether to go outside
        if node.xind <= self.P.minx or \
                node.xind >= self.P.maxx or \
                node.yind <= self.P.miny or \
                node.yind >= self.P.maxy:
            return False

        ind = range(0, len(node.x), collision_check_step)

        x = [node.x[k] for k in ind]
        y = [node.y[k] for k in ind]
        yaw = [node.yaw[k] for k in ind]
        yawt1 = [node.yawt1[k] for k in ind]
        

        if self.is_collision(x, y, yaw, yawt1):
            return False

        return True
    
    def is_collision(self, x, y, yaw, yawt1):
        '''
        check whether there is collision
        Inputs:
        x, y, yaw, yawt1: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        for ix, iy, iyaw, iyawt1 in zip(x, y, yaw, yawt1):
            # first trailer test collision
            d = self.safe_d
            deltal1 = (self.vehicle.RTF + self.vehicle.RTB) / 2.0 #which is exactly C.RTR
            rt1 = (self.vehicle.RTB - self.vehicle.RTF) / 2.0 + d #half length of trailer1 plus d

            ctx1 = ix - deltal1 * math.cos(iyawt1)
            cty1 = iy - deltal1 * math.sin(iyawt1)

            idst1 = self.P.kdtree.query_ball_point([ctx1, cty1], rt1)

            if idst1:
                for i in idst1:
                    xot1 = self.P.ox[i] - ctx1
                    yot1 = self.P.oy[i] - cty1

                    dx_trail1 = xot1 * math.cos(iyawt1) + yot1 * math.sin(iyawt1)
                    dy_trail1 = -xot1 * math.sin(iyawt1) + yot1 * math.cos(iyawt1)

                    if abs(dx_trail1) <= rt1 and \
                            abs(dy_trail1) <= self.vehicle.W / 2.0 + d:
                        return True
                        
            # check the tractor collision
            deltal = (self.vehicle.RF - self.vehicle.RB) / 2.0
            rc = (self.vehicle.RF + self.vehicle.RB) / 2.0 + d

            cx = ix + deltal * math.cos(iyaw)
            cy = iy + deltal * math.sin(iyaw)

            ids = self.P.kdtree.query_ball_point([cx, cy], rc)

            if ids:
                for i in ids:
                    xo = self.P.ox[i] - cx
                    yo = self.P.oy[i] - cy

                    dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                    dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                    if abs(dx_car) <= rc and \
                            abs(dy_car) <= self.vehicle.W / 2.0 + d:
                        return True

        return False
    
    
    def calc_all_paths_modify(self, node, ngoal, maxc, step_size):
        # TODO: modify
        
        # newly add control list when exploring
        sx, sy, syaw, syawt1 = node.x[-1], node.y[-1], node.yaw[-1], node.yawt1[-1]
        gx, gy, gyaw, gyawt1 = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1], ngoal.yawt1[-1]
        q0 = [sx, sy, syaw]
        q1 = [gx, gy, gyaw]
        input = np.array([sx, sy, syaw, syawt1])
        goal = np.array([gx, gy, gyaw, gyawt1])

        paths = curves_generator.generate_path(q0, q1, maxc)

        for path in paths:
            rscontrol_list = extract_rs_path_control(path, self.vehicle.MAX_STEER, maxc)
            # this is convient for rl to run simulation
            control_list = action_recover_from_planner(rscontrol_list)
            path.x, path.y, path.yaw, path.yawt1, path.directions, path.valid = self.forward_simulation_one_trailer(input, goal, control_list)
            path.lengths = [l / maxc for l in path.lengths]
            path.L = path.L / maxc
            # add rscontrollist once search the path
            # note that here we document the step_size
            # not scaling to 1
            path.rscontrollist = rscontrol_list
            

        return paths
    
    def calc_all_paths_simplified(self, node, ngoal, maxc):
        # Fank: 
        # Input: node - start node
        #        nogal - goal node
        #        maxc - maximum culvature
        # this function adds more information for the rspath we selected
        
        sx, sy, syaw, syawt1 = node.x[-1], node.y[-1], node.yaw[-1], node.yawt1[-1]
        gx, gy, gyaw, gyawt1 = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1], ngoal.yawt1[-1]
        q0 = [sx, sy, syaw]
        q1 = [gx, gy, gyaw]
        input = np.array([sx, sy, syaw, syawt1])
        goal = np.array([gx, gy, gyaw, gyawt1])

        paths = curves_generator.generate_path(q0, q1, maxc)

        for path in paths:
            rscontrol_list = extract_rs_path_control(path, self.vehicle.MAX_STEER, maxc)
            control_list = action_recover_from_planner(rscontrol_list)
            path.x, path.y, path.yaw, path.yawt1, path.directions, path.info = self.forward_simulation_one_trailer_modify(input, goal, control_list)
            path.lengths = [l / maxc for l in path.lengths]
            path.L = path.L / maxc
            # add rscontrollist once search the path
            path.rscontrollist = rscontrol_list
            # put calc_rs_cost_here
            path.rscost = self.calc_rs_path_cost_one_trailer_modify(path)
            # Fank: check here if there is jack_knife
            if path.info["accept"] and (not path.info["collision"]):    
                xind = round(path.x[-1] / self.xyreso)
                yind = round(path.y[-1] / self.xyreso)
                yawind = round(path.yaw[-1] / self.yawreso)
                direction = path.directions[-1]
                fpind =  self.calc_index(node) 
                fcost = node.cost + path.rscost
                fx = path.x[1:]
                fy = path.y[1:]
                fyaw = path.yaw[1:]
                fyawt1 = path.yawt1[1:]
                fd = []
                for d in path.directions[1:]:
                    if d >= 0:
                        fd.append(1.0)
                    else:
                        fd.append(-1.0)
                fsteer = 0.0
                try:
                    final_node = hyastar.Node_one_trailer(self.vehicle, xind, yind, yawind, direction,
                        fx, fy, fyaw, fyawt1, fd, fsteer, fcost, fpind)
                    path.info["jack_knife"] = False
                    path.info["final_node"] = final_node
                except:
                    path.info["jack_knife"] = True

        return paths
    
    def forward_simulation_one_trailer(self, input, goal, control_list, simulation_freq=10):
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
        path_x_list, path_y_list, path_yaw_list, path_yawt1_list = [], [], [], []
        directions = []
        controlled_vehicle = tt_envs.OneTrailer(config_dict)
        controlled_vehicle.reset(*input)
        path_x, path_y, path_yaw, path_yawt1 = controlled_vehicle.state
        path_x_list.append(path_x)
        path_y_list.append(path_y)
        path_yaw_list.append(path_yaw)
        path_yawt1_list.append(path_yawt1)
        for action_clipped in control_list:
            if action_clipped[0] > 0:
                directions.append(1)
            else:
                directions.append(-1)
            controlled_vehicle.step(action_clipped, 1 / simulation_freq)
            path_x, path_y, path_yaw, path_yawt1 = controlled_vehicle.state
            path_x_list.append(path_x)
            path_y_list.append(path_y)
            path_yaw_list.append(path_yaw)
            path_yawt1_list.append(path_yawt1)
            
        directions.append(directions[-1])
        final_state = np.array(controlled_vehicle.state)
        # distance_error = np.linalg.norm(goal - final_state)
        distance_error = mixed_norm(goal, final_state)
        if distance_error < 0.5:
            info = True
        else:
            info = False
        return path_x_list, path_y_list, path_yaw_list, path_yawt1_list, directions, info
    def forward_simulation_one_trailer_modify(self, input, goal, control_list, simulation_freq=10):
        # Fank: use the rs_path control we extract to forward simulation to 
        # check whether suitable this path
        # control_list: clip to [-1,1]
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
        path_x_list, path_y_list, path_yaw_list, path_yawt1_list = [], [], [], []
        directions = []
        controlled_vehicle = tt_envs.OneTrailer(config_dict)
        controlled_vehicle.reset(*input)
        path_x, path_y, path_yaw, path_yawt1 = controlled_vehicle.state
        path_x_list.append(path_x)
        path_y_list.append(path_y)
        path_yaw_list.append(path_yaw)
        path_yawt1_list.append(path_yawt1)
        for action_clipped in control_list:
            if action_clipped[0] > 0:
                directions.append(1)
            else:
                directions.append(-1)
            controlled_vehicle.step(action_clipped, 1 / simulation_freq)
            path_x, path_y, path_yaw, path_yawt1 = controlled_vehicle.state
            path_x_list.append(path_x)
            path_y_list.append(path_y)
            path_yaw_list.append(path_yaw)
            path_yawt1_list.append(path_yawt1)
            
        directions.append(directions[-1])
        final_state = np.array(controlled_vehicle.state)
        # distance_error = np.linalg.norm(goal - final_state)
        distance_error = mixed_norm(goal, final_state)
        # Fank: accept(false means not good)
        #       collision(false means no collision)
        #       jack_knife(false means no jack_knife)
        info = {
            "accept": False,
            "collision": None,
            "jack_knife": None,
        }
        if distance_error > self.config["acceptance_error"]:
            info["accept"] = False
        else:
            info["accept"] = True
        
        if info["accept"]:
            # Fank: check whether collision here
            ind = range(0, len(path_x_list), self.config["collision_check_step"])
            pathx = [path_x_list[k] for k in ind]
            pathy = [path_y_list[k] for k in ind]
            pathyaw = [path_yaw_list[k] for k in ind]
            pathyawt1 = [path_yawt1_list[k] for k in ind]
            if self.is_collision(pathx, pathy, pathyaw, pathyawt1):
                info["collision"] = True
            else:
                # no collision
                info["collision"] = False
        
        return path_x_list, path_y_list, path_yaw_list, path_yawt1_list, directions, info   
    def is_same_grid(self, node1, node2):
        """
        whether the two nodes are on the same grid
        """
        if node1.xind != node2.xind or \
                node1.yind != node2.yind or \
                node1.yawind != node2.yawind:
            return False

        return True
    
    def is_the_start(self, node1, node2):
        """
        whether the two nodes are all start node
        """
        if len(node1.x) == 1 and len(node2.y) == 1:
            return True
        return False
    
    def calc_rs_path_cost_one_trailer_modify(self, rspath) -> float:
        """
        calculate rs path cost
        We don't need yawt1 anymore,
        cause it's been calculated
        Inputs:
        - rspath: path class
        - yawt: the first trailer yaw
        """
        cost = 0.0

        for lr in rspath.lengths:
            if lr >= 0:
                cost += abs(lr)
            else:
                cost += abs(lr) * self.cost["backward_cost"]

        for i in range(len(rspath.lengths) - 1):
            if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
                cost += self.cost["gear_cost"]

        for ctype in rspath.ctypes:
            if ctype != "S":
                cost += self.cost["steer_angle_cost"] * abs(self.vehicle.MAX_STEER)

        nctypes = len(rspath.ctypes)
        ulist = [0.0 for _ in range(nctypes)]

        for i in range(nctypes):
            if rspath.ctypes[i] == "R":
                ulist[i] = -self.vehicle.MAX_STEER
            elif rspath.ctypes[i] == "WB":
                ulist[i] = self.vehicle.MAX_STEER

        for i in range(nctypes - 1):
            cost += self.cost["steer_change_cost"] * abs(ulist[i + 1] - ulist[i])

        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(rspath.yaw, rspath.yawt1)])

        return cost
    
    def calc_rl_path_cost_one_trailer(self, rlpath) -> float:
        cost = 0.0
        action_list = rlpath.rlcontrollist
        for j in range(len(action_list)):
            if action_list[j][0] > 0:
                cost += abs(action_list[j][0])
            else:
                cost += abs(action_list[j][0]) * self.cost["backward_cost"]
            cost += self.cost["steer_angle_cost"] * abs(action_list[j][1])
            if j > 0:
                if action_list[j][0] * action_list[j - 1][0] < 0:
                    cost += self.cost["gear_cost"]
                cost += self.cost["steer_change_cost"] * abs(action_list[j][1] - action_list[j - 1][1])
        
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y)) \
            for x, y in zip(rlpath.yaw[1:], rlpath.yawt1[1:])])
                
        return cost
            
            
        
    
    def calculate_rs_for_heuristic_modify(self, node, ngoal):
        """
        find a non_holonomic rs path for tractor trailer system
        but don't have to be collision-free
        """
        # gyawt1, gyawt2, gyawt3 = ngoal.yawt1[-1], ngoal.yawt2[-1], ngoal.yawt3[-1]
        # 1 / (minimun radius) 
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        paths = self.calc_all_paths_modify(node, ngoal, maxc, step_size=self.step_size)

        if not paths:
            return None

        pq = hyastar.QueuePrior()
        for path in paths:
            pq.put(path, self.calc_rs_path_cost_one_trailer_modify(path))

        count = 0
        while not pq.empty():
            path = pq.get()
            if count == 0:
                path_first = path
            if self.config["plot_rs_path"]:
                plot_rs_path(path, self.ox, self.oy)
                plt.close()
            # this api could be some mistake
            ind = range(0, len(path.x), self.config["collision_check_step"])

            pathx = [path.x[k] for k in ind]
            pathy = [path.y[k] for k in ind]
            pathyaw = [path.yaw[k] for k in ind]
            pathyawt1 = [path.yawt1[k] for k in ind]
            count += 1
            if not self.is_collision(pathx, pathy, pathyaw, pathyawt1):
                return path 
        
        return path_first
    
    def heuristic_RS_one_trailer_modify(self, n_curr, n_goal, plot=False) -> float:
        """
        this is used when calculating heuristics
        
        Inputs:
        - n_curr: curr node
        - n_goal: goal node
        - ox, oy: obstacle (no reso)
        - plot: whether doing visualization
        """
        # check if the start/goal configuration is legal
        if not self.is_index_ok(n_curr, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(n_goal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        
        # get start/goal configuration from the node
        sx = n_curr.x[-1]
        sy = n_curr.y[-1]
        syaw0 = n_curr.yaw[-1]
        # syawt1 = n_curr.yawt1[-1]
        # syawt2 = n_curr.yawt2[-1]
        # syawt3 = n_curr.yawt3[-1]
        
        gx = n_goal.x[-1]
        gy = n_goal.y[-1]
        gyaw0 = n_goal.yaw[-1]
        # gyawt1 = n_goal.yawt1[-1]
        # gyawt2 = n_goal.yawt2[-1]
        # gyawt3 = n_goal.yawt3[-1]
        
        # the same start and goal
        epsilon = 1e-5
        if np.abs(sx - gx) <= epsilon and np.abs(sy - gy) <= epsilon and \
            np.abs(syaw0 - gyaw0) <= epsilon:
            return 0.0
        
        path = self.calculate_rs_for_heuristic_modify(n_curr, n_goal)
        return self.calc_rs_path_cost_one_trailer_modify(path)
    
    def analystic_expantion_modify(self, node, ngoal):
        """
        the returned path contains the start and the end
        which is also admissible(TT configuration and no collision)
        which is different from calculate_rs_for_heuristic
        """
        

        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        # I add a new attribute to this function 
        paths = self.calc_all_paths_modify(node, ngoal, maxc, step_size=self.step_size)

        if not paths:
            return None

        pq = hyastar.QueuePrior()
        for path in paths:
            if not path.valid:
                continue
            pq.put(path, self.calc_rs_path_cost_one_trailer_modify(path))
            # pq.put(path, calc_rs_path_cost_one_trailer(path, yawt1))

        while not pq.empty():
            path = pq.get()
            # check whether collision
            ind = range(0, len(path.x), self.config["collision_check_step"])
            pathx = [path.x[k] for k in ind]
            pathy = [path.y[k] for k in ind]
            pathyaw = [path.yaw[k] for k in ind]
            pathyawt1 = [path.yawt1[k] for k in ind]

            if not self.is_collision(pathx, pathy, pathyaw, pathyawt1):
                return path

        return None
    
    def update_node_with_analystic_expantion_modify(self, n_curr, ngoal):
        """
        find a admissible rs path for one trailer system
        
        Inputs:
        - n_curr: curr node
        - ngoal: goal node
        - P: parameters
        Return:
        - flag: Boolean whether we find a admissible path
        - fpath: a node from n_curr -> ngoal(contains ngoal configuration not n_curr configuration)
        """
        # now the returnd path has a new attribute rscontrollist
        # return the waypoints and control_list all in path
        path = self.analystic_expantion_modify(n_curr, ngoal)  # rs path: n -> ngoal

        if not path:
            return False, None, None, None

        
        fx = path.x[1:]
        fy = path.y[1:]
        fyaw = path.yaw[1:]
        fyawt1 = path.yawt1[1:]
        fd = []
        for d in path.directions[1:]:
            if d >= 0:
                fd.append(1.0)
            else:
                fd.append(-1.0)
        
        fsteer = 0.0
        # fd = path.directions[1:-1]

        fcost = n_curr.cost + self.calc_rs_path_cost_one_trailer_modify(path)
        # fcost = n_curr.cost + calc_rs_path_cost_one_trailer(path, yawt1)
        fpind = self.calc_index(n_curr)

        try:
            #here n_curr.xind might be wrong
            #but doesn't matter
            fpath = hyastar.Node_one_trailer(self.vehicle, ngoal.xind, ngoal.yind, ngoal.yawind, ngoal.direction,
                        fx, fy, fyaw, fyawt1, fd, fsteer, fcost, fpind)
        except:
            return False, None, None, None
        # abandon the first method
        return True, fpath, path.rscontrollist, path
    
    def visualize_hmap(self, hmap):
        """visualize hmap"""
        hmap = np.where(np.isinf(hmap), np.nan, hmap)


        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        hmap_for_vis = np.flip(np.transpose(hmap), 0)
        
        x_min = min(self.ox) - (self.heuristic_reso / 2)
        x_max = max(self.oy) + (self.heuristic_reso / 2)
        y_min = min(self.oy) - (self.heuristic_reso / 2)
        y_max = max(self.oy) + (self.heuristic_reso / 2)
        cax = ax.imshow(hmap_for_vis, cmap="jet", extent=[x_min, x_max, y_min, y_max], aspect='auto')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 添加颜色条
        cbar = fig.colorbar(cax)
        
        plt.plot(self.ox, self.oy, 'sk', markersize=1)

        plt.savefig('hmap_challenge_cases.png')
    
    def extract_path_and_control(self, closed, ngoal, nstart, reverse=False, find_rs_path=True):
        """
        extract the path before rs path
        notice that there will be some unavoidable mistakes
        - find_rs_path: whether we find rs path (always yes)
        """
        rx, ry, ryaw, ryawt1, direc = [], [], [], [], []
        expand_control_list = []
        # TODO: here you made a mistake
        step = self.config["mp_step"]
        nlist = math.ceil(step / self.config["move_step"])
        # cost = 0.0
        node = ngoal
        count = 0
        
        while True:
            #append the current node state configuration
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            ryawt1 += node.yawt1[::-1]
            direc += node.directions[::-1]
            # cost += node.cost

            if self.is_the_start(node, nstart) and self.is_same_grid(node, nstart):
                break
            if find_rs_path:
                if count > 0: #which means this is definitely not rs path
                    for i in range(nlist):
                        expand_control_list.append(np.array([node.directions[-1] * self.step_size, node.steer]))       
            else:
                for i in range(nlist):
                    expand_control_list.append(np.array([node.directions[-1] * self.step_size, node.steer
                                                        ]))
            # tracking parent ind
            node = closed[node.pind]
            count += 1
        if not reverse:
            rx = rx[::-1]
            ry = ry[::-1]
            ryaw = ryaw[::-1]
            ryawt1 = ryawt1[::-1]
            direc = direc[::-1]
            direc[0] = direc[1]
        
        if self.config["plot_final_path"]:
            self.plot_real_path(rx, ry)
            save_dir = './planner_result/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            files = os.listdir(save_dir)
            
            file_index = 0
            file_name = f"hybrid_expand_tree_one_trailer_{file_index}.png"
            while file_name in files:
                file_index += 1
                file_name = f"hybrid_expand_tree_one_trailer_{file_index}.png"
            plt.savefig(os.path.join(save_dir, file_name))
        plt.close()  
         
        path = hyastar.Path_one_trailer(rx, ry, ryaw, ryawt1, direc)
        expand_control_list = expand_control_list[::-1]
        
        return path, expand_control_list
    
    def extract_path(self, closed, ngoal, nstart, reverse=False):
        """
        extract the final path
        closed: closed_set (dictionary: key is node_ind, value is node class)
        ngoal: goal node class
        nstart: start node class
        reverse: whether to reverse or not
        
        returns:
        path class
        """
        rx, ry, ryaw, ryawt1, direc = [], [], [], [], []
        # cost = 0.0
        node = ngoal
        
        while True:
            #append the current node state configuration
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            ryawt1 += node.yawt1[::-1]
            direc += node.directions[::-1]
            # cost += node.cost

            if self.is_the_start(node, nstart) and self.is_same_grid(node, nstart):
                break
            # tracking parent ind
            node = closed[node.pind]
        if not reverse:
            rx = rx[::-1]
            ry = ry[::-1]
            ryaw = ryaw[::-1]
            ryawt1 = ryawt1[::-1]
            direc = direc[::-1]
            direc[0] = direc[1]
            
        if self.config["plot_final_path"]:
            self.plot_real_path(rx, ry)
            save_dir = './planner_result/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            files = os.listdir(save_dir)
            
            file_index = 0
            file_name = f"hybrid_expand_tree_one_trailer_{file_index}.png"
            while file_name in files:
                file_index += 1
                file_name = f"hybrid_expand_tree_one_trailer_{file_index}.png"
            plt.savefig(os.path.join(save_dir, file_name))
        plt.close() 
        path = hyastar.Path_one_trailer(rx, ry, ryaw, ryawt1, direc)

        return path
    
    def calc_hybrid_cost_new(self, n_curr, n_goal):
        """
        A newly implemented function to calculate the heuristic
        it takes the maximun of the non-holonomic heuristic and the holonomic heuristic with 
        obtacles
        Inputs:
        - ox/oy: obstacle(before reso)
        """
        # currently we use path length as our heuristic
        
        
        # heuristic_non_holonomic = self.heuristic_RS_one_trailer(n_curr, n_goal)
        heuristic_non_holonomic = self.heuristic_RS_one_trailer_modify(n_curr, n_goal)
        heuristic_holonomic_obstacles = hyastar.calc_holonomic_heuristic_with_obstacle_value(n_curr, self.hmap, self.ox, self.oy, self.heuristic_reso)
        # heuristic_holonomic_obstacles = hmap[n_curr.xind - P.minx][n_curr.yind - P.miny]
        cost = n_curr.cost + \
            self.cost["h_cost"] * max(heuristic_holonomic_obstacles, heuristic_non_holonomic)

        return cost
    
    def calc_hybrid_cost_simplify(self, n_curr, n_goal, cost):
        heuristic_non_holonomic = cost
        heuristic_holonomic_obstacles = self.hmap[n_curr.xind - self.P.minx][n_curr.yind - self.P.miny]
        cost = n_curr.cost + \
             self.cost["h_cost"] * max(heuristic_non_holonomic, heuristic_holonomic_obstacles)

        return cost
    
    def calc_euclidean_distance(self, n_curr, n_goal):
        n_curr_x = n_curr.x[-1]
        n_curr_y = n_curr.y[-1]
        n_curr_yaw = n_curr.yaw[-1]
        n_curr_yawt1 = n_curr.yawt1[-1]
        curr_state = np.array([n_curr_x, n_curr_y, n_curr_yaw, n_curr_yawt1, 0, 0], dtype=np.float32)
        n_goal_x = n_goal.x[-1]
        n_goal_y = n_goal.y[-1]
        n_goal_yaw = n_goal.yaw[-1]
        n_goal_yawt1 = n_goal.yawt1[-1]
        goal_state = np.array([n_goal_x, n_goal_y, n_goal_yaw, n_goal_yawt1, 0, 0], dtype=np.float32)
        return np.linalg.norm(curr_state - goal_state)
        
    def calc_hybrid_cost_new_critic(self, n_curr, n_goal):
        heuristic_nn = self.heuristic_nn_one_trailer(n_curr, n_goal)
        heuristic_holonomic_obstacles = hyastar.calc_holonomic_heuristic_with_obstacle_value(n_curr, self.hmap, self.ox, self.oy, self.heuristic_reso)
        # heuristic_holonomic_obstacles = hmap[n_curr.xind - P.minx][n_curr.yind - P.miny]
        cost = n_curr.cost + \
            self.cost["h_cost"] * max(heuristic_holonomic_obstacles, heuristic_nn)

        return cost
    
    def heuristic_nn_one_trailer(self, n_curr, n_goal):
        n_curr_x = n_curr.x[-1]
        n_curr_y = n_curr.y[-1]
        n_curr_yaw = n_curr.yaw[-1]
        n_curr_yawt1 = n_curr.yawt1[-1]
        curr_state = np.array([n_curr_x, n_curr_y, n_curr_yaw, n_curr_yawt1, 0, 0], dtype=np.float32)
        n_goal_x = n_goal.x[-1]
        n_goal_y = n_goal.y[-1]
        n_goal_yaw = n_goal.yaw[-1]
        n_goal_yawt1 = n_goal.yawt1[-1]
        goal_state = np.array([n_goal_x, n_goal_y, n_goal_yaw, n_goal_yawt1, 0, 0], dtype=np.float32)
        o = np.concatenate([curr_state, curr_state, goal_state])
        a = np.array([0.0, 0.0], dtype=np.float32)
        with torch.no_grad():
            q1 = self.agent.ac.q1(torch.as_tensor(o).unsqueeze(0).to(self.device), torch.as_tensor(a).unsqueeze(0).to(self.device))
            q2 = self.agent.ac.q2(torch.as_tensor(o).unsqueeze(0).to(self.device), torch.as_tensor(a).unsqueeze(0).to(self.device))
            critic_value = ((q1 + q2) / 2).item()
         
        return critic_value
    
    def rl_gear(self, n_curr, n_goal, max_step=60, terminated=0.5):
        # TODO: remains to be fixed a lot of things
        rl_path = RlPath()
        # use rl agent to guide our search
        n_curr_x = n_curr.x[-1]
        n_curr_y = n_curr.y[-1]
        n_curr_yaw = n_curr.yaw[-1]
        n_curr_yawt1 = n_curr.yawt1[-1]
        n_goal_x = n_goal.x[-1]
        n_goal_y = n_goal.y[-1]
        n_goal_yaw = n_goal.yaw[-1]
        n_goal_yawt1 = n_goal.yawt1[-1]
        
        # Apply Coordinate rotation: global -> local
        n_goal_x_local = np.cos(n_curr_yaw) * (n_goal_x - n_curr_x) + np.sin(n_curr_yaw) * (n_goal_y - n_curr_y)
        n_goal_y_local = -np.sin(n_curr_yaw) * (n_goal_x - n_curr_x) + np.cos(n_curr_yaw) * (n_goal_y - n_curr_y)
        start_list = [[0.0, 0.0, 0.0, n_curr_yawt1 - n_curr_yaw, 0.0, 0.0]]
        # test the "start from un-equili configuration" generalization
        self.agent.test_env.unwrapped.update_start_list(start_list)
        goal_list = [[n_goal_x_local, n_goal_y_local, n_goal_yaw - n_curr_yaw, n_goal_yawt1 - n_curr_yaw, 0.0, 0.0]]
        # update the test_env to this goal
        self.agent.test_env.unwrapped.update_goal_list(goal_list)
        # give to the reaching env to simulate path
        o, info = self.agent.test_env.reset()
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        # here we take out the truncated to test "go-further" generalization
        while not(terminated):
            # Take deterministic actions at test time 
            a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), True)
            o, r, terminated, truncated, info = self.agent.test_env.step(a)
            ep_ret += r
            ep_len += 1
            if (ep_len >= max_step):
                break
        # visualize the result
        if info['is_success'] == True:
            find_feasible = True
        else:
            find_feasible = False
        
        path_x = []
        path_y = []
        path_yaw = []
        path_yawt1 = []
        # local -> global
        for state in self.agent.test_env.unwrapped.state_list:
            x_global = np.cos(n_curr_yaw) * state[0] - np.sin(n_curr_yaw) * state[1] + n_curr_x
            y_global = np.sin(n_curr_yaw) * state[0] + np.cos(n_curr_yaw) * state[1] + n_curr_y
            yaw_global = state[2] + n_curr_yaw
            yawt1_global = state[3] + n_curr_yaw
            path_x.append(x_global)
            path_y.append(y_global)
            path_yaw.append(yaw_global)
            path_yawt1.append(yawt1_global)
        rl_path.x = path_x
        rl_path.y = path_y
        rl_path.yaw = path_yaw
        rl_path.yawt1 = path_yawt1
        rl_path.rlcontrollist = action_recover_to_planner(self.agent.test_env.unwrapped.action_list)
        rl_path.info = {
            "is_success": info["is_success"],
            "crashed": info["crashed"],
            "jack_knife": info["jack_knife"],
            "final_node": None,
        }
        
        # check collision from the real obstacle setting
        ind = range(0, len(rl_path.x), self.config["collision_check_step"])
        
        pathx = [rl_path.x[k] for k in ind]
        pathy = [rl_path.y[k] for k in ind]
        pathyaw = [rl_path.yaw[k] for k in ind]
        pathyawt1 = [rl_path.yawt1[k] for k in ind]
        if self.is_collision(pathx, pathy, pathyaw, pathyawt1):
            rl_path.info["crashed"] = True
            rl_path.info["is_success"] = False
            find_feasible = False
        cost = self.calc_rl_path_cost_one_trailer(rl_path)
        rl_path.rlcost = cost
        if find_feasible:
            xind = round(path_x[-1] / self.xyreso)
            yind = round(path_y[-1] / self.xyreso)
            yawind = round(path_yaw[-1] / self.yawreso)
            fd = []
            for action in rl_path.rlcontrollist:
                if action[0] > 0:
                    fd.append(1)
                else:
                    fd.append(-1)
            fx = path_x[1:]
            fy = path_y[1:]
            fyaw = path_yaw[1:]
            fyawt1 = path_yawt1[1:]
            fsteer = 0.0
            fcost = n_curr.cost + cost
            fpind = self.calc_index(n_curr)
            direction = fd[-1]
            final_node = hyastar.Node_one_trailer(self.vehicle, xind, yind, yawind, direction,
                                                  fx, fy, fyaw, fyawt1, fd, fsteer, fcost, fpind)
            rl_path.info["final_node"] = final_node
        # test rl path animation
        # self.agent.test_env.unwrapped.run_simulation()
        return find_feasible, rl_path
    
    def plan_version1(self, start:np.ndarray, goal:np.ndarray, get_control_sequence:bool, verbose=False, *args, **kwargs):
        """
        Main Planning Algorithm for 1-tt systems
        An older version (already been used for training)
        :param start: starting point (np_array)
        :param goal: goal point (np_array)
        - path: all the six-dim state along the way (using extract function)
        - rs_path: contains the rspath and rspath control list
        - control list: rspath control list + expand control list
        """
        
        self.sx, self.sy, self.syaw, self.syawt1 = start
        self.gx, self.gy, self.gyaw, self.gyawt1 = goal
        self.syaw, self.syawt1 = self.pi_2_pi(self.syaw), self.pi_2_pi(self.syawt1)
        self.gyaw, self.gyawt1 = self.pi_2_pi(self.gyaw), self.pi_2_pi(self.gyawt1)
        self.sxr, self.syr = round(self.sx / self.xyreso), round(self.sy / self.xyreso)
        self.gxr, self.gyr = round(self.gx / self.xyreso), round(self.gy / self.xyreso)
        self.syawr = round(self.syaw / self.yawreso)
        self.gyawr = round(self.gyaw / self.yawreso)
        
        # TODO: change to rl
        if self.heuristic_type == "nn":
            self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with open("configs/agents/sac_astar.yaml", 'r') as file:
                config_algo = yaml.safe_load(file)
            
            env_name = config_algo['env_name']
            seed = config_algo['seed']
            exp_name = env_name + '_' + config_algo['algo_name'] + '_' + str(seed)
            logger_kwargs = {
                'output_dir': config_algo['logging_dir'] + exp_name,
                'output_fname': config_algo['output_fname'],
                'exp_name': exp_name,
            }
            if config_algo['activation'] == 'ReLU':
                activation_fn = nn.ReLU
            elif config_algo['activation'] == 'Tanh':
                activation_fn = nn.Tanh
            else:
                raise ValueError(f"Unsupported activation function: {config_algo['activation']}")
            ac_kwargs = {
                "hidden_sizes": tuple(config_algo['hidden_sizes']),
                "activation": activation_fn
            }
            with open("configs/envs/reaching_v0_eval.yaml", 'r') as file:
                config = yaml.safe_load(file)
            
            self.agent = agents.SAC_ASTAR(env_fn=gym_reaching_tt_env_fn,
                        algo=config_algo['algo_name'],
                        ac_kwargs=ac_kwargs,
                        seed=seed,
                        steps_per_epoch=config_algo['sac_steps_per_epoch'],
                        epochs=config_algo['sac_epochs'],
                        replay_size=config_algo['replay_size'],
                        gamma=config_algo['gamma'],
                        polyak=config_algo['polyak'],
                        lr=config_algo['lr'],
                        alpha=config_algo['alpha'],
                        batch_size=config_algo['batch_size'],
                        start_steps=config_algo['start_steps'],
                        update_after=config_algo['update_after'],
                        update_every=config_algo['update_every'],
                        # missing max_ep_len
                        logger_kwargs=logger_kwargs, 
                        save_freq=config_algo['save_freq'],
                        num_test_episodes=config_algo['num_test_episodes'],
                        log_dir=config_algo['log_dir'],
                        whether_her=config_algo['whether_her'],
                        use_automatic_entropy_tuning=config_algo['use_auto'],
                        env_name=config_algo['env_name'],
                        pretrained=config_algo['pretrained'],
                        pretrained_itr=config_algo['pretrained_itr'],
                        pretrained_dir=config_algo['pretrained_dir'],
                        whether_astar=config_algo['whether_astar'],
                        config=config)
            self.agent.load('runs_rl/reaching-v0_sac_astar_50_20240112_220524/model_6999999.pth')
        # put the class in this file
        nstart = hyastar.Node_one_trailer(self.vehicle, self.sxr, self.syr, self.syawr, 1, \
            [self.sx], [self.sy], [self.pi_2_pi(self.syaw)], [self.pi_2_pi(self.syawt1)], [1], 0.0, 0.0, -1)
        ngoal = hyastar.Node_one_trailer(self.vehicle, self.gxr, self.gyr, self.gyawr, 1, \
            [self.gx], [self.gy], [self.pi_2_pi(self.gyaw)], [self.pi_2_pi(self.gyawt1)], [1], 0.0, 0.0, -1)
        if not self.is_index_ok(nstart, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(ngoal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        # TODO: change the api (seems this one is better)
        if self.obs:
            self.hmap = hyastar.calc_holonomic_heuristic_with_obstacle(ngoal, self.P.ox, self.P.oy, self.heuristic_reso, self.heuristic_rr)
        if self.config["plot_heuristic_nonholonomic"]:
            self.visualize_hmap(self.hmap)
        
        
        steer_set, direc_set = self.calc_motion_set()
        # change the way to calc_index
        open_set, closed_set = {self.calc_index(nstart): nstart}, {}
        # non_h = calc_non_holonomic_heuritstic(nstart, ngoal, gyawt1, gyawt2, gyawt3, fP)
        
        # reset qp for next using
        self.qp.reset()
        # the main change here will be the heuristic value
        if self.heuristic_type == "traditional":
            self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_new(nstart, ngoal))
        else:
            # here you need to change this heuristic
            # not using
            self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_new_critic(nstart, ngoal))
        # an indicator whether find the rs path at last
        find_rs_path = False
        # the loop number for analystic expansion
        count = 0
        # Main Loop
        while True:
            # I will try not to use this
            # may need to modify when there's obstacle
            if not open_set or self.qp.empty():
                # consider this function
                print("failed finding a feasible path")
                self.extract_failed_path(closed_set, nstart)
                return None, None, None
            count += 1
            ind = self.qp.get()
            n_curr = open_set[ind]
            closed_set[ind] = n_curr
            open_set.pop(ind)
            
            # key and the most tricky part of the algorithm
            update, fpath, rs_control_list, rs_path = self.update_node_with_analystic_expantion_modify(n_curr, ngoal)

            if update:
                fnode = fpath
                find_rs_path = True
                if self.config["plot_expand_tree"]:
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    plot_rs_path(rs_path, self.ox, self.oy)
                    # plt.close()
                if verbose:
                    print("final analystic expantion node number:", count)
                break
            # add if the loop's too much
            if count > self.max_iter:
                print("waste a long time to find")
                return None, None, None
            for i in range(len(steer_set)):
                
                node = self.calc_next_node(n_curr, ind, steer_set[i], direc_set[i])
                if not node:
                    continue
                if not self.is_index_ok(node, self.config["collision_check_step"]):
                    continue
                node_ind = self.calc_index(node)
                if node_ind in closed_set:
                    continue
                if node_ind not in open_set:
                    open_set[node_ind] = node
                    if self.heuristic_type == "traditional":
                        self.qp.put(node_ind, self.calc_hybrid_cost_new(node, ngoal))
                    else:
                        self.qp.put(node_ind, self.calc_hybrid_cost_new_critic(node, ngoal))
                else:
                    if open_set[node_ind].cost > node.cost:
                        open_set[node_ind] = node
                        if self.qp_type == "heapdict":  
                            # if using heapdict, here you can modify the value  
                            if self.heuristic_type == "traditional":
                                self.qp.queue[node_ind] = self.calc_hybrid_cost_new(node, ngoal)
                            else:
                                self.qp.queue[node_ind] = self.calc_hybrid_cost_new_critic(node, ngoal)             
        if verbose:
            print("final expand node: ", len(open_set) + len(closed_set))
        
        if get_control_sequence:
            path, expand_control_list = self.extract_path_and_control(closed_set, fnode, nstart,find_rs_path=find_rs_path)
            if find_rs_path:
                all_control_list = expand_control_list + rs_control_list
            else:
                rs_path = None
                all_control_list = all_control_list
            return path, all_control_list, rs_path
        else:
            if find_rs_path: 
                return self.extract_path(closed_set, fnode, nstart), None, rs_path
            else:
                return self.extract_path(closed_set, fnode, nstart), None, None
    
    
    def rs_gear(self, node, ngoal):
        # Fank: put all rs related tech here
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        # I add a new attribute to this function 
        # Using a simplified version of calc_all_paths
        paths = self.calc_all_paths_simplified(node, ngoal, maxc)
        
        
        find_feasible = False
        if not paths:
            return find_feasible, None
        pq = hyastar.QueuePrior()
        
        for path in paths:
            if path.info["jack_knife"] == False:
                find_feasible = True
                return find_feasible, path
            pq.put(path, path.rscost)
        #TODO: may have to adjust
        while not pq.empty():
            path = pq.get()
            find_feasible = False
            return find_feasible, path
    
    
    def plan_version2(self, start:np.ndarray, goal:np.ndarray, get_control_sequence:bool, verbose=False, *args, **kwargs):
        """
        Main Planning Algorithm for 1-tt systems
        More advanced version
        :param start: starting point (np_array)
        :param goal: goal point (np_array)
        - path: all the six-dim state along the way (using extract function)
        - rs_path: contains the rspath and rspath control list
        - control list: rspath control list + expand control list
        """
        
        self.sx, self.sy, self.syaw, self.syawt1 = start
        self.gx, self.gy, self.gyaw, self.gyawt1 = goal
        self.syaw, self.syawt1 = self.pi_2_pi(self.syaw), self.pi_2_pi(self.syawt1)
        self.gyaw, self.gyawt1 = self.pi_2_pi(self.gyaw), self.pi_2_pi(self.gyawt1)
        self.sxr, self.syr = round(self.sx / self.xyreso), round(self.sy / self.xyreso)
        self.gxr, self.gyr = round(self.gx / self.xyreso), round(self.gy / self.xyreso)
        self.syawr = round(self.syaw / self.yawreso)
        self.gyawr = round(self.gyaw / self.yawreso)
        
        
        # put the class in this file
        nstart = hyastar.Node_one_trailer(self.vehicle, self.sxr, self.syr, self.syawr, 1, \
            [self.sx], [self.sy], [self.pi_2_pi(self.syaw)], [self.pi_2_pi(self.syawt1)], [1], 0.0, 0.0, -1)
        ngoal = hyastar.Node_one_trailer(self.vehicle, self.gxr, self.gyr, self.gyawr, 1, \
            [self.gx], [self.gy], [self.pi_2_pi(self.gyaw)], [self.pi_2_pi(self.gyawt1)], [1], 0.0, 0.0, -1)
        # check whether valid
        if not self.is_index_ok(nstart, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(ngoal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        # TODO: change the api (seems this one is better)
        if self.obs:
            self.hmap = hyastar.calc_holonomic_heuristic_with_obstacle(ngoal, self.P.ox, self.P.oy, self.heuristic_reso, self.heuristic_rr)
        if self.config["plot_heuristic_nonholonomic"]:
            self.visualize_hmap(self.hmap)
        
        
        steer_set, direc_set = self.calc_motion_set()
        # Initialize open_set and closed_set
        open_set, closed_set = {self.calc_index(nstart): nstart}, {}
        
        # reset qp for next using
        self.qp.reset()
        # an indicator whether find the rs path at last
        find_rs_path = False
        # the loop number for analystic expansion
        count = 0
        # update parameter
        update = False
        
        # the main change here will be the heuristic value
        if self.heuristic_type == "traditional":
            find_feasible, path = self.rs_gear(nstart, ngoal)
            # if find feasible, then go to extract
            # else calculate heuristic
            if find_feasible:
                fnode = path.info["final_node"]
                find_rs_path = True
                update = find_feasible
                rs_path = path
                rs_control_list = path.rscontrollist
                if self.config["plot_expand_tree"]:
                    plot_rs_path(rs_path, self.ox, self.oy)
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    # plt.close()
                if verbose:
                    print("find path at first time")
                closed_set[self.calc_index(nstart)] = nstart
            else:
                self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_simplify(nstart, ngoal, path.rscost))
        else:
            # TODO wait for RL to guide search
            t1 = time.time()
            find_feasible, path = self.rl_gear(nstart, ngoal)
            t2 = time.time()
            print("time for rl simulation:", t2 - t1)
            if find_feasible:
                fnode = path.info["final_node"]
                find_rl_path = True
                update = find_feasible
                rl_path = path
                rl_control_list = path.rlcontrollist
                if self.config["plot_expand_tree"]:
                    plot_rs_path(rl_path, self.ox, self.oy)
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    # plt.close()
                if verbose:
                    print("find path at first time")
                closed_set[self.calc_index(nstart)] = nstart
            else:
                cost_qp = self.calc_euclidean_distance(nstart, ngoal)
                self.qp.put(self.calc_index(nstart), cost_qp)
                # self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_simplify(nstart, ngoal, path.rlcost))
        
        # Main Loop
        while True:
            if update:
                # use the flag update to break the main loop
                break
            if not open_set or self.qp.empty():
                print("failed finding a feasible path")
                self.extract_failed_path(closed_set, nstart)
                return None, None, None
            count += 1
            # add if the loop's too much
            if count > self.max_iter:
                print("waste a long time to find")
                return None, None, None
            ind = self.qp.get()
            n_curr = open_set[ind]
            closed_set[ind] = n_curr
            open_set.pop(ind)
            
            # expand tree using motion primitive
            for i in range(len(steer_set)):
                node = self.calc_next_node(n_curr, ind, steer_set[i], direc_set[i])
                if not node:
                    # encounter jack_knife
                    continue
                if not self.is_index_ok(node, self.config["collision_check_step"]):
                    # check go outside or collision
                    continue
                node_ind = self.calc_index(node)
                if node_ind in closed_set:
                    # we will not calculate twice 
                    # Note that this can be a limitation
                    continue
                if node_ind not in open_set:
                    open_set[node_ind] = node
                    if self.heuristic_type == "traditional":
                        find_feasible, path = self.rs_gear(node, ngoal)
                        if find_feasible:
                            fnode = path.info["final_node"]
                            find_rs_path = True
                            update = find_feasible
                            rs_path = path
                            rs_control_list = path.rscontrollist
                            if self.config["plot_expand_tree"]:
                                plot_rs_path(rs_path, self.ox, self.oy)
                                self.plot_expand_tree(start, goal, closed_set, open_set)
                                # plt.close()
                            if verbose:
                                print("final expansion node number:", count)
                            # Here you need to add node to closed set
                            closed_set[node_ind] = node
                            # break the inner expand_tree loop
                            break
                        else:
                            self.qp.put(node_ind, self.calc_hybrid_cost_simplify(node, ngoal, path.rscost))
                    else:
                        # wait for RL to guide search
                        t1 = time.time()
                        find_feasible, path = self.rl_gear(node, ngoal)
                        t2 = time.time()
                        print("time for rl simulation:", t2 - t1)
                        if find_feasible:
                            fnode = path.info["final_node"]
                            find_rl_path = True
                            update = find_feasible
                            rl_path = path
                            rl_control_list = path.rlcontrollist
                            if self.config["plot_expand_tree"]:
                                plot_rs_path(rl_path, self.ox, self.oy)
                                self.plot_expand_tree(start, goal, closed_set, open_set)
                                # plt.close()
                            if verbose:
                                print("final expansion node number:", count)
                            closed_set[node_ind] = node
                            break
                        else:
                            cost_qp = self.calc_euclidean_distance(node, ngoal)
                            self.qp.put(node_ind, cost_qp)
                            # self.qp.put(node_ind, self.calc_hybrid_cost_simplify(nstart, ngoal, path.rlcost))
                else:
                    if open_set[node_ind].cost > node.cost:
                        open_set[node_ind] = node
                        if self.qp_type == "heapdict":  
                            # if using heapdict, here you can modify the value 
                            if self.heuristic_type == "traditional":
                                find_feasible, path = self.rs_gear(node, ngoal)
                                if find_feasible:
                                    fnode = path.info["final_node"]
                                    find_rs_path = True
                                    update = find_feasible
                                    rs_path = path
                                    rs_control_list = path.rscontrollist
                                    if self.config["plot_expand_tree"]:
                                        plot_rs_path(rs_path, self.ox, self.oy)
                                        self.plot_expand_tree(start, goal, closed_set, open_set)
                                        # plt.close()
                                    if verbose:
                                        print("final expansion node number:", count)
                                    closed_set[node_ind] = node
                                    break
                                else:
                                    self.qp.queue[node_ind] = self.calc_hybrid_cost_simplify(node, ngoal, path.rscost)
                            else:
                                t1 = time.time()
                                find_feasible, path = self.rl_gear(node, ngoal)
                                t2 = time.time()
                                print("time for rl simulation:", t2 - t1)
                                if find_feasible:
                                    fnode = path.info["final_node"]
                                    find_rl_path = True
                                    update = find_feasible
                                    rl_path = path
                                    rl_control_list = path.rlcontrollist
                                    if self.config["plot_expand_tree"]:
                                        plot_rs_path(rl_path, self.ox, self.oy)
                                        self.plot_expand_tree(start, goal, closed_set, open_set)
                                        # plt.close()
                                    if verbose:
                                        print("final expansion node number:", count)
                                    closed_set[node_ind] = node
                                else:
                                    cost_qp = self.calc_euclidean_distance(node, ngoal)
                                    self.qp.queue[node_ind] = cost_qp
                                    # self.qp.queue[node_ind] = self.calc_hybrid_cost_simplify(node, ngoal, path.rlcost)          
        if verbose:
            print("final expand node: ", len(open_set) + len(closed_set) - 1)
        
        if get_control_sequence:
            path, expand_control_list = self.extract_path_and_control(closed_set, fnode, nstart,find_rs_path=find_rs_path)
            if self.heuristic_type == "rl":
                if find_rl_path:
                    all_control_list = expand_control_list + rl_control_list
                else:
                    rl_path = None
                    all_control_list = expand_control_list
                return path, all_control_list, rl_path
            else:
                if find_rs_path:
                    all_control_list = expand_control_list + rs_control_list
                else:
                    rs_path = None
                    all_control_list = expand_control_list
                return path, all_control_list, rs_path
        else:
            if self.heuristic_type == "rl":
                if find_rl_path: 
                    return self.extract_path(closed_set, fnode, nstart), None, rl_path
                else:
                    return self.extract_path(closed_set, fnode, nstart), None, None
            else:
                if find_rs_path: 
                    return self.extract_path(closed_set, fnode, nstart), None, rs_path
                else:
                    return self.extract_path(closed_set, fnode, nstart), None, None
    
    
    def plot_expand_tree(self, start, goal, closed_set, open_set):
        plt.axis("equal")
        ax = plt.gca() 
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
        for key, value in open_set.items():
            self.plot_node(value, color='gray')
        for key, value in closed_set.items():
            self.plot_node(value, color='red')
        self.vehicle.reset(*goal)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        self.vehicle.reset(*start)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
    
    def plot_node(self, node, color):
        xlist = node.x
        ylist = node.y
        plt.plot(xlist, ylist, color=color, markersize=1)
    
    def plot_real_path(self, rx, ry):
        plt.plot(rx, ry, color="blue", markersize=1)
    
    
    def visualize_planning(self, start, goal, path, 
                           gif=True, save_dir='./planner_result/gif'):
        """visuliaze the planning result
        : param path: a path class
        : start & goal: cast as np.ndarray
        """
        print("Start Visulizate the Result")
        x = path.x
        y = path.y
        yaw = path.yaw
        yawt1 = path.yawt1
        direction = path.direction
        
        if gif:
            fig, ax = plt.subplots()

            def update(num):
                ax.clear()
                plt.axis("equal")
                k = num
                # plot env (obstacle)
                plt.plot(self.ox, self.oy, "sk", markersize=1)
                self.vehicle.reset(*start)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
                
                # plot the planning path
                plt.plot(x, y, linewidth=1.5, color='r')
                self.vehicle.reset(*goal)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
                self.vehicle.reset(x[k], y[k], yaw[k], yawt1[k])
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0

                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')
                plt.axis("equal")

            ani = FuncAnimation(fig, update, frames=len(x), repeat=True)

            # Save the animation
            writer = PillowWriter(fps=20)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                
            # base_path = "./HybridAstarPlanner/gif/path_animation"
            base_path = os.path.join(save_dir, 'hybrid_astar_path_plan_one_tractor_trailer')
            extension = ".gif"
            
            all_files = os.listdir(save_dir)
            matched_files = [re.match(r'hybrid_astar_path_plan_one_tractor_trailer(\d+)\.gif', f) for f in all_files]
            numbers = [int(match.group(1)) for match in matched_files if match]
            
            if numbers:
                save_index = max(numbers) + 1
            else:
                save_index = 1
            ani.save(base_path + str(save_index) + extension, writer=writer)
            print("Done Plotting")
            
        else:
            # this is when your device has display setting
            fig, ax = plt.subplots()
            # this is when your device has display setting
            plt.pause(5)

            for k in range(len(x)):
                plt.cla()
                plt.axis("equal")
                # plot env (obstacle)
                plt.plot(self.ox, self.oy, "sk", markersize=1)
                # plot the planning path
                plt.plot(x, y, linewidth=1.5, color='r')
                self.vehicle.reset(*start)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
                self.vehicle.reset(*goal)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')

                # calculate every time step
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    # different from a single car
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0
                # draw goal model
                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')
                plt.pause(0.0001)

            plt.show()

    def extract_failed_path(self, closed, nstart):
        
        for value in closed.values():
            plt.plot(value.x, value.y,'.', color='grey', markersize=1)
        plt.plot(nstart.x, nstart.y, 'o', color='r', markersize=3)    
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
        # plt.legend()
        plt.axis("equal")
        if not os.path.exists("planner_result/failed_trajectory_one_trailer"):
            os.makedirs("planner_result/failed_trajectory_one_trailer")
            
        base_path = "./planner_result/failed_trajectory_one_trailer"
        extension = ".png"
            
        all_files = os.listdir("./planner_result/failed_trajectory_one_trailer")
        matched_files = [re.match(r'explored(\d+)\.png', f) for f in all_files]
        numbers = [int(match.group(1)) for match in matched_files if match]
        
        if numbers:
            save_index = max(numbers) + 1
        else:
            save_index = 0
        plt.savefig(base_path + "/explored" + str(save_index) + extension)
        plt.close()
        # plt.savefig("HybridAstarPlanner/trajectory/explored.png")

class TwoTractorTrailerHybridAstarPlanner(hyastar.BasicHybridAstarPlanner):
    @classmethod
    def default_config(cls) -> dict:
        return {
            "verbose": False, 
            "heuristic_type": "traditional",
            "vehicle_type": "two_trailer",
            "act_limit": 1, 
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
            "xy_reso": 2.0,
            "yaw_reso": np.deg2rad(15.0),
            "qp_type": "heapdict",
            "max_iter": 1000, # outer loop number
            "step_size": 0.2, # rs path sample step size
            "move_step": 0.2, # expand node tree
            "n_steer": 20, #how many parts to divide in motion primitives
            "mp_step": 40.0, # previous 2.0 * reso, how many moves in mp step
            "is_save_animation": True,
            "is_save_expand_tree": True,
            "visualize_mode": True,
            "dt": 0.1,
            "heuristic_reso": 2.0,
            "heuristic_rr": 1.0, # 0.5 * heuristic_reso
            "whether_obs": True,
            "safe_d": 0.0,
            "extend_area": 0.0,
            "collision_check_step": 10,
            "goal_yaw_error": np.deg2rad(3.0),
            "cost_configuration":
                {
                    "scissors_cost": 200.0,
                    "gear_cost": 100.0,
                    "backward_cost": 5.0,
                    "steer_change_cost": 5.0,
                    "h_cost": 10.0,
                    "steer_angle_cost": 1.0,
                },   
            
            
            "plot_heuristic_nonholonomic": False,
            "plot_rs_path": True,
            "plot_expand_tree": True,
            "plot_final_path": True,
            "range_steer_set": 8, #need to set the same as n_steer
            "acceptance_error": 0.5,
        }
    
    def configure(self, config: Optional[dict]):
        if config:
            self.config.update(config)
        self.vehicle = tt_envs.TwoTrailer(self.config["controlled_vehicle_config"])
        self.max_iter = self.config["max_iter"] 
        self.xyreso = self.config["xy_reso"]
        self.yawreso = self.config["yaw_reso"]
        self.qp_type = self.config["qp_type"] 
        self.safe_d = self.config["safe_d"]
        self.extend_area = self.config["extend_area"]
        self.obs = self.config['whether_obs']
        self.cost = self.config['cost_configuration']
        self.step_size = self.config["step_size"]
        self.n_steer = self.config["n_steer"]
        self.heuristic_type = self.config['heuristic_type']
        if self.obs:
            self.heuristic_reso = self.config["heuristic_reso"]
            self.heuristic_rr = self.config["heuristic_rr"]
        if self.qp_type == "heapdict":
            self.qp = hyastar.NewQueuePrior()
        else:
            self.qp = hyastar.QueuePrior()
    
    
    def __init__(self, ox, oy, config: Optional[dict] = None) -> None:
        self.config = self.default_config()
        self.configure(config)
        
        super().__init__(ox, oy)
    
    def calc_parameters(self):
        """calculate parameters of the planning problem
        return: para class implemented in hybrid_astar.py
        """
        minxm = min(self.ox) - self.extend_area
        minym = min(self.oy) - self.extend_area
        maxxm = max(self.ox) + self.extend_area
        maxym = max(self.oy) + self.extend_area

        self.ox.append(minxm)
        self.oy.append(minym)
        self.ox.append(maxxm)
        self.oy.append(maxym)

        minx = round(minxm / self.xyreso)
        miny = round(minym / self.xyreso)
        maxx = round(maxxm / self.xyreso)
        maxy = round(maxym / self.xyreso)

        xw, yw = maxx - minx + 1, maxy - miny + 1

        minyaw = round(-self.vehicle.PI / self.yawreso)
        maxyaw = round(self.vehicle.PI / self.yawreso)
        yaww = maxyaw - minyaw + 1

        minyawt1, maxyawt1, yawt1w = minyaw, maxyaw, yaww
        minyawt2, maxyawt2, yawt2w = minyaw, maxyaw, yaww
        

        P = hyastar.Para_two_trailer(minx, miny, minyaw, minyawt1, minyawt2, maxx, maxy, maxyaw,
                maxyawt1, maxyawt2, xw, yw, yaww, yawt1w, yawt2w, self.xyreso, self.yawreso, self.ox, self.oy, self.kdtree)

        return P
    
    
    def calc_motion_set(self):
        """
        this is much alike motion primitives
        """
        s = [i for i in np.arange(self.vehicle.MAX_STEER / self.n_steer,
                                self.config["range_steer_set"] * self.vehicle.MAX_STEER / self.n_steer, self.vehicle.MAX_STEER / self.n_steer)]

        steer = [0.0] + s + [-i for i in s]
        direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
        steer = steer + steer

        return steer, direc
    
    
    def heuristic_RS_two_trailer_modify(self, n_curr, n_goal, plot=False) -> float:
        """
        this is used when calculating heuristics
        
        Inputs:
        - n_curr: curr node
        - n_goal: goal node
        - ox, oy: obstacle (no reso)
        - plot: whether doing visualization
        """
        # check if the start/goal configuration is legal
        if not self.is_index_ok(n_curr, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(n_goal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        
        # get start/goal configuration from the node
        sx = n_curr.x[-1]
        sy = n_curr.y[-1]
        syaw0 = n_curr.yaw[-1]
        # syawt1 = n_curr.yawt1[-1]
        # syawt2 = n_curr.yawt2[-1]
        # syawt3 = n_curr.yawt3[-1]
        
        gx = n_goal.x[-1]
        gy = n_goal.y[-1]
        gyaw0 = n_goal.yaw[-1]
        # gyawt1 = n_goal.yawt1[-1]
        # gyawt2 = n_goal.yawt2[-1]
        # gyawt3 = n_goal.yawt3[-1]
        
        # the same start and goal
        epsilon = 1e-5
        if np.abs(sx - gx) <= epsilon and np.abs(sy - gy) <= epsilon and \
            np.abs(syaw0 - gyaw0) <= epsilon:
            return 0.0
        
        path = self.calculate_rs_for_heuristic_modify(n_curr, n_goal)
        return self.calc_rs_path_cost_two_trailer_modify(path)
    
    def calc_rs_path_cost_two_trailer_modify(self, rspath) -> float:
        """
        A newly version that rspath contains all the information
        this function calculate rs path cost based on rspath and yawt
        
        Inputs:
        - rspath: path class
        - yawt: the first trailer yaw
        """
        cost = 0.0

        for lr in rspath.lengths:
            if lr >= 0:
                cost += abs(lr)
            else:
                cost += abs(lr) * self.cost["backward_cost"]

        for i in range(len(rspath.lengths) - 1):
            if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
                cost += self.cost["gear_cost"]

        for ctype in rspath.ctypes:
            if ctype != "S":
                cost += self.cost["steer_angle_cost"] * abs(self.vehicle.MAX_STEER)

        nctypes = len(rspath.ctypes)
        ulist = [0.0 for _ in range(nctypes)]

        for i in range(nctypes):
            if rspath.ctypes[i] == "R":
                ulist[i] = -self.vehicle.MAX_STEER
            elif rspath.ctypes[i] == "WB":
                ulist[i] = self.vehicle.MAX_STEER

        for i in range(nctypes - 1):
            cost += self.cost["steer_change_cost"] * abs(ulist[i + 1] - ulist[i])

        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(rspath.yaw, rspath.yawt1)])
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(rspath.yawt1, rspath.yawt2)])

        return cost
    
    def calculate_rs_for_heuristic_modify(self, node, ngoal):
        """
        A newly version
        find a non_holonomic rs path for two tractor trailer system
        """
        
        # 1 / (minimun radius) 
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        paths = self.calc_all_paths_modify(node, ngoal, maxc, step_size=self.step_size)

        if not paths:
            return None

        pq = hyastar.QueuePrior()
        for path in paths:
            pq.put(path, self.calc_rs_path_cost_two_trailer_modify(path))

        count = 0
        while not pq.empty():
            path = pq.get()
            if count == 0:
                path_first = path
            if self.config["plot_rs_path"]:
                plot_rs_path(path, self.ox, self.oy)
                plt.close()
            # this api could be some mistake
            ind = range(0, len(path.x), self.config["collision_check_step"])

            pathx = [path.x[k] for k in ind]
            pathy = [path.y[k] for k in ind]
            pathyaw = [path.yaw[k] for k in ind]
            pathyawt1 = [path.yawt1[k] for k in ind]
            pathyawt2 = [path.yawt2[k] for k in ind]
            count += 1
            if not self.is_collision(pathx, pathy, pathyaw, pathyawt1, pathyawt2):
                return path 
        
        return path_first
    
    def calc_hybrid_cost_new(self, n_curr, n_goal):
        """
        A newly implemented function to calculate the heuristic
        it takes the maximun of the non-holonomic heuristic and the holonomic heuristic with 
        obtacles
        Inputs:
        - ox/oy: obstacle(before reso)
        """
        # currently we use path length as our heuristic
        heuristic_non_holonomic = self.heuristic_RS_two_trailer_modify(n_curr, n_goal)
        heuristic_holonomic_obstacles = hyastar.calc_holonomic_heuristic_with_obstacle_value(n_curr, self.hmap, self.ox, self.oy, self.heuristic_reso)
        # heuristic_holonomic_obstacles = hmap[n_curr.xind - P.minx][n_curr.yind - P.miny]
        cost = n_curr.cost + \
            self.cost["h_cost"] * max(heuristic_holonomic_obstacles, heuristic_non_holonomic)

        return cost
    
    def calc_hybrid_cost_simplify(self, n_curr, n_goal, rscost):
        heuristic_non_holonomic = rscost
        heuristic_holonomic_obstacles = self.hmap[n_curr.xind - self.P.minx][n_curr.yind - self.P.miny]
        cost = n_curr.cost + \
             self.cost["h_cost"] * max(heuristic_non_holonomic, heuristic_holonomic_obstacles)

        return cost
    
    def calc_index(self, node):
        '''
        change the way to calculate node index
        '''
        ind = (node.yawind - self.P.minyaw) * self.P.xw * self.P.yw + \
            (node.yind - self.P.miny) * self.P.xw + \
            (node.xind - self.P.minx)

        yawt1_ind = round(node.yawt1[-1] / self.P.yawreso)
        yawt2_ind = round(node.yawt2[-1] / self.P.yawreso)
        ind += (yawt1_ind - self.P.minyawt1) * self.P.xw * self.P.yw * self.P.yaww
        ind += (yawt2_ind - self.P.minyawt2) * self.P.xw * self.P.yw * self.P.yaww * self.P.yawt1w

        return ind
    
    def calc_next_node(self, n, ind, u, d):
        '''
        Using the current node/ind and steer/direction to 
        generate new node
        
        n: current node (Node class)
        ind: node index (calc_index)
        u: steer
        d: direction
        P: parameters
        
        returns:
        a node class
        '''
        step = self.config["mp_step"]
        # step = self.xyreso * 2.0

        nlist = math.ceil(step / self.config["move_step"])
        xlist = [n.x[-1] + d * self.config["move_step"] * math.cos(n.yaw[-1])]
        ylist = [n.y[-1] + d * self.config["move_step"] * math.sin(n.yaw[-1])]
        yawlist = [self.pi_2_pi(n.yaw[-1] + d * self.config["move_step"] / self.vehicle.WB * math.tan(u))]
        yawt1list = [self.pi_2_pi(n.yawt1[-1] +
                            d * self.config["move_step"] / self.vehicle.RTR * math.sin(n.yaw[-1] - n.yawt1[-1]))]
        yawt2list = [self.pi_2_pi(n.yawt2[-1] +
                            d * self.config["move_step"] / self.vehicle.RTR2 * math.sin(n.yawt1[-1] - n.yawt2[-1]) * math.cos(n.yaw[-1] - n.yawt1[-1]))]
        
        

        for i in range(nlist - 1):
            xlist.append(xlist[i] + d * self.config["move_step"] * math.cos(yawlist[i]))
            ylist.append(ylist[i] + d * self.config["move_step"] * math.sin(yawlist[i]))
            yawlist.append(self.pi_2_pi(yawlist[i] + d * self.config["move_step"] / self.vehicle.WB * math.tan(u)))
            yawt1list.append(self.pi_2_pi(yawt1list[i] +
                                    d * self.config["move_step"] / self.vehicle.RTR * math.sin(yawlist[i] - yawt1list[i])))
            yawt2list.append(self.pi_2_pi(yawt2list[i] +
                                    d * self.config["move_step"] / self.vehicle.RTR2 * math.sin(yawt1list[i] - yawt2list[i]) * math.cos(yawlist[i] - yawt1list[i])))

        xind = round(xlist[-1] / self.xyreso)
        yind = round(ylist[-1] / self.xyreso)
        yawind = round(yawlist[-1] / self.yawreso)

        # The following includes the procedure to 
        # calculate the cost of each node
        cost = 0.0

        if d > 0:
            direction = 1.0
            cost += abs(step)
        else:
            direction = -1.0
            cost += abs(step) * self.cost["backward_cost"]

        if direction != n.direction:  # switch back penalty
            cost += self.cost["gear_cost"]

        cost += self.cost["steer_angle_cost"] * abs(u)  # steer penalyty
        cost += self.cost["steer_change_cost"] * abs(n.steer - u)  # steer change penalty
        # may need to cancel this
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(yawlist, yawt1list)])  # jacknif cost
        # I add a term
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(yawt1list, yawt2list)])
        
        cost = n.cost + cost

        directions = [direction for _ in range(len(xlist))]

        # check whether there is jack-knife state
        try:
            node = hyastar.Node_two_trailer(self.vehicle, xind, yind, yawind, direction, xlist, ylist,
                        yawlist, yawt1list, yawt2list, directions, u, cost, ind)
        except:
            return None

        return node
    
    def is_index_ok(self, node, collision_check_step: int) -> bool:
        """
        check if the node is legal for a two trailer system
        - node: calc node (Node class)
        - P: parameters
        returns:
        whether the current node is ok
        """
        # check node index
        # check whether to go outside
        # TODO: check whether this is necessary
        if node.xind <= self.P.minx or \
                node.xind >= self.P.maxx or \
                node.yind <= self.P.miny or \
                node.yind >= self.P.maxy:
            return False

        ind = range(0, len(node.x), collision_check_step)

        x = [node.x[k] for k in ind]
        y = [node.y[k] for k in ind]
        yaw = [node.yaw[k] for k in ind]
        yawt1 = [node.yawt1[k] for k in ind]
        yawt2 = [node.yawt2[k] for k in ind]

        if self.is_collision(x, y, yaw, yawt1, yawt2):
            return False

        return True
    
    def is_collision(self, x, y, yaw, yawt1, yawt2) -> bool:
        '''
        check whether there is collision
        Inputs:
        x, y, yaw, yawt1, yawt2: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        for ix, iy, iyaw, iyawt1, iyawt2 in zip(x, y, yaw, yawt1, yawt2):
            # first trailer test collision
            d = self.safe_d
            deltal1 = (self.vehicle.RTF + self.vehicle.RTB) / 2.0 #which is exactly C.RTR
            rt1 = (self.vehicle.RTB - self.vehicle.RTF) / 2.0 + d #half length of trailer1 plus d

            ctx1 = ix - deltal1 * math.cos(iyawt1)
            cty1 = iy - deltal1 * math.sin(iyawt1)

            idst1 = self.P.kdtree.query_ball_point([ctx1, cty1], rt1)

            if idst1:
                for i in idst1:
                    xot1 = self.P.ox[i] - ctx1
                    yot1 = self.P.oy[i] - cty1

                    dx_trail1 = xot1 * math.cos(iyawt1) + yot1 * math.sin(iyawt1)
                    dy_trail1 = -xot1 * math.sin(iyawt1) + yot1 * math.cos(iyawt1)

                    if abs(dx_trail1) <= rt1 and \
                            abs(dy_trail1) <= self.vehicle.W / 2.0 + d:
                        return True
            # check the second trailer collision
            deltal2 = (self.vehicle.RTF2 + self.vehicle.RTB2) / 2.0
            rt2 = (self.vehicle.RTB2 - self.vehicle.RTF2) / 2.0 + d
            
            ctx2 = ctx1 - deltal2 * math.cos(iyawt2)
            cty2 = cty1 - deltal2 * math.sin(iyawt2)
            
            idst2 = self.P.kdtree.query_ball_point([ctx2, cty2], rt2)
            
            if idst2:
                for i in idst2:
                    xot2 = self.P.ox[i] - ctx2
                    yot2 = self.P.oy[i] - cty2
                    
                    dx_trail2 = xot2 * math.cos(iyawt2) + yot2 * math.sin(iyawt2)
                    dy_trail2 = -xot2 * math.cos(iyawt2) + yot2 * math.cos(iyawt2)
                    
                    if abs(dx_trail2) <= rt2 and \
                        abs(dy_trail2) <= self.vehicle.W / 2.0 + d:
                            return True
                        
            # check the tractor collision
            deltal = (self.vehicle.RF - self.vehicle.RB) / 2.0
            rc = (self.vehicle.RF + self.vehicle.RB) / 2.0 + d

            cx = ix + deltal * math.cos(iyaw)
            cy = iy + deltal * math.sin(iyaw)

            ids = self.P.kdtree.query_ball_point([cx, cy], rc)

            if ids:
                for i in ids:
                    xo = self.P.ox[i] - cx
                    yo = self.P.oy[i] - cy

                    dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                    dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                    if abs(dx_car) <= rc and \
                            abs(dy_car) <= self.vehicle.W / 2.0 + d:
                        return True

        return False
    
    def calc_all_paths_modify(self, node, ngoal, maxc, step_size):
        # Fank: 
        # Input: node - start node
        #        nogal - goal node
        #        maxc - maximum culvature
        # newly add control list when exploring
        # newly add control list when exploring
        sx, sy, syaw, syawt1, syawt2 = node.x[-1], node.y[-1], node.yaw[-1], node.yawt1[-1], node.yawt2[-1]
        gx, gy, gyaw, gyawt1, gyawt2 = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1], ngoal.yawt1[-1], ngoal.yawt2[-1]
        q0 = [sx, sy, syaw]
        q1 = [gx, gy, gyaw]
        input = np.array([sx, sy, syaw, syawt1, syawt2])
        goal = np.array([gx, gy, gyaw, gyawt1, gyawt2])

        paths = curves_generator.generate_path(q0, q1, maxc)

        for path in paths:
            rscontrol_list = extract_rs_path_control(path, self.vehicle.MAX_STEER, maxc)
            control_list = action_recover_from_planner(rscontrol_list)
            path.x, path.y, path.yaw, path.yawt1, path.yawt2, path.directions, path.valid = self.forward_simulation_two_trailer(input, goal, control_list)
            path.lengths = [l / maxc for l in path.lengths]
            path.L = path.L / maxc
            # add rscontrollist once search the path
            path.rscontrollist = rscontrol_list
            

        return paths
    
    def calc_all_paths_simplified(self, node, ngoal, maxc):
        # Fank: 
        # Input: node - start node
        #        nogal - goal node
        #        maxc - maximum culvature
        # this function adds more information for the rspath we selected
        
        sx, sy, syaw, syawt1, syawt2 = node.x[-1], node.y[-1], node.yaw[-1], node.yawt1[-1], node.yawt2[-1]
        gx, gy, gyaw, gyawt1, gyawt2 = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1], ngoal.yawt1[-1], ngoal.yawt2[-1]
        q0 = [sx, sy, syaw]
        q1 = [gx, gy, gyaw]
        input = np.array([sx, sy, syaw, syawt1, syawt2])
        goal = np.array([gx, gy, gyaw, gyawt1, gyawt2])

        paths = curves_generator.generate_path(q0, q1, maxc)

        for path in paths:
            rscontrol_list = extract_rs_path_control(path, self.vehicle.MAX_STEER, maxc)
            control_list = action_recover_from_planner(rscontrol_list)
            path.x, path.y, path.yaw, path.yawt1, path.yawt2, path.directions, path.info = self.forward_simulation_two_trailer_modify(input, goal, control_list)
            path.lengths = [l / maxc for l in path.lengths]
            path.L = path.L / maxc
            # add rscontrollist once search the path
            path.rscontrollist = rscontrol_list
            # put calc_rs_cost_here
            path.rscost = self.calc_rs_path_cost_two_trailer_modify(path)
            # Fank: check here if there is jack_knife
            if path.info["accept"] and (not path.info["collision"]):    
                xind = round(path.x[-1] / self.xyreso)
                yind = round(path.y[-1] / self.xyreso)
                yawind = round(path.yaw[-1] / self.yawreso)
                direction = path.directions[-1]
                fpind =  self.calc_index(node) 
                fcost = node.cost + path.rscost
                fx = path.x[1:]
                fy = path.y[1:]
                fyaw = path.yaw[1:]
                fyawt1 = path.yawt1[1:]
                fyawt2 = path.yawt2[1:]
                
                fd = []
                for d in path.directions[1:]:
                    if d >= 0:
                        fd.append(1.0)
                    else:
                        fd.append(-1.0)
                fsteer = 0.0
                try:
                    final_node = hyastar.Node_two_trailer(self.vehicle, xind, yind, yawind, direction,
                        fx, fy, fyaw, fyawt1, fyawt2, fd, fsteer, fcost, fpind)
                    path.info["jack_knife"] = False
                    path.info["final_node"] = final_node
                except:
                    path.info["jack_knife"] = True

        return paths
    
    def forward_simulation_two_trailer(self, input, goal, control_list, simulation_freq=10):
        # Fank: use the rs_path control we extract to forward simulation to 
        # check whether suitable this path
        # control_list: clip to [-1,1]
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
        path_x_list, path_y_list, path_yaw_list, path_yawt1_list = [], [], [], []
        path_yawt2_list = []
        directions = []
        controlled_vehicle = tt_envs.TwoTrailer(config_dict)
        controlled_vehicle.reset(*input)
        path_x, path_y, path_yaw, path_yawt1, path_yawt2 = controlled_vehicle.state
        path_x_list.append(path_x)
        path_y_list.append(path_y)
        path_yaw_list.append(path_yaw)
        path_yawt1_list.append(path_yawt1)
        path_yawt2_list.append(path_yawt2)
        for action_clipped in control_list:
            if action_clipped[0] > 0:
                directions.append(1)
            else:
                directions.append(-1)
            controlled_vehicle.step(action_clipped, 1 / simulation_freq)
            path_x, path_y, path_yaw, path_yawt1, path_yawt2 = controlled_vehicle.state
            path_x_list.append(path_x)
            path_y_list.append(path_y)
            path_yaw_list.append(path_yaw)
            path_yawt1_list.append(path_yawt1)
            path_yawt2_list.append(path_yawt2)
            
        directions.append(directions[-1])
        final_state = np.array(controlled_vehicle.state)
        # distance_error = np.linalg.norm(goal - final_state)
        distance_error = mixed_norm(goal, final_state)
        if distance_error > self.config["acceptance_error"]:
            info = False
        else:
            
            info = True
        return path_x_list, path_y_list, path_yaw_list, path_yawt1_list, path_yawt2_list, directions, info
    
    def forward_simulation_two_trailer_modify(self, input, goal, control_list, simulation_freq=10):
        # Fank: use the rs_path control we extract to forward simulation to 
        # check whether suitable this path
        # control_list: clip to [-1,1]
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
        path_x_list, path_y_list, path_yaw_list, path_yawt1_list = [], [], [], []
        path_yawt2_list, path_yawt3_list = [], []
        directions = []
        controlled_vehicle = tt_envs.TwoTrailer(config_dict)
        controlled_vehicle.reset(*input)
        path_x, path_y, path_yaw, path_yawt1, path_yawt2 = controlled_vehicle.state
        path_x_list.append(path_x)
        path_y_list.append(path_y)
        path_yaw_list.append(path_yaw)
        path_yawt1_list.append(path_yawt1)
        path_yawt2_list.append(path_yawt2)
        
        for action_clipped in control_list:
            if action_clipped[0] > 0:
                directions.append(1)
            else:
                directions.append(-1)
            controlled_vehicle.step(action_clipped, 1 / simulation_freq)
            path_x, path_y, path_yaw, path_yawt1, path_yawt2 = controlled_vehicle.state
            path_x_list.append(path_x)
            path_y_list.append(path_y)
            path_yaw_list.append(path_yaw)
            path_yawt1_list.append(path_yawt1)
            path_yawt2_list.append(path_yawt2)
            
        directions.append(directions[-1])
        final_state = np.array(controlled_vehicle.state)
        # distance_error = np.linalg.norm(goal - final_state)
        distance_error = mixed_norm(goal, final_state)
        # Fank: accept(false means not good)
        #       collision(false means no collision)
        #       jack_knife(false means no jack_knife)
        info = {
            "accept": False,
            "collision": None,
            "jack_knife": None,
        }
        if distance_error > self.config["acceptance_error"]:
            info["accept"] = False
        else:
            info["accept"] = True
        
        if info["accept"]:
            # Fank: check whether collision here
            ind = range(0, len(path_x_list), self.config["collision_check_step"])
            pathx = [path_x_list[k] for k in ind]
            pathy = [path_y_list[k] for k in ind]
            pathyaw = [path_yaw_list[k] for k in ind]
            pathyawt1 = [path_yawt1_list[k] for k in ind]
            pathyawt2 = [path_yawt2_list[k] for k in ind]
            if self.is_collision(pathx, pathy, pathyaw, pathyawt1, pathyawt2):
                info["collision"] = True
            else:
                # no collision
                info["collision"] = False
        
        return path_x_list, path_y_list, path_yaw_list, path_yawt1_list, path_yawt2_list, directions, info
    
    
    def is_same_grid(self, node1, node2):
        """
        whether the two nodes are on the same grid
        """
        if node1.xind != node2.xind or \
                node1.yind != node2.yind or \
                node1.yawind != node2.yawind:
            return False

        return True
    
    def is_the_start(self, node1, node2):
        """
        whether the two nodes are all start node
        """
        if len(node1.x) == 1 and len(node2.y) == 1:
            return True
        return False
    
    def analystic_expantion_modify(self, node, ngoal):
        """
        the returned path contains the start and the end
        which is also admissible(TT configuration and no collision)
        """
        # Fank
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        # I add a new attribute to this function 
        paths = self.calc_all_paths_modify(node, ngoal, maxc, step_size=self.step_size)

        if not paths:
            return None

        pq = hyastar.QueuePrior()
        for path in paths:
            if not path.valid:
                continue
            pq.put(path, self.calc_rs_path_cost_two_trailer_modify(path))
            # pq.put(path, calc_rs_path_cost_one_trailer(path, yawt1))

        while not pq.empty():
            path = pq.get()
            # check whether collision
            ind = range(0, len(path.x), self.config["collision_check_step"])
            pathx = [path.x[k] for k in ind]
            pathy = [path.y[k] for k in ind]
            pathyaw = [path.yaw[k] for k in ind]
            pathyawt1 = [path.yawt1[k] for k in ind]
            pathyawt2 = [path.yawt2[k] for k in ind]

            if not self.is_collision(pathx, pathy, pathyaw, pathyawt1, pathyawt2):
                return path

        return None
    
    
    def update_node_with_analystic_expantion_modify(self, n_curr, ngoal):
        """
        find a admissible rs path for two trailer system
        this is a modified version
        Inputs:
        - n_curr: curr node
        - ngoal: goal node
        - P: parameters
        Return:
        - flag: Boolean whether we find a admissible path
        - fpath: a node from n_curr -> ngoal(contains ngoal configuration not n_curr configuration)
        """
        # now the returnd path has a new attribute rscontrollist
        # return the waypoints and control_list all in path
        path = self.analystic_expantion_modify(n_curr, ngoal)  # rs path: n -> ngoal

        if not path:
            return False, None, None, None

        
        fx = path.x[1:]
        fy = path.y[1:]
        fyaw = path.yaw[1:]
        fyawt1 = path.yawt1[1:]
        fyawt2 = path.yawt2[1:]
        

        fd = []
        for d in path.directions[1:]:
            if d >= 0:
                fd.append(1.0)
            else:
                fd.append(-1.0)
        fsteer = 0.0
        # fd = path.directions[1:-1]

        fcost = n_curr.cost + self.calc_rs_path_cost_two_trailer_modify(path)
        # fcost = n_curr.cost + calc_rs_path_cost_one_trailer(path, yawt1)
        fpind = self.calc_index(n_curr)

        try:
            #here n_curr.xind might be wrong
            #but doesn't matter
            fpath = hyastar.Node_two_trailer(self.vehicle, ngoal.xind, ngoal.yind, ngoal.yawind, ngoal.direction,
                        fx, fy, fyaw, fyawt1, fyawt2, fd, fsteer, fcost, fpind)
        except:
            return False, None, None, None
        # abandon the first method
        return True, fpath, path.rscontrollist, path
    
    def rs_gear(self, node, ngoal):
        # Fank: put all rs related tech here
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        # I add a new attribute to this function 
        # Using a simplified version of calc_all_paths
        paths = self.calc_all_paths_simplified(node, ngoal, maxc)
        
        
        find_feasible = False
        if not paths:
            return find_feasible, None
        pq = hyastar.QueuePrior()
        
        for path in paths:
            if path.info["jack_knife"] == False:
                find_feasible = True
                return find_feasible, path
            pq.put(path, path.rscost)
        #TODO: may have to adjust
        while not pq.empty():
            path = pq.get()
            find_feasible = False
            return find_feasible, path
    
    def extract_path_and_control(self, closed, ngoal, nstart, reverse=False, find_rs_path=True):
        """
        extract the path before rs path
        notice that there will be some unavoidable mistakes
        - find_rs_path: whether we find rs path (always yes)
        """
        rx, ry, ryaw, ryawt1, ryawt2, direc = [], [], [], [], [], []
        expand_control_list = []
        # TODO: here you made a mistake
        step = self.config["mp_step"]
        nlist = math.ceil(step / self.config["move_step"])
        # cost = 0.0
        node = ngoal
        count = 0
        
        while True:
            #append the current node state configuration
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            ryawt1 += node.yawt1[::-1]
            ryawt2 += node.yawt2[::-1]
            direc += node.directions[::-1]
            # cost += node.cost

            if self.is_the_start(node, nstart) and self.is_same_grid(node, nstart):
                break
            if find_rs_path:
                if count > 0: #which means this is definitely not rs path
                    for i in range(nlist):
                        expand_control_list.append(np.array([node.directions[-1] * self.step_size, node.steer]))       
            else:
                for i in range(nlist):
                    expand_control_list.append(np.array([node.directions[-1] * self.step_size, node.steer
                                                        ]))
            # tracking parent ind
            node = closed[node.pind]
            count += 1
        if not reverse:
            rx = rx[::-1]
            ry = ry[::-1]
            ryaw = ryaw[::-1]
            ryawt1 = ryawt1[::-1]
            ryawt2 = ryawt2[::-1]
            direc = direc[::-1]
            direc[0] = direc[1]
        
        if self.config["plot_final_path"]:
            self.plot_real_path(rx, ry)
            save_dir = './planner_result/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            files = os.listdir(save_dir)
            
            file_index = 0
            file_name = f"hybrid_expand_tree_two_trailer_{file_index}.png"
            while file_name in files:
                file_index += 1
                file_name = f"hybrid_expand_tree_two_trailer_{file_index}.png"
            plt.savefig(os.path.join(save_dir, file_name))
        plt.close()  
         
        path = hyastar.Path_two_trailer(rx, ry, ryaw, ryawt1, ryawt2, direc)
        expand_control_list = expand_control_list[::-1]
        
        return path, expand_control_list
    
    def extract_path(self, closed, ngoal, nstart, reverse=False):
        """
        extract the final path
        closed: closed_set (dictionary: key is node_ind, value is node class)
        ngoal: goal node class
        nstart: start node class
        reverse: whether to reverse or not
        
        returns:
        path class
        """
        rx, ry, ryaw, ryawt1, ryawt2, direc = [], [], [], [], [], []
        # cost = 0.0
        node = ngoal
        
        while True:
            #append the current node state configuration
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            ryawt1 += node.yawt1[::-1]
            ryawt2 += node.yawt2[::-1]
            direc += node.directions[::-1]
            # cost += node.cost

            if self.is_the_start(node, nstart) and self.is_same_grid(node, nstart):
                break
            # tracking parent ind
            node = closed[node.pind]
        if not reverse:
            rx = rx[::-1]
            ry = ry[::-1]
            ryaw = ryaw[::-1]
            ryawt1 = ryawt1[::-1]
            ryawt2 = ryawt2[::-1]
            direc = direc[::-1]
            direc[0] = direc[1]
        path = hyastar.Path_two_trailer(rx, ry, ryaw, ryawt1, ryawt2, direc)

        return path
    
    def visualize_hmap(self, hmap):
        # x = ngoal.x[-1]
        # y = ngoal.y[-1]
        # yaw = ngoal.yaw[-1]
        # yawt1 = ngoal.yawt1[-1]
        # yawt2 = ngoal.yawt2[-1]
        # yawt3 = ngoal.yawt3[-1]
        
        # ox, oy = map_env1()
        # define your map
        hmap = np.where(np.isinf(hmap), np.nan, hmap)


        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        hmap_for_vis = np.flip(np.transpose(hmap), 0)
        
        # cax = ax.imshow(hmap_for_vis, cmap="inferno", extent=[-1, 41, -1, 41], aspect='auto')

        # # 设置轴的范围
        # ax.set_xlim(0, 40)
        # ax.set_ylim(0, 40)
        #differnt map and resolution set differently
        # x_min = 0 - (reso / 2)
        # x_max = 29 + (reso / 2)
        # y_min = 0 - (reso / 2)
        # y_max = 36 + (reso / 2)
        
        x_min = min(self.ox) - (self.heuristic_reso / 2)
        x_max = max(self.oy) + (self.heuristic_reso / 2)
        y_min = min(self.oy) - (self.heuristic_reso / 2)
        y_max = max(self.oy) + (self.heuristic_reso / 2)
        
        # cax = ax.imshow(hmap_for_vis, cmap="inferno", extent=[-1, 29, -5, 37], aspect='auto')
        cax = ax.imshow(hmap_for_vis, cmap="jet", extent=[x_min, x_max, y_min, y_max], aspect='auto')

        # 设置轴的范围
        # ax.set_xlim(0, 29)
        # ax.set_ylim(0, 36)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 添加颜色条
        cbar = fig.colorbar(cax)
        
        plt.plot(self.ox, self.oy, 'sk', markersize=1)

        plt.savefig('hmap_challenge_cases.png')
        
    def plan(self, start:np.ndarray, goal:np.ndarray, get_control_sequence:bool, verbose=False, *args, **kwargs):
        """
        Main Planning Algorithm for 3-tt systems
        :param start: starting point (np_array)
        :param goal: goal point (np_array)
        - path: all the six-dim state along the way (using extract function)
        - rs_path: contains the rspath and rspath control list
        - control list: rspath control list + expand control list
        """
        
        self.sx, self.sy, self.syaw, self.syawt1, self.syawt2 = start
        self.gx, self.gy, self.gyaw, self.gyawt1, self.gyawt2 = goal
        self.syaw, self.syawt1, self.syawt2 = self.pi_2_pi(self.syaw), self.pi_2_pi(self.syawt1), self.pi_2_pi(self.syawt2)
        self.gyaw, self.gyawt1, self.gyawt2 = self.pi_2_pi(self.gyaw), self.pi_2_pi(self.gyawt1), self.pi_2_pi(self.gyawt2)
        self.sxr, self.syr = round(self.sx / self.xyreso), round(self.sy / self.xyreso)
        self.gxr, self.gyr = round(self.gx / self.xyreso), round(self.gy / self.xyreso)
        self.syawr = round(self.syaw / self.yawreso)
        self.gyawr = round(self.gyaw / self.yawreso)
        
        if self.heuristic_type == "critic":
            with open(f'HybridAstarPlanner/supervised_data.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            # inputs_tensor = torch.tensor(data_dict['inputs'], dtype=torch.float32)
            labels_tensor = torch.tensor(data_dict['labels'], dtype=torch.float32)
            # inputs = data_dict['inputs']
            # labels = data_dict['labels']
            labels_mean = labels_tensor.mean()
            labels_std = labels_tensor.std()
            # labels_normalized = (labels_tensor - labels_mean) / labels_std
        # put the class in this file
        nstart = hyastar.Node_two_trailer(self.vehicle, self.sxr, self.syr, self.syawr, 1, \
            [self.sx], [self.sy], [self.pi_2_pi(self.syaw)], [self.pi_2_pi(self.syawt1)], [self.pi_2_pi(self.syawt2)], [1], 0.0, 0.0, -1)
        ngoal = hyastar.Node_two_trailer(self.vehicle, self.gxr, self.gyr, self.gyawr, 1, \
            [self.gx], [self.gy], [self.pi_2_pi(self.gyaw)], [self.pi_2_pi(self.gyawt1)], [self.pi_2_pi(self.gyawt2)], [1], 0.0, 0.0, -1)
        if not self.is_index_ok(nstart, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(ngoal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        # TODO: change the api (seems this one is better)
        if self.obs:
            self.hmap = hyastar.calc_holonomic_heuristic_with_obstacle(ngoal, self.P.ox, self.P.oy, self.heuristic_reso, self.heuristic_rr)
        if self.config["plot_heuristic_nonholonomic"]:
            self.visualize_hmap(self.hmap)
        
        
        steer_set, direc_set = self.calc_motion_set()
        # change the way to calc_index
        open_set, closed_set = {self.calc_index(nstart): nstart}, {}
        # non_h = calc_non_holonomic_heuritstic(nstart, ngoal, gyawt1, gyawt2, gyawt3, fP)
        
        # reset qp for next using
        self.qp.reset()
        # the main change here will be the heuristic value
        if self.heuristic_type == "traditional":
            self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_new(nstart, ngoal))
        else:
            # here you need to change this heuristic
            self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_new_critic(nstart, labels_mean, labels_std))
        # an indicator whether find the rs path at last
        find_rs_path = False
        # the loop number for analystic expansion
        count = 0
        # Main Loop
        while True:
            # I will try not to use this
            # may need to modify when there's obstacle
            if not open_set or self.qp.empty():
                # consider this function
                print("failed finding a feasible path")
                self.extract_failed_path(closed_set, nstart)
                return None, None, None
            count += 1
            # add if the loop's too much
            if count > self.max_iter:
                print("waste a long time to find")
                return None, None, None
            ind = self.qp.get()
            n_curr = open_set[ind]
            closed_set[ind] = n_curr
            open_set.pop(ind)
            
            # key and the most tricky part of the algorithm
            update, fpath, rs_control_list, rs_path = self.update_node_with_analystic_expantion_modify(n_curr, ngoal)

            if update:
                fnode = fpath
                find_rs_path = True
                if self.config["plot_expand_tree"]:
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    plot_rs_path(rs_path, self.ox, self.oy)
                    # plt.close()
                if verbose:
                    print("final analystic expantion node number:", count)
                break
            
            for i in range(len(steer_set)):
                node = self.calc_next_node(n_curr, ind, steer_set[i], direc_set[i])
                if not node:
                    continue
                if not self.is_index_ok(node, self.config["collision_check_step"]):
                    continue
                node_ind = self.calc_index(node)
                if node_ind in closed_set:
                    continue
                if node_ind not in open_set:
                    open_set[node_ind] = node
                    if self.heuristic_type == "traditional":
                        self.qp.put(node_ind, self.calc_hybrid_cost_new(node, ngoal))
                    else:
                        self.qp.put(node_ind, self.calc_hybrid_cost_new_critic(node, ngoal, labels_mean, labels_std))
                else:
                    if open_set[node_ind].cost > node.cost:
                        open_set[node_ind] = node
                        if self.qp_type == "heapdict":  
                            # if using heapdict, here you can modify the value
                            if self.heuristic_type == "traditional":
                                self.qp.queue[node_ind] = self.calc_hybrid_cost_new(node, ngoal)
                            else:
                                self.qp.queue[node_ind] = self.calc_hybrid_cost_new_critic(node, ngoal, labels_mean, labels_std)           
        if verbose:
            print("final expand node: ", len(open_set) + len(closed_set))
        
        if get_control_sequence:
            path, expand_control_list = self.extract_path_and_control(closed_set, fnode, nstart,find_rs_path=find_rs_path)
            if find_rs_path:
                all_control_list = expand_control_list + rs_control_list
            else:
                rs_path = None
                all_control_list = all_control_list
            return path, all_control_list, rs_path
        else:
            if find_rs_path: 
                return self.extract_path(closed_set, fnode, nstart), None, rs_path
            else:
                return self.extract_path(closed_set, fnode, nstart), None, None
    
    def plan_new_version(self, start:np.ndarray, goal:np.ndarray, get_control_sequence:bool, verbose=False, *args, **kwargs):
        """
        New Version of Main Planning Algorithm for 3-tt systems
        this algorithm saves some time
        :param start: starting point (np_array)
        :param goal: goal point (np_array)
        - path: all the six-dim state along the way (using extract function)
        - rs_path: contains the rspath and rspath control list
        - control list: rspath control list + expand control list
        """
        # input the given information
        self.sx, self.sy, self.syaw, self.syawt1, self.syawt2 = start
        self.gx, self.gy, self.gyaw, self.gyawt1, self.gyawt2 = goal
        self.syaw, self.syawt1, self.syawt2 = self.pi_2_pi(self.syaw), self.pi_2_pi(self.syawt1), self.pi_2_pi(self.syawt2)
        self.gyaw, self.gyawt1, self.gyawt2 = self.pi_2_pi(self.gyaw), self.pi_2_pi(self.gyawt1), self.pi_2_pi(self.gyawt2)
        self.sxr, self.syr = round(self.sx / self.xyreso), round(self.sy / self.xyreso)
        self.gxr, self.gyr = round(self.gx / self.xyreso), round(self.gy / self.xyreso)
        self.syawr = round(self.syaw / self.yawreso)
        self.gyawr = round(self.gyaw / self.yawreso)
        
        if self.heuristic_type == "critic":
            with open(f'HybridAstarPlanner/supervised_data.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            # inputs_tensor = torch.tensor(data_dict['inputs'], dtype=torch.float32)
            labels_tensor = torch.tensor(data_dict['labels'], dtype=torch.float32)
            # inputs = data_dict['inputs']
            # labels = data_dict['labels']
            labels_mean = labels_tensor.mean()
            labels_std = labels_tensor.std()
            # labels_normalized = (labels_tensor - labels_mean) / labels_std
        
        # initialzie start and goal node class
        nstart = hyastar.Node_two_trailer(self.vehicle, self.sxr, self.syr, self.syawr, 1, \
            [self.sx], [self.sy], [self.pi_2_pi(self.syaw)], [self.pi_2_pi(self.syawt1)], [self.pi_2_pi(self.syawt2)], [1], 0.0, 0.0, -1)
        ngoal = hyastar.Node_two_trailer(self.vehicle, self.gxr, self.gyr, self.gyawr, 1, \
            [self.gx], [self.gy], [self.pi_2_pi(self.gyaw)], [self.pi_2_pi(self.gyawt1)], [self.pi_2_pi(self.gyawt2)], [1], 0.0, 0.0, -1)
        # check whether valid
        if not self.is_index_ok(nstart, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(ngoal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        # calculate heuristic for obstacle
        if self.obs:
            self.hmap = hyastar.calc_holonomic_heuristic_with_obstacle(ngoal, self.P.ox, self.P.oy, self.heuristic_reso, self.heuristic_rr)
        if self.config["plot_heuristic_nonholonomic"]:
            self.visualize_hmap(self.hmap)
        
        
        steer_set, direc_set = self.calc_motion_set()
        # Initialize open_set and closed_set
        open_set, closed_set = {self.calc_index(nstart): nstart}, {}
        
        # reset qp for next using
        self.qp.reset()
        # an indicator whether find the rs path at last(for extract)
        find_rs_path = False
        # the loop number for analystic expansion(counting number)
        count = 0
        # update parameter
        update = False
        # the main change here will be the heuristic value
        if self.heuristic_type == "traditional":
            find_feasible, path = self.rs_gear(nstart, ngoal)
            # if find feasible, then go to extract
            # else calculate heuristic
            if find_feasible:
                fnode = path.info["final_node"]
                find_rs_path = True
                update = find_feasible
                rs_path = path
                rs_control_list = path.rscontrollist
                if self.config["plot_expand_tree"]:
                    plot_rs_path(rs_path, self.ox, self.oy)
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    # plt.close()
                if verbose:
                    print("find path at first time")
                closed_set[self.calc_index(nstart)] = nstart
            else:
                self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_simplify(nstart, ngoal, path.rscost))
        else:
            # wait for RL to guide search
            self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_new_critic(nstart, labels_mean, labels_std))
        
        # Main Loop
        while True:
            if update:
                # use the flag update to break the main loop
                break
            if not open_set or self.qp.empty():
                print("failed finding a feasible path")
                self.extract_failed_path(closed_set, nstart)
                return None, None, None
            count += 1
            # add if the loop's too much
            if count > self.max_iter:
                print("waste a long time to find")
                return None, None, None
            
            ind = self.qp.get()
            n_curr = open_set[ind]
            closed_set[ind] = n_curr
            open_set.pop(ind)

            # expand tree using motion primitive
            for i in range(len(steer_set)):
                node = self.calc_next_node(n_curr, ind, steer_set[i], direc_set[i])
                if not node:
                    # encounter jack_knife
                    continue
                if not self.is_index_ok(node, self.config["collision_check_step"]):
                    # check go outside or collision
                    continue
                node_ind = self.calc_index(node)
                if node_ind in closed_set:
                    # we will not calculate twice 
                    # Note that this can be a limitation
                    continue
                if node_ind not in open_set:
                    open_set[node_ind] = node
                    if self.heuristic_type == "traditional":
                        find_feasible, path = self.rs_gear(node, ngoal)
                        if find_feasible:
                            fnode = path.info["final_node"]
                            find_rs_path = True
                            update = find_feasible
                            rs_path = path
                            rs_control_list = path.rscontrollist
                            if self.config["plot_expand_tree"]:
                                plot_rs_path(rs_path, self.ox, self.oy)
                                self.plot_expand_tree(start, goal, closed_set, open_set)
                                # plt.close()
                            if verbose:
                                print("final expansion node number:", count)
                            # Here you need to add node to closed set
                            closed_set[node_ind] = node
                            # break the inner expand_tree loop
                            break
                        else:
                            self.qp.put(node_ind, self.calc_hybrid_cost_simplify(node, ngoal, path.rscost))
                    else:
                        # wait for RL to guide search
                        self.qp.put(node_ind, self.calc_hybrid_cost_new_critic(node, ngoal, labels_mean, labels_std))
                else:
                    if open_set[node_ind].cost > node.cost:
                        open_set[node_ind] = node
                        if self.qp_type == "heapdict":  
                            # if using heapdict, here you can modify the value
                            if self.heuristic_type == "traditional":
                                find_feasible, path = self.rs_gear(node, ngoal)
                                if find_feasible:
                                    fnode = path.info["final_node"]
                                    find_rs_path = True
                                    update = find_feasible
                                    rs_path = path
                                    rs_control_list = path.rscontrollist
                                    if self.config["plot_expand_tree"]:
                                        plot_rs_path(rs_path, self.ox, self.oy)
                                        self.plot_expand_tree(start, goal, closed_set, open_set)
                                        # plt.close()
                                    if verbose:
                                        print("final expansion node number:", count)
                                    closed_set[node_ind] = node
                                    break
                                else:
                                    self.qp.queue[node_ind] = self.calc_hybrid_cost_simplify(node, ngoal, path.rscost)
                                    
                            else:
                                #TODO: wait for the RL to guide
                                self.qp.queue[node_ind] = self.calc_hybrid_cost_new_critic(node, ngoal, labels_mean, labels_std)             
        if verbose:
            print("final expand node: ", len(open_set) + len(closed_set) - 1)
        
        if get_control_sequence:
            path, expand_control_list = self.extract_path_and_control(closed_set, fnode, nstart,find_rs_path=find_rs_path)
            if find_rs_path:
                all_control_list = expand_control_list + rs_control_list
            else:
                rs_path = None
                all_control_list = all_control_list
            return path, all_control_list, rs_path
        else:
            if find_rs_path: 
                return self.extract_path(closed_set, fnode, nstart), None, rs_path
            else:
                return self.extract_path(closed_set, fnode, nstart), None, None
    
    def plot_expand_tree(self, start, goal, closed_set, open_set):
        plt.axis("equal")
        ax = plt.gca() 
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
        
        for key, value in open_set.items():
            self.plot_node(value, color='gray')
        for key, value in closed_set.items():
            self.plot_node(value, color='red')
        # change here last plot goal and start
        self.vehicle.reset(*goal)
        
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        self.vehicle.reset(*start)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
    
    def plot_node(self, node, color):
        xlist = node.x
        ylist = node.y
        plt.plot(xlist, ylist, color=color, markersize=1)
    
    def plot_real_path(self, rx, ry):
        plt.plot(rx, ry, color="blue", markersize=1)
    
    
    def visualize_planning(self, start, goal, path, 
                           gif=True, save_dir='./HybridAstarPlanner/gif'):
        """visuliaze the planning result
        : param path: a path class
        : start & goal: cast as np.ndarray
        """
        print("Start Visulizate the Result")
        x = path.x
        y = path.y
        yaw = path.yaw
        yawt1 = path.yawt1
        yawt2 = path.yawt2
        yawt3 = path.yawt3
        direction = path.direction
        
        if gif:
            fig, ax = plt.subplots()

            def update(num):
                ax.clear()
                plt.axis("equal")
                k = num
                # plot env (obstacle)
                plt.plot(self.ox, self.oy, "sk", markersize=1)
                self.vehicle.reset(*start)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
                
                # plot the planning path
                plt.plot(x, y, linewidth=1.5, color='r')
                self.vehicle.reset(*goal)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
                self.vehicle.reset(x[k], y[k], yaw[k], yawt1[k], yawt2[k], yawt3[k])
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0

                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')
                plt.axis("equal")

            ani = FuncAnimation(fig, update, frames=len(x), repeat=True)

            # Save the animation
            writer = PillowWriter(fps=20)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                
            # base_path = "./HybridAstarPlanner/gif/path_animation"
            base_path = os.path.join(save_dir, 'hybrid_astar_path_plan_two_tractor_trailer')
            extension = ".gif"
            
            all_files = os.listdir(save_dir)
            matched_files = [re.match(r'hybrid_astar_path_plan_two_tractor_trailer(\d+)\.gif', f) for f in all_files]
            numbers = [int(match.group(1)) for match in matched_files if match]
            
            if numbers:
                save_index = max(numbers) + 1
            else:
                save_index = 1
            ani.save(base_path + str(save_index) + extension, writer=writer)
            print("Done Plotting")
            
        else:
            # this is when your device has display setting
            fig, ax = plt.subplots()
            # this is when your device has display setting
            plt.pause(5)

            for k in range(len(x)):
                plt.cla()
                plt.axis("equal")
                # plot env (obstacle)
                plt.plot(self.ox, self.oy, "sk", markersize=1)
                # plot the planning path
                plt.plot(x, y, linewidth=1.5, color='r')
                self.vehicle.reset(*start)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
                self.vehicle.reset(*goal)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')

                # calculate every time step
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    # different from a single car
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0
                # draw goal model
                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')
                plt.pause(0.0001)

            plt.show()

    def extract_failed_path(self, closed, nstart):
        
        for value in closed.values():
            plt.plot(value.x, value.y,'.', color='grey', markersize=1)
        plt.plot(nstart.x, nstart.y, 'o', color='r', markersize=3)    
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
        # plt.legend()
        plt.axis("equal")
        if not os.path.exists("planner_result/failed_trajectory_two_trailer"):
            os.makedirs("planner_result/failed_trajectory_two_trailer")
            
        base_path = "./planner_result/failed_trajectory_two_trailer"
        extension = ".png"
            
        all_files = os.listdir("./planner_result/failed_trajectory_two_trailer")
        matched_files = [re.match(r'explored(\d+)\.png', f) for f in all_files]
        numbers = [int(match.group(1)) for match in matched_files if match]
        
        if numbers:
            save_index = max(numbers) + 1
        else:
            save_index = 0
        plt.savefig(base_path + "/explored" + str(save_index) + extension)
        plt.close()
        # plt.savefig("HybridAstarPlanner/trajectory/explored.png")




class ThreeTractorTrailerHybridAstarPlanner(hyastar.BasicHybridAstarPlanner):
    @classmethod
    def default_config(cls) -> dict:
        return {
            "verbose": False, 
            "heuristic_type": "traditional",
            "vehicle_type": "three_trailer",
            "act_limit": 1, 
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
                "safe_metric": 3.0, 
            },
            "xy_reso": 2.0,
            "yaw_reso": np.deg2rad(15.0),
            "qp_type": "heapdict",
            "max_iter": 1000, # outer loop number
            "step_size": 0.2, # rs path sample step size
            "move_step": 0.2, # expand node tree
            "n_steer": 20, #how many parts to divide in motion primitives
            "mp_step": 40.0, # previous 2.0 * reso, how many moves in mp step
            "is_save_animation": True,
            "is_save_expand_tree": True,
            "visualize_mode": True,
            "dt": 0.1,
            "heuristic_reso": 2.0,
            "heuristic_rr": 1.0, # 0.5 * heuristic_reso
            "whether_obs": True,
            "safe_d": 0.0,
            "extend_area": 0.0,
            "collision_check_step": 10,
            "goal_yaw_error": np.deg2rad(3.0),
            "cost_configuration":
                {
                    "scissors_cost": 200.0,
                    "gear_cost": 100.0,
                    "backward_cost": 5.0,
                    "steer_change_cost": 5.0,
                    "h_cost": 10.0,
                    "steer_angle_cost": 1.0,
                },   
            "plot_heuristic_nonholonomic": False,
            "plot_rs_path": True,
            "plot_expand_tree": True,
            "plot_final_path": True,
            "plot_failed_path": False,
            "range_steer_set": 8, #need to set the same as n_steer
            "acceptance_error": 0.5,
            "N_steps": 20,
            "save_final_plot": False,
            "observation": "original",
        }
    
    def configure(self, config: Optional[dict]):
        if config:
            self.config.update(config)
        self.vehicle = tt_envs.ThreeTrailer(self.config["controlled_vehicle_config"])
        self.max_iter = self.config["max_iter"] 
        self.xyreso = self.config["xy_reso"]
        self.yawreso = self.config["yaw_reso"]
        self.qp_type = self.config["qp_type"] 
        self.safe_d = self.config["safe_d"]
        self.extend_area = self.config["extend_area"]
        self.obs = self.config['whether_obs']
        self.cost = self.config['cost_configuration']
        self.step_size = self.config["step_size"]
        self.n_steer = self.config["n_steer"]
        self.heuristic_type = self.config['heuristic_type']
        self.observation_type = self.config["observation"]
        if self.obs:
            self.heuristic_reso = self.config["heuristic_reso"]
            self.heuristic_rr = self.config["heuristic_rr"]
        if self.qp_type == "heapdict":
            self.qp = hyastar.NewQueuePrior()
        else:
            self.qp = hyastar.QueuePrior()
        # TODO: change the model path to the suited one
        # heuristic_type: traditional, rl, mix_original
        # mix_original_with_obstacles_info, mix_lidar_detection_one_hot, mix_lidar_detection_one_hot_triple
        # mix_attention
        if self.heuristic_type == 'rl':
            config_filename = "configs/agents/training/planner1.yaml"
            model_filename = "datasets/models/original_model.pth"
        elif self.heuristic_type == 'mix_original':
            config_filename = "configs/agents/eval/rl0.yaml"
            model_filename = "datasets/models/original_model.pth"
        elif self.heuristic_type == "mix_original_with_obstacles_info":
            config_filename = "configs/agents/eval/rl1_obs_mlp.yaml"
            model_filename = "datasets/models/original_model.pth"
        elif self.heuristic_type == "mix_lidar_detection_one_hot":
            config_filename = "configs/agents/eval/rl1_lidar_detection_one_hot.yaml"
            model_filename = "datasets/models/original_model.pth"
        elif self.heuristic_type == "mix_lidar_detection_one_hot_triple":
            config_filename = "configs/agents/eval/rl1_lidar_detection_one_hot_triple.yaml"
            model_filename = "datasets/models/original_model.pth"
        elif self.heuristic_type == "mix_attention":
            config_filename = "configs/agents/eval/rl1_attention.yaml"
            model_filename = "datasets/models/original_model.pth"
        if self.heuristic_type != "traditional":
            with open(config_filename, "r") as f:
                # TODO: each time you need to set the config file align with self.observation_type 
                config_algo = yaml.safe_load(f)
            self.agent = agents.SAC_ASTAR_META_NEW(env_fn=gym_tt_planning_env_fn,
                config=config_algo,
                device='cpu')
            self.agent.load(model_filename, whether_load_buffer=False)
            
    
    
    def __init__(self, ox, oy, config: Optional[dict] = None) -> None:
        self.config = self.default_config()
        self.configure(config)
        
        super().__init__(ox, oy)
    
    def calc_parameters(self):
        """calculate parameters of the planning problem
        return: para class implemented in hybrid_astar.py
        """
        minxm = min(self.ox) - self.extend_area
        minym = min(self.oy) - self.extend_area
        maxxm = max(self.ox) + self.extend_area
        maxym = max(self.oy) + self.extend_area

        self.ox.append(minxm)
        self.oy.append(minym)
        self.ox.append(maxxm)
        self.oy.append(maxym)

        minx = round(minxm / self.xyreso)
        miny = round(minym / self.xyreso)
        maxx = round(maxxm / self.xyreso)
        maxy = round(maxym / self.xyreso)

        xw, yw = maxx - minx + 1, maxy - miny + 1

        minyaw = round(-self.vehicle.PI / self.yawreso)
        maxyaw = round(self.vehicle.PI / self.yawreso)
        yaww = maxyaw - minyaw + 1

        minyawt1, maxyawt1, yawt1w = minyaw, maxyaw, yaww
        minyawt2, maxyawt2, yawt2w = minyaw, maxyaw, yaww
        minyawt3, maxyawt3, yawt3w = minyaw, maxyaw, yaww

        P = hyastar.Para_three_trailer(minx, miny, minyaw, minyawt1, minyawt2, minyawt3, maxx, maxy, maxyaw,
                maxyawt1, maxyawt2, maxyawt3, xw, yw, yaww, yawt1w, yawt2w, yawt3w, self.xyreso, self.yawreso, self.ox, self.oy, self.kdtree)

        return P
    
    
    def calc_motion_set(self):
        """
        this is much alike motion primitives
        """
        s = [i for i in np.arange(self.vehicle.MAX_STEER / self.n_steer,
                                self.config["range_steer_set"] * self.vehicle.MAX_STEER / self.n_steer, self.vehicle.MAX_STEER / self.n_steer)]

        steer = [0.0] + s + [-i for i in s]
        direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
        steer = steer + steer

        return steer, direc
    
    
    def heuristic_RS_three_trailer_modify(self, n_curr, n_goal, plot=False) -> float:
        """
        this is used when calculating heuristics
        
        Inputs:
        - n_curr: curr node
        - n_goal: goal node
        - ox, oy: obstacle (no reso)
        - plot: whether doing visualization
        """
        # check if the start/goal configuration is legal
        if not self.is_index_ok(n_curr, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(n_goal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        
        # get start/goal configuration from the node
        sx = n_curr.x[-1]
        sy = n_curr.y[-1]
        syaw0 = n_curr.yaw[-1]
        # syawt1 = n_curr.yawt1[-1]
        # syawt2 = n_curr.yawt2[-1]
        # syawt3 = n_curr.yawt3[-1]
        
        gx = n_goal.x[-1]
        gy = n_goal.y[-1]
        gyaw0 = n_goal.yaw[-1]
        # gyawt1 = n_goal.yawt1[-1]
        # gyawt2 = n_goal.yawt2[-1]
        # gyawt3 = n_goal.yawt3[-1]
        
        # the same start and goal
        epsilon = 1e-5
        if np.abs(sx - gx) <= epsilon and np.abs(sy - gy) <= epsilon and \
            np.abs(syaw0 - gyaw0) <= epsilon:
            return 0.0
        
        path = self.calculate_rs_for_heuristic_modify(n_curr, n_goal)
        return self.calc_rs_path_cost_three_trailer_modify(path)
    
    def calc_rs_path_cost_three_trailer_modify(self, rspath) -> float:
        """
        A newly version that rspath contains all the information
        this function calculate rs path cost based on rspath and yawt
        
        Inputs:
        - rspath: path class
        - yawt: the first trailer yaw
        """
        cost = 0.0

        for lr in rspath.lengths:
            if lr >= 0:
                cost += abs(lr)
            else:
                cost += abs(lr) * self.cost["backward_cost"]

        for i in range(len(rspath.lengths) - 1):
            if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
                cost += self.cost["gear_cost"]

        for ctype in rspath.ctypes:
            if ctype != "S":
                cost += self.cost["steer_angle_cost"] * abs(self.vehicle.MAX_STEER)

        nctypes = len(rspath.ctypes)
        ulist = [0.0 for _ in range(nctypes)]

        for i in range(nctypes):
            if rspath.ctypes[i] == "R":
                ulist[i] = -self.vehicle.MAX_STEER
            elif rspath.ctypes[i] == "WB":
                ulist[i] = self.vehicle.MAX_STEER

        for i in range(nctypes - 1):
            cost += self.cost["steer_change_cost"] * abs(ulist[i + 1] - ulist[i])

        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(rspath.yaw, rspath.yawt1)])
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(rspath.yawt1, rspath.yawt2)])
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(rspath.yawt2, rspath.yawt3)])

        return cost
    
    def calculate_rs_for_heuristic_modify(self, node, ngoal):
        """
        A newly version
        find a non_holonomic rs path for three tractor trailer system
        """
        
        # 1 / (minimun radius) 
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        paths = self.calc_all_paths_modify(node, ngoal, maxc, step_size=self.step_size)

        if not paths:
            return None

        pq = hyastar.QueuePrior()
        for path in paths:
            pq.put(path, self.calc_rs_path_cost_three_trailer_modify(path))

        count = 0
        while not pq.empty():
            path = pq.get()
            if count == 0:
                path_first = path
            if self.config["plot_rs_path"]:
                plot_rs_path(path, self.ox, self.oy)
                plt.close()
            # this api could be some mistake
            ind = range(0, len(path.x), self.config["collision_check_step"])

            pathx = [path.x[k] for k in ind]
            pathy = [path.y[k] for k in ind]
            pathyaw = [path.yaw[k] for k in ind]
            pathyawt1 = [path.yawt1[k] for k in ind]
            pathyawt2 = [path.yawt2[k] for k in ind]
            pathyawt3 = [path.yawt3[k] for k in ind]
            count += 1
            if not self.is_collision(pathx, pathy, pathyaw, pathyawt1, pathyawt2, pathyawt3):
                return path 
        
        return path_first
    
    def calc_hybrid_cost_new(self, n_curr, n_goal):
        """
        A newly implemented function to calculate the heuristic
        it takes the maximun of the non-holonomic heuristic and the holonomic heuristic with 
        obtacles
        Inputs:
        - ox/oy: obstacle(before reso)
        """
        # currently we use path length as our heuristic
        heuristic_non_holonomic = self.heuristic_RS_three_trailer_modify(n_curr, n_goal)
        heuristic_holonomic_obstacles = hyastar.calc_holonomic_heuristic_with_obstacle_value(n_curr, self.hmap, self.ox, self.oy, self.heuristic_reso)
        # heuristic_holonomic_obstacles = hmap[n_curr.xind - P.minx][n_curr.yind - P.miny]
        cost = n_curr.cost + \
            self.cost["h_cost"] * max(heuristic_holonomic_obstacles, heuristic_non_holonomic)

        return cost
    
    def calc_hybrid_cost_simplify(self, n_curr, n_goal, rscost):
        heuristic_non_holonomic = rscost
        heuristic_holonomic_obstacles = self.hmap[n_curr.xind - self.P.minx][n_curr.yind - self.P.miny]
        cost = n_curr.cost + \
             self.cost["h_cost"] * max(heuristic_non_holonomic, heuristic_holonomic_obstacles)

        return cost
    
    def calc_index(self, node):
        '''
        change the way to calculate node index
        '''
        ind = (node.yawind - self.P.minyaw) * self.P.xw * self.P.yw + \
            (node.yind - self.P.miny) * self.P.xw + \
            (node.xind - self.P.minx)

        yawt1_ind = round(node.yawt1[-1] / self.P.yawreso)
        yawt2_ind = round(node.yawt2[-1] / self.P.yawreso)
        yawt3_ind = round(node.yawt3[-1] / self.P.yawreso)
        ind += (yawt1_ind - self.P.minyawt1) * self.P.xw * self.P.yw * self.P.yaww
        ind += (yawt2_ind - self.P.minyawt2) * self.P.xw * self.P.yw * self.P.yaww * self.P.yawt1w
        ind += (yawt3_ind - self.P.minyawt3) * self.P.xw * self.P.yw * self.P.yaww * self.P.yawt1w * self.P.yawt2w

        return ind
    
    def calc_next_node(self, n, ind, u, d):
        '''
        Using the current node/ind and steer/direction to 
        generate new node
        
        n: current node (Node class)
        ind: node index (calc_index)
        u: steer
        d: direction
        P: parameters
        
        returns:
        a node class
        '''
        step = self.config["mp_step"]
        # step = self.xyreso * 2.0

        nlist = math.ceil(step / self.config["move_step"])
        assert nlist % self.config["N_steps"] == 0, "nlist should be divisible by N_steps"
        xlist = [n.x[-1] + d * self.config["move_step"] * math.cos(n.yaw[-1])]
        ylist = [n.y[-1] + d * self.config["move_step"] * math.sin(n.yaw[-1])]
        yawlist = [self.pi_2_pi(n.yaw[-1] + d * self.config["move_step"] / self.vehicle.WB * math.tan(u))]
        yawt1list = [self.pi_2_pi(n.yawt1[-1] +
                            d * self.config["move_step"] / self.vehicle.RTR * math.sin(n.yaw[-1] - n.yawt1[-1]))]
        yawt2list = [self.pi_2_pi(n.yawt2[-1] +
                            d * self.config["move_step"] / self.vehicle.RTR2 * math.sin(n.yawt1[-1] - n.yawt2[-1]) * math.cos(n.yaw[-1] - n.yawt1[-1]))]
        yawt3list = [self.pi_2_pi(n.yawt3[-1] +
                                d * self.config["move_step"] / self.vehicle.RTR3 * math.sin(n.yawt2[-1] - n.yawt3[-1]) * math.cos(n.yawt1[-1] - n.yawt2[-1]) * math.cos(n.yaw[-1] - n.yawt1[-1]))]
        

        for i in range(nlist - 1):
            xlist.append(xlist[i] + d * self.config["move_step"] * math.cos(yawlist[i]))
            ylist.append(ylist[i] + d * self.config["move_step"] * math.sin(yawlist[i]))
            yawlist.append(self.pi_2_pi(yawlist[i] + d * self.config["move_step"] / self.vehicle.WB * math.tan(u)))
            yawt1list.append(self.pi_2_pi(yawt1list[i] +
                                    d * self.config["move_step"] / self.vehicle.RTR * math.sin(yawlist[i] - yawt1list[i])))
            yawt2list.append(self.pi_2_pi(yawt2list[i] +
                                    d * self.config["move_step"] / self.vehicle.RTR2 * math.sin(yawt1list[i] - yawt2list[i]) * math.cos(yawlist[i] - yawt1list[i])))
            yawt3list.append(self.pi_2_pi(yawt3list[-1] +
                                d * self.config["move_step"] / self.vehicle.RTR3 * math.sin(yawt2list[-1] - yawt3list[-1]) * math.cos(yawt1list[-1] - yawt2list[-1]) * math.cos(yawlist[-1] - yawt1list[-1])))

        xind = round(xlist[-1] / self.xyreso)
        yind = round(ylist[-1] / self.xyreso)
        yawind = round(yawlist[-1] / self.yawreso)

        # The following includes the procedure to 
        # calculate the cost of each node
        cost = 0.0

        if d > 0:
            direction = 1.0
            cost += abs(step)
        else:
            direction = -1.0
            cost += abs(step) * self.cost["backward_cost"]

        if direction != n.direction:  # switch back penalty
            cost += self.cost["gear_cost"]

        cost += self.cost["steer_angle_cost"] * abs(u)  # steer penalyty
        cost += self.cost["steer_change_cost"] * abs(n.steer - u)  # steer change penalty
        # may need to cancel this
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(yawlist, yawt1list)])  # jacknif cost
        # I add a term
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(yawt1list, yawt2list)])
        
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y))
                                    for x, y in zip(yawt2list, yawt3list)])
        
        cost = n.cost + cost

        directions = [direction for _ in range(len(xlist))]

        # check whether there is jack-knife state
        try:
            node = hyastar.Node_three_trailer(self.vehicle, xind, yind, yawind, direction, xlist, ylist,
                        yawlist, yawt1list, yawt2list, yawt3list, directions, u, cost, ind)
        except:
            return None

        return node
    
    def is_index_ok(self, node, collision_check_step: int) -> bool:
        """
        check if the node is legal for a three trailer system
        - node: calc node (Node class)
        - P: parameters
        returns:
        whether the current node is ok
        """
        # check node index
        # check whether to go outside
        # TODO: check whether this is necessary
        if node.xind <= self.P.minx or \
                node.xind >= self.P.maxx or \
                node.yind <= self.P.miny or \
                node.yind >= self.P.maxy:
            return False

        ind = range(0, len(node.x), collision_check_step)

        x = [node.x[k] for k in ind]
        y = [node.y[k] for k in ind]
        yaw = [node.yaw[k] for k in ind]
        yawt1 = [node.yawt1[k] for k in ind]
        yawt2 = [node.yawt2[k] for k in ind]
        yawt3 = [node.yawt3[k] for k in ind]

        if self.is_collision(x, y, yaw, yawt1, yawt2, yawt3):
            return False

        return True
    
    def is_collision(self, x, y, yaw, yawt1, yawt2, yawt3) -> bool:
        '''
        check whether there is collision
        Inputs:
        x, y, yaw, yawt1, yawt2, yawt3: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        for ix, iy, iyaw, iyawt1, iyawt2, iyawt3 in zip(x, y, yaw, yawt1, yawt2, yawt3):
            # first trailer test collision
            d = self.safe_d
            deltal1 = (self.vehicle.RTF + self.vehicle.RTB) / 2.0 #which is exactly C.RTR
            rt1 = (self.vehicle.RTB - self.vehicle.RTF) / 2.0 + d #half length of trailer1 plus d

            ctx1 = ix - deltal1 * math.cos(iyawt1)
            cty1 = iy - deltal1 * math.sin(iyawt1)

            idst1 = self.P.kdtree.query_ball_point([ctx1, cty1], rt1)

            if idst1:
                for i in idst1:
                    xot1 = self.P.ox[i] - ctx1
                    yot1 = self.P.oy[i] - cty1

                    dx_trail1 = xot1 * math.cos(iyawt1) + yot1 * math.sin(iyawt1)
                    dy_trail1 = -xot1 * math.sin(iyawt1) + yot1 * math.cos(iyawt1)

                    if abs(dx_trail1) <= rt1 and \
                            abs(dy_trail1) <= self.vehicle.W / 2.0 + d:
                        return True
            # check the second trailer collision
            deltal2 = (self.vehicle.RTF2 + self.vehicle.RTB2) / 2.0
            rt2 = (self.vehicle.RTB2 - self.vehicle.RTF2) / 2.0 + d
            
            ctx2 = ctx1 - deltal2 * math.cos(iyawt2)
            cty2 = cty1 - deltal2 * math.sin(iyawt2)
            
            idst2 = self.P.kdtree.query_ball_point([ctx2, cty2], rt2)
            
            if idst2:
                for i in idst2:
                    xot2 = self.P.ox[i] - ctx2
                    yot2 = self.P.oy[i] - cty2
                    
                    dx_trail2 = xot2 * math.cos(iyawt2) + yot2 * math.sin(iyawt2)
                    dy_trail2 = -xot2 * math.cos(iyawt2) + yot2 * math.cos(iyawt2)
                    
                    if abs(dx_trail2) <= rt2 and \
                        abs(dy_trail2) <= self.vehicle.W / 2.0 + d:
                            return True
                        
            # check the third trailer collision
            deltal3 = (self.vehicle.RTF3 + self.vehicle.RTB3) / 2.0
            rt3 = (self.vehicle.RTB3 - self.vehicle.RTF3) / 2.0 + d
            
            ctx3 = ctx2 - deltal3 * math.cos(iyawt3)
            cty3 = cty2 - deltal3 * math.sin(iyawt3)
            
            idst3 = self.P.kdtree.query_ball_point([ctx3, cty3], rt3)
            
            if idst3:
                for i in idst3:
                    xot3 = self.P.ox[i] - ctx3
                    yot3 = self.P.oy[i] - cty3
                    
                    dx_trail3 = xot3 * math.cos(iyawt3) + yot3 * math.sin(iyawt3)
                    dy_trail3 = -xot3 * math.cos(iyawt3) + yot3 * math.cos(iyawt3)
                    
                    if abs(dx_trail3) <= rt3 and \
                        abs(dy_trail3) <= self.vehicle.W / 2.0 + d:
                            return True
                        
            # check the tractor collision
            deltal = (self.vehicle.RF - self.vehicle.RB) / 2.0
            rc = (self.vehicle.RF + self.vehicle.RB) / 2.0 + d

            cx = ix + deltal * math.cos(iyaw)
            cy = iy + deltal * math.sin(iyaw)

            ids = self.P.kdtree.query_ball_point([cx, cy], rc)

            if ids:
                for i in ids:
                    xo = self.P.ox[i] - cx
                    yo = self.P.oy[i] - cy

                    dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                    dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                    if abs(dx_car) <= rc and \
                            abs(dy_car) <= self.vehicle.W / 2.0 + d:
                        return True

        return False
    
    def calc_all_paths_modify(self, node, ngoal, maxc, step_size):
        # Fank: 
        # Input: node - start node
        #        nogal - goal node
        #        maxc - maximum culvature
        # newly add control list when exploring
        # newly add control list when exploring
        sx, sy, syaw, syawt1, syawt2, syawt3 = node.x[-1], node.y[-1], node.yaw[-1], node.yawt1[-1], node.yawt2[-1], node.yawt3[-1]
        gx, gy, gyaw, gyawt1, gyawt2, gyawt3 = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1], ngoal.yawt1[-1], ngoal.yawt2[-1], ngoal.yawt3[-1]
        q0 = [sx, sy, syaw]
        q1 = [gx, gy, gyaw]
        input = np.array([sx, sy, syaw, syawt1, syawt2, syawt3])
        goal = np.array([gx, gy, gyaw, gyawt1, gyawt2, gyawt3])

        paths = curves_generator.generate_path(q0, q1, maxc)

        for path in paths:
            rscontrol_list = extract_rs_path_control(path, self.vehicle.MAX_STEER, maxc)
            control_list = action_recover_from_planner(rscontrol_list)
            path.x, path.y, path.yaw, path.yawt1, path.yawt2, path.yawt3, path.directions, path.valid = self.forward_simulation_three_trailer(input, goal, control_list)
            path.lengths = [l / maxc for l in path.lengths]
            path.L = path.L / maxc
            # add rscontrollist once search the path
            path.rscontrollist = rscontrol_list
            

        return paths
    
    def calc_all_paths_simplified(self, node, ngoal, maxc):
        # Fank: 
        # Input: node - start node
        #        nogal - goal node
        #        maxc - maximum culvature
        # this function adds more information for the rspath we selected
        
        sx, sy, syaw, syawt1, syawt2, syawt3 = node.x[-1], node.y[-1], node.yaw[-1], node.yawt1[-1], node.yawt2[-1], node.yawt3[-1]
        gx, gy, gyaw, gyawt1, gyawt2, gyawt3 = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1], ngoal.yawt1[-1], ngoal.yawt2[-1], ngoal.yawt3[-1]
        q0 = [sx, sy, syaw]
        q1 = [gx, gy, gyaw]
        input = np.array([sx, sy, syaw, syawt1, syawt2, syawt3])
        goal = np.array([gx, gy, gyaw, gyawt1, gyawt2, gyawt3])

        paths = curves_generator.generate_path(q0, q1, maxc)

        for path in paths:
            rscontrol_list = extract_rs_path_control(path, self.vehicle.MAX_STEER, maxc, N_step=self.config["N_steps"], max_step_size=self.config["step_size"])
            control_list = action_recover_from_planner(rscontrol_list)
            path.x, path.y, path.yaw, path.yawt1, path.yawt2, path.yawt3, path.directions, path.info = self.forward_simulation_three_trailer_modify(input, goal, control_list)
            path.lengths = [l / maxc for l in path.lengths]
            path.L = path.L / maxc
            # add rscontrollist once search the path
            path.rscontrollist = rscontrol_list
            # put calc_rs_cost_here
            path.rscost = self.calc_rs_path_cost_three_trailer_modify(path)
            path.stepcost = len(path.x)
            # Fank: check here if there is jack_knife
            if path.info["accept"] and (not path.info["collision"]) and (not path.info["jack_knife"]):    
                xind = round(path.x[-1] / self.xyreso)
                yind = round(path.y[-1] / self.xyreso)
                yawind = round(path.yaw[-1] / self.yawreso)
                direction = path.directions[-1]
                fpind =  self.calc_index(node) 
                fcost = node.cost + path.rscost
                fx = path.x[1:]
                fy = path.y[1:]
                fyaw = path.yaw[1:]
                fyawt1 = path.yawt1[1:]
                fyawt2 = path.yawt2[1:]
                fyawt3 = path.yawt3[1:]
                fd = []
                for d in path.directions[1:]:
                    if d >= 0:
                        fd.append(1.0)
                    else:
                        fd.append(-1.0)
                fsteer = 0.0
                
                final_node = hyastar.Node_three_trailer(self.vehicle, xind, yind, yawind, direction,
                    fx, fy, fyaw, fyawt1, fyawt2, fyawt3, fd, fsteer, fcost, fpind)
                path.info["final_node"] = final_node

        return paths
    
    def forward_simulation_three_trailer(self, input, goal, control_list, simulation_freq=10):
        # Fank: use the rs_path control we extract to forward simulation to 
        # check whether suitable this path
        # control_list: clip to [-1,1]
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
        path_x_list, path_y_list, path_yaw_list, path_yawt1_list = [], [], [], []
        path_yawt2_list, path_yawt3_list = [], []
        directions = []
        controlled_vehicle = tt_envs.ThreeTrailer(config_dict)
        controlled_vehicle.reset(*input)
        path_x, path_y, path_yaw, path_yawt1, path_yawt2, path_yawt3 = controlled_vehicle.state
        path_x_list.append(path_x)
        path_y_list.append(path_y)
        path_yaw_list.append(path_yaw)
        path_yawt1_list.append(path_yawt1)
        path_yawt2_list.append(path_yawt2)
        path_yawt3_list.append(path_yawt3)
        for action_clipped in control_list:
            if action_clipped[0] > 0:
                directions.append(1)
            else:
                directions.append(-1)
            controlled_vehicle.step(action_clipped, 1 / simulation_freq)
            path_x, path_y, path_yaw, path_yawt1, path_yawt2, path_yawt3 = controlled_vehicle.state
            path_x_list.append(path_x)
            path_y_list.append(path_y)
            path_yaw_list.append(path_yaw)
            path_yawt1_list.append(path_yawt1)
            path_yawt2_list.append(path_yawt2)
            path_yawt3_list.append(path_yawt3)
            
        directions.append(directions[-1])
        final_state = np.array(controlled_vehicle.state)
        # distance_error = np.linalg.norm(goal - final_state)
        distance_error = mixed_norm(goal, final_state)
        if distance_error > self.config["acceptance_error"]:
            info = False
        else:
            
            info = True
        return path_x_list, path_y_list, path_yaw_list, path_yawt1_list, path_yawt2_list, path_yawt3_list, directions, info
    
    def forward_simulation_three_trailer_modify(self, input, goal, control_list, simulation_freq=10):
        # Fank: use the rs_path control we extract to forward simulation to 
        # check whether suitable this path
        # control_list: clip to [-1,1]
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
        path_x_list, path_y_list, path_yaw_list, path_yawt1_list = [], [], [], []
        path_yawt2_list, path_yawt3_list = [], []
        directions = []
        controlled_vehicle = tt_envs.ThreeTrailer(config_dict)
        controlled_vehicle.reset(*input)
        path_x, path_y, path_yaw, path_yawt1, path_yawt2, path_yawt3 = controlled_vehicle.state
        path_x_list.append(path_x)
        path_y_list.append(path_y)
        path_yaw_list.append(path_yaw)
        path_yawt1_list.append(path_yawt1)
        path_yawt2_list.append(path_yawt2)
        path_yawt3_list.append(path_yawt3)
        for action_clipped in control_list:
            if action_clipped[0] > 0:
                directions.append(1)
            else:
                directions.append(-1)
            controlled_vehicle.step(action_clipped, 1 / simulation_freq)
            path_x, path_y, path_yaw, path_yawt1, path_yawt2, path_yawt3 = controlled_vehicle.state
            path_x_list.append(path_x)
            path_y_list.append(path_y)
            path_yaw_list.append(path_yaw)
            path_yawt1_list.append(path_yawt1)
            path_yawt2_list.append(path_yawt2)
            path_yawt3_list.append(path_yawt3)
            
        directions.append(directions[-1])
        final_state = np.array(controlled_vehicle.state)
        # distance_error = np.linalg.norm(goal - final_state)
        distance_error = mixed_norm(goal, final_state)
        # Fank: accept(false means not good)
        #       collision(false means no collision)
        #       jack_knife(false means no jack_knife)
        info = {
            "accept": False,
            "collision": None,
            "jack_knife": None,
        }
        if distance_error > self.config["acceptance_error"]:
            info["accept"] = False
        else:
            info["accept"] = True
        
        
        # Fank: check whether collision here
        # Always perform collision check, regardless of the acceptance
        ind = range(0, len(path_x_list), int(self.config["collision_check_step"]/4))
        pathx = [path_x_list[k] for k in ind]
        pathy = [path_y_list[k] for k in ind]
        pathyaw = [path_yaw_list[k] for k in ind]
        pathyawt1 = [path_yawt1_list[k] for k in ind]
        pathyawt2 = [path_yawt2_list[k] for k in ind]
        pathyawt3 = [path_yawt3_list[k] for k in ind]
        if self.is_collision(pathx, pathy, pathyaw, pathyawt1, pathyawt2, pathyawt3):
            info["collision"] = True
        else:
            # no collision
            info["collision"] = False
            
            
        # Check for jack-knife
        for yaw, yawt1, yawt2, yawt3 in zip(path_yaw_list, path_yawt1_list, path_yawt2_list, path_yawt3_list):
            if abs(yaw - yawt1) >= config_dict["xi_max"] or abs(yawt1 - yawt2) >= config_dict["xi_max"] or abs(yawt2 - yawt3) >= config_dict["xi_max"]:
                info["jack_knife"] = True
                break
        else:
            info["jack_knife"] = False
        
        return path_x_list, path_y_list, path_yaw_list, path_yawt1_list, path_yawt2_list, path_yawt3_list, directions, info
    
    
    def is_same_grid(self, node1, node2):
        """
        whether the two nodes are on the same grid
        """
        if node1.xind != node2.xind or \
                node1.yind != node2.yind or \
                node1.yawind != node2.yawind:
            return False

        return True
    
    def is_the_start(self, node1, node2):
        """
        whether the two nodes are all start node
        """
        if len(node1.x) == 1 and len(node2.y) == 1:
            return True
        return False
    
    def analystic_expantion_modify(self, node, ngoal):
        """
        the returned path contains the start and the end
        which is also admissible(TT configuration and no collision)
        """
        # Fank
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        # I add a new attribute to this function 
        paths = self.calc_all_paths_modify(node, ngoal, maxc, step_size=self.step_size)

        if not paths:
            return None

        pq = hyastar.QueuePrior()
        for path in paths:
            if not path.valid:
                continue
            pq.put(path, self.calc_rs_path_cost_three_trailer_modify(path))
            # pq.put(path, calc_rs_path_cost_one_trailer(path, yawt1))

        while not pq.empty():
            path = pq.get()
            # check whether collision
            ind = range(0, len(path.x), self.config["collision_check_step"])
            pathx = [path.x[k] for k in ind]
            pathy = [path.y[k] for k in ind]
            pathyaw = [path.yaw[k] for k in ind]
            pathyawt1 = [path.yawt1[k] for k in ind]
            pathyawt2 = [path.yawt2[k] for k in ind]
            pathyawt3 = [path.yawt3[k] for k in ind]

            if not self.is_collision(pathx, pathy, pathyaw, pathyawt1, pathyawt2, pathyawt3):
                return path

        return None
    
    
    def update_node_with_analystic_expantion_modify(self, n_curr, ngoal):
        """
        find a admissible rs path for three trailer system
        this is a modified version
        Inputs:
        - n_curr: curr node
        - ngoal: goal node
        - P: parameters
        Return:
        - flag: Boolean whether we find a admissible path
        - fpath: a node from n_curr -> ngoal(contains ngoal configuration not n_curr configuration)
        """
        # now the returnd path has a new attribute rscontrollist
        # return the waypoints and control_list all in path
        path = self.analystic_expantion_modify(n_curr, ngoal)  # rs path: n -> ngoal

        if not path:
            return False, None, None, None

        
        fx = path.x[1:]
        fy = path.y[1:]
        fyaw = path.yaw[1:]
        fyawt1 = path.yawt1[1:]
        fyawt2 = path.yawt2[1:]
        fyawt3 = path.yawt3[1:]

        fd = []
        for d in path.directions[1:]:
            if d >= 0:
                fd.append(1.0)
            else:
                fd.append(-1.0)
        fsteer = 0.0
        # fd = path.directions[1:-1]

        fcost = n_curr.cost + self.calc_rs_path_cost_three_trailer_modify(path)
        # fcost = n_curr.cost + calc_rs_path_cost_one_trailer(path, yawt1)
        fpind = self.calc_index(n_curr)

        try:
            #here n_curr.xind might be wrong
            #but doesn't matter
            fpath = hyastar.Node_three_trailer(self.vehicle, ngoal.xind, ngoal.yind, ngoal.yawind, ngoal.direction,
                        fx, fy, fyaw, fyawt1, fyawt2, fyawt3, fd, fsteer, fcost, fpind)
        except:
            return False, None, None, None
        # abandon the first method
        return True, fpath, path.rscontrollist, path
    
    def rs_gear(self, node, ngoal):
        # Fank: put all rs related tech here
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        # I add a new attribute to this function 
        # Using a simplified version of calc_all_paths
        paths = self.calc_all_paths_simplified(node, ngoal, maxc)
        
        
        find_feasible = False
        if not paths:
            return find_feasible, None
        pq = hyastar.QueuePrior()
        
        for path in paths:
            if (path.info["jack_knife"] == False) and (path.info["accept"] == True) and (path.info["collision"] == False):
                find_feasible = True
                return find_feasible, path
            #TODO: I change rscost to stepcost for debug test
            pq.put(path, path.rscost)
        #TODO: may have to adjust
        while not pq.empty():
            path = pq.get()
            find_feasible = False
            return find_feasible, path
    
    def extract_path_and_control(self, closed, ngoal, nstart, reverse=False, find_rs_path=True, find_rl_path=True):
        """
        extract the path before rs path
        notice that there will be some unavoidable mistakes
        - find_rs_path: whether we find rs path (always yes)
        return:
        all the path point
        and all the control before rs path or rl path
        """
        rx, ry, ryaw, ryawt1, ryawt2, ryawt3, direc = [], [], [], [], [], [], []
        expand_control_list = []
        # TODO: here you made a mistake
        step = self.config["mp_step"]
        nlist = math.ceil(step / self.config["move_step"])
        # cost = 0.0
        node = ngoal
        count = 0
        
        while True:
            #append the current node state configuration
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            ryawt1 += node.yawt1[::-1]
            ryawt2 += node.yawt2[::-1]
            ryawt3 += node.yawt3[::-1]
            direc += node.directions[::-1]
            # cost += node.cost

            if self.is_the_start(node, nstart) and self.is_same_grid(node, nstart):
                break
            if find_rs_path or find_rl_path:
                if count > 0: #which means this is definitely not rs path
                    for i in range(nlist):
                        expand_control_list.append(np.array([node.directions[-1] * self.step_size, node.steer]))       
            else:
                for i in range(nlist):
                    expand_control_list.append(np.array([node.directions[-1] * self.step_size, node.steer
                                                        ]))
            # tracking parent ind
            node = closed[node.pind]
            count += 1
        if not reverse:
            rx = rx[::-1]
            ry = ry[::-1]
            ryaw = ryaw[::-1]
            ryawt1 = ryawt1[::-1]
            ryawt2 = ryawt2[::-1]
            ryawt3 = ryawt3[::-1]
            direc = direc[::-1]
            direc[0] = direc[1]
        
        if self.config["plot_final_path"]:
            self.plot_real_path(rx, ry)
            save_dir = './planner_result/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            files = os.listdir(save_dir)
            
            file_index = 0
            file_name = f"hybrid_expand_tree_three_trailer_{file_index}.png"
            while file_name in files:
                file_index += 1
                file_name = f"hybrid_expand_tree_three_trailer_{file_index}.png"
            plt.savefig(os.path.join(save_dir, file_name))
        plt.close()  
         
        path = hyastar.Path_three_trailer(rx, ry, ryaw, ryawt1, ryawt2, ryawt3, direc)
        expand_control_list = expand_control_list[::-1]
        
        return path, expand_control_list
    
    def extract_path(self, closed, ngoal, nstart, reverse=False):
        """
        extract the final path
        closed: closed_set (dictionary: key is node_ind, value is node class)
        ngoal: goal node class
        nstart: start node class
        reverse: whether to reverse or not
        
        returns:
        path class
        """
        rx, ry, ryaw, ryawt1, ryawt2, ryawt3, direc = [], [], [], [], [], [], []
        # cost = 0.0
        node = ngoal
        
        while True:
            #append the current node state configuration
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            ryawt1 += node.yawt1[::-1]
            ryawt2 += node.yawt2[::-1]
            ryawt3 += node.yawt3[::-1]
            direc += node.directions[::-1]
            # cost += node.cost

            if self.is_same_grid(node, nstart):
                break
            # tracking parent ind
            node = closed[node.pind]
        if not reverse:
            rx = rx[::-1]
            ry = ry[::-1]
            ryaw = ryaw[::-1]
            ryawt1 = ryawt1[::-1]
            ryawt2 = ryawt2[::-1]
            ryawt3 = ryawt3[::-1]
            direc = direc[::-1]
            direc[0] = direc[1]
        path = hyastar.Path_three_trailer(rx, ry, ryaw, ryawt1, ryawt2, ryawt3, direc)

        return path
    
    def visualize_hmap(self, hmap):
        # x = ngoal.x[-1]
        # y = ngoal.y[-1]
        # yaw = ngoal.yaw[-1]
        # yawt1 = ngoal.yawt1[-1]
        # yawt2 = ngoal.yawt2[-1]
        # yawt3 = ngoal.yawt3[-1]
        
        # ox, oy = map_env1()
        # define your map
        hmap = np.where(np.isinf(hmap), np.nan, hmap)


        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        hmap_for_vis = np.flip(np.transpose(hmap), 0)
        
        # cax = ax.imshow(hmap_for_vis, cmap="inferno", extent=[-1, 41, -1, 41], aspect='auto')

        # # 设置轴的范围
        # ax.set_xlim(0, 40)
        # ax.set_ylim(0, 40)
        #differnt map and resolution set differently
        # x_min = 0 - (reso / 2)
        # x_max = 29 + (reso / 2)
        # y_min = 0 - (reso / 2)
        # y_max = 36 + (reso / 2)
        
        x_min = min(self.ox) - (self.heuristic_reso / 2)
        x_max = max(self.oy) + (self.heuristic_reso / 2)
        y_min = min(self.oy) - (self.heuristic_reso / 2)
        y_max = max(self.oy) + (self.heuristic_reso / 2)
        
        # cax = ax.imshow(hmap_for_vis, cmap="inferno", extent=[-1, 29, -5, 37], aspect='auto')
        cax = ax.imshow(hmap_for_vis, cmap="jet", extent=[x_min, x_max, y_min, y_max], aspect='auto')

        # 设置轴的范围
        # ax.set_xlim(0, 29)
        # ax.set_ylim(0, 36)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 添加颜色条
        cbar = fig.colorbar(cax)
        # draw_model_three_trailer(TT, x, y, yaw, yawt1, yawt2, yawt3, 0.0, 'black')
        plt.plot(self.ox, self.oy, 'sk', markersize=1)

        plt.savefig('hmap_challenge_cases.png')
        
    def plan(self, start:np.ndarray, goal:np.ndarray, get_control_sequence:bool, verbose=False, *args, **kwargs):
        """
        Main Planning Algorithm for 3-tt systems
        :param start: starting point (np_array)
        :param goal: goal point (np_array)
        - path: all the six-dim state along the way (using extract function)
        - rs_path: contains the rspath and rspath control list
        - control list: rspath control list + expand control list
        """
        
        self.sx, self.sy, self.syaw, self.syawt1, self.syawt2, self.syawt3 = start
        self.gx, self.gy, self.gyaw, self.gyawt1, self.gyawt2, self.gyawt3 = goal
        self.syaw, self.syawt1, self.syawt2, self.syawt3 = self.pi_2_pi(self.syaw), self.pi_2_pi(self.syawt1), self.pi_2_pi(self.syawt2), self.pi_2_pi(self.syawt3)
        self.gyaw, self.gyawt1, self.gyawt2, self.gyawt3 = self.pi_2_pi(self.gyaw), self.pi_2_pi(self.gyawt1), self.pi_2_pi(self.gyawt2), self.pi_2_pi(self.gyawt3)
        self.sxr, self.syr = round(self.sx / self.xyreso), round(self.sy / self.xyreso)
        self.gxr, self.gyr = round(self.gx / self.xyreso), round(self.gy / self.xyreso)
        self.syawr = round(self.syaw / self.yawreso)
        self.gyawr = round(self.gyaw / self.yawreso)
        
        if self.heuristic_type == "critic":
            with open(f'HybridAstarPlanner/supervised_data.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            # inputs_tensor = torch.tensor(data_dict['inputs'], dtype=torch.float32)
            labels_tensor = torch.tensor(data_dict['labels'], dtype=torch.float32)
            # inputs = data_dict['inputs']
            # labels = data_dict['labels']
            labels_mean = labels_tensor.mean()
            labels_std = labels_tensor.std()
            # labels_normalized = (labels_tensor - labels_mean) / labels_std
        # put the class in this file
        nstart = hyastar.Node_three_trailer(self.vehicle, self.sxr, self.syr, self.syawr, 1, \
            [self.sx], [self.sy], [self.pi_2_pi(self.syaw)], [self.pi_2_pi(self.syawt1)], [self.pi_2_pi(self.syawt2)], [self.pi_2_pi(self.syawt3)], [1], 0.0, 0.0, -1)
        ngoal = hyastar.Node_three_trailer(self.vehicle, self.gxr, self.gyr, self.gyawr, 1, \
            [self.gx], [self.gy], [self.pi_2_pi(self.gyaw)], [self.pi_2_pi(self.gyawt1)], [self.pi_2_pi(self.gyawt2)], [self.pi_2_pi(self.gyawt3)], [1], 0.0, 0.0, -1)
        if not self.is_index_ok(nstart, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        if not self.is_index_ok(ngoal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        # TODO: change the api (seems this one is better)
        if self.obs:
            self.hmap = hyastar.calc_holonomic_heuristic_with_obstacle(ngoal, self.P.ox, self.P.oy, self.heuristic_reso, self.heuristic_rr)
        if self.config["plot_heuristic_nonholonomic"]:
            self.visualize_hmap(self.hmap)
        
        
        steer_set, direc_set = self.calc_motion_set()
        # change the way to calc_index
        open_set, closed_set = {self.calc_index(nstart): nstart}, {}
        # non_h = calc_non_holonomic_heuritstic(nstart, ngoal, gyawt1, gyawt2, gyawt3, fP)
        
        # reset qp for next using
        self.qp.reset()
        # the main change here will be the heuristic value
        if self.heuristic_type == "traditional":
            self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_new(nstart, ngoal))
        else:
            # here you need to change this heuristic
            self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_new_critic(nstart, labels_mean, labels_std))
        # an indicator whether find the rs path at last
        find_rs_path = False
        # the loop number for analystic expansion
        count = 0
        # Main Loop
        while True:
            # I will try not to use this
            # may need to modify when there's obstacle
            if not open_set or self.qp.empty():
                # consider this function
                print("failed finding a feasible path")
                self.extract_failed_path(closed_set, nstart)
                return None, None, None
            count += 1
            # add if the loop's too much
            if count > self.max_iter:
                print("waste a long time to find")
                return None, None, None
            ind = self.qp.get()
            n_curr = open_set[ind]
            closed_set[ind] = n_curr
            open_set.pop(ind)
            
            # key and the most tricky part of the algorithm
            update, fpath, rs_control_list, rs_path = self.update_node_with_analystic_expantion_modify(n_curr, ngoal)

            if update:
                fnode = fpath
                find_rs_path = True
                if self.config["plot_expand_tree"]:
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    plot_rs_path(rs_path, self.ox, self.oy)
                    # plt.close()
                if verbose:
                    print("final analystic expantion node number:", count)
                break
            
            for i in range(len(steer_set)):
                node = self.calc_next_node(n_curr, ind, steer_set[i], direc_set[i])
                if not node:
                    continue
                if not self.is_index_ok(node, self.config["collision_check_step"]):
                    continue
                node_ind = self.calc_index(node)
                if node_ind in closed_set:
                    continue
                if node_ind not in open_set:
                    open_set[node_ind] = node
                    if self.heuristic_type == "traditional":
                        self.qp.put(node_ind, self.calc_hybrid_cost_new(node, ngoal))
                    else:
                        self.qp.put(node_ind, self.calc_hybrid_cost_new_critic(node, ngoal, labels_mean, labels_std))
                else:
                    if open_set[node_ind].cost > node.cost:
                        open_set[node_ind] = node
                        if self.qp_type == "heapdict":  
                            # if using heapdict, here you can modify the value
                            if self.heuristic_type == "traditional":
                                self.qp.queue[node_ind] = self.calc_hybrid_cost_new(node, ngoal)
                            else:
                                self.qp.queue[node_ind] = self.calc_hybrid_cost_new_critic(node, ngoal, labels_mean, labels_std)           
        if verbose:
            print("final expand node: ", len(open_set) + len(closed_set))
        
        if get_control_sequence:
            path, expand_control_list = self.extract_path_and_control(closed_set, fnode, nstart,find_rs_path=find_rs_path)
            if find_rs_path:
                all_control_list = expand_control_list + rs_control_list
            else:
                rs_path = None
                all_control_list = all_control_list
            return path, all_control_list, rs_path
        else:
            if find_rs_path: 
                return self.extract_path(closed_set, fnode, nstart), None, rs_path
            else:
                return self.extract_path(closed_set, fnode, nstart), None, None
    def calc_rl_path_cost_three_trailer(self, rlpath) -> float:
        cost = 0.0
        action_list = rlpath.rlcontrollist
        for j in range(len(action_list)):
            if action_list[j][0] > 0:
                cost += abs(action_list[j][0])
            else:
                cost += abs(action_list[j][0]) * self.cost["backward_cost"]
            cost += self.cost["steer_angle_cost"] * abs(action_list[j][1])
            if j > 0:
                if action_list[j][0] * action_list[j - 1][0] < 0:
                    cost += self.cost["gear_cost"]
                cost += self.cost["steer_change_cost"] * abs(action_list[j][1] - action_list[j - 1][1])
        
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y)) \
            for x, y in zip(rlpath.yaw[1:], rlpath.yawt1[1:])])
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y)) \
            for x, y in zip(rlpath.yawt1[1:], rlpath.yawt2[1:])])
        cost += self.cost["scissors_cost"] * sum([abs(self.pi_2_pi(x - y)) \
            for x, y in zip(rlpath.yawt2[1:], rlpath.yawt3[1:])])
                
        return cost
    
    def calc_euclidean_distance(self, n_curr, n_goal):
        n_curr_x = n_curr.x[-1]
        n_curr_y = n_curr.y[-1]
        n_curr_yaw = n_curr.yaw[-1]
        n_curr_yawt1 = n_curr.yawt1[-1]
        n_curr_yawt2 = n_curr.yawt2[-1]
        n_curr_yawt3 = n_curr.yawt3[-1]
        curr_state = np.array([n_curr_x, n_curr_y, n_curr_yaw, n_curr_yawt1, n_curr_yawt2, n_curr_yawt3], dtype=np.float32)
        n_goal_x = n_goal.x[-1]
        n_goal_y = n_goal.y[-1]
        n_goal_yaw = n_goal.yaw[-1]
        n_goal_yawt1 = n_goal.yawt1[-1]
        n_goal_yawt2 = n_goal.yawt2[-1]
        n_goal_yawt3 = n_goal.yawt3[-1]
        goal_state = np.array([n_goal_x, n_goal_y, n_goal_yaw, n_goal_yawt1, n_goal_yawt2, n_goal_yawt3], dtype=np.float32)
        return np.linalg.norm(curr_state - goal_state)
    
    
    def rl_gear(self, n_curr, n_goal, obstacles_info=None, map_vertices=None,  max_step=60, terminated=0.5):
        """In this function we use meta env as a simulated env
        As a result, we should change the obstacles_info to local coordinates
        TODO: remains to be fixed
        """
        rl_path = RlPath()
        # use rl agent to guide our search
        n_curr_x = n_curr.x[-1]
        n_curr_y = n_curr.y[-1]
        n_curr_yaw = n_curr.yaw[-1]
        n_curr_yawt1 = n_curr.yawt1[-1]
        n_curr_yawt2 = n_curr.yawt2[-1]
        n_curr_yawt3 = n_curr.yawt3[-1]
        n_goal_x = n_goal.x[-1]
        n_goal_y = n_goal.y[-1]
        n_goal_yaw = n_goal.yaw[-1]
        n_goal_yawt1 = n_goal.yawt1[-1]
        n_goal_yawt2 = n_goal.yawt2[-1]
        n_goal_yawt3 = n_goal.yawt3[-1]
        
        
        # Apply Coordinate rotation: global -> local
        n_goal_x_local = np.cos(n_curr_yaw) * (n_goal_x - n_curr_x) + np.sin(n_curr_yaw) * (n_goal_y - n_curr_y)
        n_goal_y_local = -np.sin(n_curr_yaw) * (n_goal_x - n_curr_x) + np.cos(n_curr_yaw) * (n_goal_y - n_curr_y)
        start = np.array([0.0, 0.0, 0.0, n_curr_yawt1 - n_curr_yaw, n_curr_yawt2 - n_curr_yaw, n_curr_yawt3 - n_curr_yaw], dtype=np.float32)
        # test the "start from un-equili configuration" generalization
        goal = np.array([n_goal_x_local, n_goal_y_local, n_goal_yaw - n_curr_yaw, n_goal_yawt1 - n_curr_yaw, n_goal_yawt2 - n_curr_yaw, n_goal_yawt3 - n_curr_yaw],
                        dtype=np.float32)
        
        # global -> local obstacles_info
        local_obstacles_info = convert_obstacles_to_local(obstacles_info, n_curr_x, n_curr_y, n_curr_yaw)
        
        # global -> local map_vertices
        local_map_vertices = convert_map_vertices_to_local(map_vertices, n_curr_x, n_curr_y, n_curr_yaw)
        
        task_list = [
            {
                "start": start,
                "goal": goal,
                "obstacles_info": local_obstacles_info,
                "map_vertices": local_map_vertices,
            }
        ]
    
        # update the test_env to this goal
        self.agent.test_env.unwrapped.update_task_list(task_list)
        # give to the reaching env to simulate path
        o, info = self.agent.test_env.reset()
        # check whether accept this kind of obstacles
        # self.agent.test_env.render_jingyu()
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        # here we take out the truncated to test "go-further" generalization
        while not(terminated):
            # Take deterministic actions at test time 
            if self.heuristic_type == "mix_original" or self.heuristic_type == "rl":
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), deterministic=True)
            elif self.heuristic_type == "mix_original_with_obstacles_info":
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], process_obstacles_properties_to_array(info['obstacles_properties'])]), deterministic=True)
            elif self.heuristic_type == "mix_lidar_detection_one_hot":
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), deterministic=True)
            elif self.heuristic_type == "mix_lidar_detection_one_hot_triple":
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot_triple']]), deterministic=True)
            else: # attention
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), info, deterministic=True)
            o, r, terminated, truncated, info = self.agent.test_env.step(a)
            ep_ret += r
            ep_len += 1
            if (ep_len >= max_step):
                break
        # visualize the result
        if info['is_success'] == True:
            find_feasible = True
        else:
            find_feasible = False
        
        path_x = []
        path_y = []
        path_yaw = []
        path_yawt1 = []
        path_yawt2 = []
        path_yawt3 = []
        # local -> global
        for state in self.agent.test_env.unwrapped.state_list:
            x_global = np.cos(n_curr_yaw) * state[0] - np.sin(n_curr_yaw) * state[1] + n_curr_x
            y_global = np.sin(n_curr_yaw) * state[0] + np.cos(n_curr_yaw) * state[1] + n_curr_y
            yaw_global = state[2] + n_curr_yaw
            yawt1_global = state[3] + n_curr_yaw
            yawt2_global = state[4] + n_curr_yaw
            yawt3_global = state[5] + n_curr_yaw
            path_x.append(x_global)
            path_y.append(y_global)
            path_yaw.append(yaw_global)
            path_yawt1.append(yawt1_global)
            path_yawt2.append(yawt2_global)
            path_yawt3.append(yawt3_global)
        rl_path.x = path_x
        rl_path.y = path_y
        rl_path.yaw = path_yaw
        rl_path.yawt1 = path_yawt1
        rl_path.yawt2 = path_yawt2
        rl_path.yawt3 = path_yawt3
        rl_path.rlcontrollist = action_recover_to_planner(self.agent.test_env.unwrapped.action_list)
        rl_path.info = {
            "is_success": info["is_success"],
            "crashed": info["crashed"],
            "jack_knife": info["jack_knife"],
            "final_node": None,
        }
        
        # check collision from the real obstacle setting
        ind = range(0, len(rl_path.x), int(self.config["collision_check_step"]/4))
        
        pathx = [rl_path.x[k] for k in ind]
        pathy = [rl_path.y[k] for k in ind]
        pathyaw = [rl_path.yaw[k] for k in ind]
        pathyawt1 = [rl_path.yawt1[k] for k in ind]
        pathyawt2 = [rl_path.yawt2[k] for k in ind]
        pathyawt3 = [rl_path.yawt3[k] for k in ind]
        if self.is_collision(pathx, pathy, pathyaw, pathyawt1, pathyawt2, pathyawt3):
            rl_path.info["crashed"] = True
            rl_path.info["is_success"] = False
            find_feasible = False
        cost = self.calc_rl_path_cost_three_trailer(rl_path)
        rl_path.rlcost = cost
        if find_feasible:
            xind = round(path_x[-1] / self.xyreso)
            yind = round(path_y[-1] / self.xyreso)
            yawind = round(path_yaw[-1] / self.yawreso)
            fd = []
            for action in rl_path.rlcontrollist:
                if action[0] > 0:
                    fd.append(1)
                else:
                    fd.append(-1)
            fx = path_x[1:]
            fy = path_y[1:]
            fyaw = path_yaw[1:]
            fyawt1 = path_yawt1[1:]
            fyawt2 = path_yawt2[1:]
            fyawt3 = path_yawt3[1:]
            fsteer = 0.0
            fcost = n_curr.cost + cost
            fpind = self.calc_index(n_curr)
            direction = fd[-1]
            final_node = hyastar.Node_three_trailer(self.vehicle, xind, yind, yawind, direction,
                                                  fx, fy, fyaw, fyawt1, fyawt2, fyawt3, fd, fsteer, fcost, fpind)
            rl_path.info["final_node"] = final_node
        # test rl path animation
        # self.agent.test_env.unwrapped.run_simulation()
        return find_feasible, rl_path
    
    def new_rl_gear(self, n_curr, n_goal, obstacles_info=None, map_vertices=None, max_step=60, terminated=0.5):
        """
        This function uses the RL agent to guide the search directly in global coordinates.
        """
        rl_path = RlPath()
        
        n_curr_x = n_curr.x[-1]
        n_curr_y = n_curr.y[-1]
        n_curr_yaw = n_curr.yaw[-1]
        n_curr_yawt1 = n_curr.yawt1[-1]
        n_curr_yawt2 = n_curr.yawt2[-1]
        n_curr_yawt3 = n_curr.yawt3[-1]
        n_goal_x = n_goal.x[-1]
        n_goal_y = n_goal.y[-1]
        n_goal_yaw = n_goal.yaw[-1]
        n_goal_yawt1 = n_goal.yawt1[-1]
        n_goal_yawt2 = n_goal.yawt2[-1]
        n_goal_yawt3 = n_goal.yawt3[-1]

        # Fank: Notice here that we don't need to convert the global coordinates to local coordinates
        start = np.array([n_curr_x, n_curr_y, n_curr_yaw, n_curr_yawt1, n_curr_yawt2, n_curr_yawt3], dtype=np.float32)
        goal = np.array([n_goal_x, n_goal_y, n_goal_yaw, n_goal_yawt1, n_goal_yawt2, n_goal_yawt3], dtype=np.float32)

        task_list = [
            {
                "start": start,
                "goal": goal,
                "obstacles_info": obstacles_info,
                "map_vertices": map_vertices,
            }
        ]
    
        # Update the test_env with this task list
        self.agent.test_env.unwrapped.update_task_list(task_list)
        o, info = self.agent.test_env.reset()
        
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        while not terminated:
            if self.heuristic_type == "mix_original" or self.heuristic_type == "rl":
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), deterministic=True)
            elif self.heuristic_type == "mix_original_with_obstacles_info":
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], process_obstacles_properties_to_array(info['obstacles_properties'])]), deterministic=True)
            elif self.heuristic_type == "mix_lidar_detection_one_hot":
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot']]), deterministic=True)
            elif self.heuristic_type == "mix_lidar_detection_one_hot_triple":
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal'], o['lidar_detection_one_hot_triple']]), deterministic=True)
            else: # attention
                a = self.agent.get_action(np.concatenate([o['observation'], o['achieved_goal'], o['desired_goal']]), info, deterministic=True)
            o, r, terminated, truncated, info = self.agent.test_env.step(a)
            ep_ret += r
            ep_len += 1
            if ep_len >= max_step:
                break

        if info['is_success'] == True:
            find_feasible = True
        else:
            find_feasible = False
        
        path_x = []
        path_y = []
        path_yaw = []
        path_yawt1 = []
        path_yawt2 = []
        path_yawt3 = []
        
        for state in self.agent.test_env.unwrapped.state_list:
            x_global = state[0]
            y_global = state[1]
            yaw_global = state[2]
            yawt1_global = state[3]
            yawt2_global = state[4]
            yawt3_global = state[5]
            path_x.append(x_global)
            path_y.append(y_global)
            path_yaw.append(yaw_global)
            path_yawt1.append(yawt1_global)
            path_yawt2.append(yawt2_global)
            path_yawt3.append(yawt3_global)
        
        rl_path.x = path_x
        rl_path.y = path_y
        rl_path.yaw = path_yaw
        rl_path.yawt1 = path_yawt1
        rl_path.yawt2 = path_yawt2
        rl_path.yawt3 = path_yawt3
        rl_path.rlcontrollist = action_recover_to_planner(self.agent.test_env.unwrapped.action_list)
        rl_path.info = {
            "is_success": info["is_success"],
            "crashed": info["crashed"],
            "jack_knife": info["jack_knife"],
            "final_node": None,
        }
        
        ind = range(0, len(rl_path.x), int(self.config["collision_check_step"]/4))
        
        pathx = [rl_path.x[k] for k in ind]
        pathy = [rl_path.y[k] for k in ind]
        pathyaw = [rl_path.yaw[k] for k in ind]
        pathyawt1 = [rl_path.yawt1[k] for k in ind]
        pathyawt2 = [rl_path.yawt2[k] for k in ind]
        pathyawt3 = [rl_path.yawt3[k] for k in ind]
        if self.is_collision(pathx, pathy, pathyaw, pathyawt1, pathyawt2, pathyawt3):
            rl_path.info["crashed"] = True
            rl_path.info["is_success"] = False
            find_feasible = False
        
        cost = self.calc_rl_path_cost_three_trailer(rl_path)
        rl_path.rlcost = cost
        
        if find_feasible:
            xind = round(path_x[-1] / self.xyreso)
            yind = round(path_y[-1] / self.xyreso)
            yawind = round(path_yaw[-1] / self.yawreso)
            fd = [1 if action[0] > 0 else -1 for action in rl_path.rlcontrollist]
            fx = path_x[1:]
            fy = path_y[1:]
            fyaw = path_yaw[1:]
            fyawt1 = path_yawt1[1:]
            fyawt2 = path_yawt2[1:]
            fyawt3 = path_yawt3[1:]
            fsteer = 0.0
            fcost = n_curr.cost + cost
            fpind = self.calc_index(n_curr)
            direction = fd[-1]
            final_node = hyastar.Node_three_trailer(self.vehicle, xind, yind, yawind, direction,
                                                    fx, fy, fyaw, fyawt1, fyawt2, fyawt3, fd, fsteer, fcost, fpind)
            rl_path.info["final_node"] = final_node

        return find_feasible, rl_path
    
    def check_start_mp_feasible(self, start:np.ndarray) -> bool:
        """first check whether the start point is feasible 
        given a complex obstacles environment(not collision)
        """
        self.sx, self.sy, self.syaw, self.syawt1, self.syawt2, self.syawt3 = start
        self.syaw, self.syawt1, self.syawt2, self.syawt3 = self.pi_2_pi(self.syaw), self.pi_2_pi(self.syawt1), self.pi_2_pi(self.syawt2), self.pi_2_pi(self.syawt3)
        self.sxr, self.syr = round(self.sx / self.xyreso), round(self.sy / self.xyreso)
        self.syawr = round(self.syaw / self.yawreso)
        nstart = hyastar.Node_three_trailer(self.vehicle, self.sxr, self.syr, self.syawr, 1, \
            [self.sx], [self.sy], [self.pi_2_pi(self.syaw)], [self.pi_2_pi(self.syawt1)], [self.pi_2_pi(self.syawt2)], [self.pi_2_pi(self.syawt3)], [1], 0.0, 0.0, -1)
        if not self.is_index_ok(nstart, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
            
    
        steer_set, direc_set = self.calc_motion_set()
        whether_feasible = False
        for i in range(len(steer_set)):
            node = self.calc_next_node(nstart, self.calc_index(nstart), steer_set[i], direc_set[i])
            if not node:
                continue
            if self.is_index_ok(node, int(self.config["collision_check_step"]/4)):
                # not collision mp
                return True    
        return whether_feasible
    
    def plan_new_version(self, start:np.ndarray, goal:np.ndarray, get_control_sequence:bool, verbose=False, *args, **kwargs):
        """
        New Version of Main Planning Algorithm for 3-tt systems
        this algorithm saves some time
        :param start: starting point (np_array)
        :param goal: goal point (np_array)
        - path: all the six-dim state along the way (using extract function)
        - rs_path: contains the rspath and rspath control list
        - control list: rspath control list + expand control list
        """
        # input the given information
        if 'obstacles_info' in kwargs:
            # [] or [[(),(),(),()],...]
            obstacles_info = kwargs['obstacles_info']
        if 'map_vertices' in kwargs:
            map_vertices = kwargs["map_vertices"]
        
        self.sx, self.sy, self.syaw, self.syawt1, self.syawt2, self.syawt3 = start
        self.syaw, self.syawt1, self.syawt2, self.syawt3 = self.pi_2_pi(self.syaw), self.pi_2_pi(self.syawt1), self.pi_2_pi(self.syawt2), self.pi_2_pi(self.syawt3)
        self.sxr, self.syr = round(self.sx / self.xyreso), round(self.sy / self.xyreso)
        self.syawr = round(self.syaw / self.yawreso)
        nstart = hyastar.Node_three_trailer(self.vehicle, self.sxr, self.syr, self.syawr, 1, \
            [self.sx], [self.sy], [self.pi_2_pi(self.syaw)], [self.pi_2_pi(self.syawt1)], [self.pi_2_pi(self.syawt2)], [self.pi_2_pi(self.syawt3)], [1], 0.0, 0.0, -1)
        if not self.is_index_ok(nstart, self.config["collision_check_step"]):
            sys.exit("illegal start configuration")
        
        self.gx, self.gy, self.gyaw, self.gyawt1, self.gyawt2, self.gyawt3 = goal
        self.gyaw, self.gyawt1, self.gyawt2, self.gyawt3 = self.pi_2_pi(self.gyaw), self.pi_2_pi(self.gyawt1), self.pi_2_pi(self.gyawt2), self.pi_2_pi(self.gyawt3)
        self.gxr, self.gyr = round(self.gx / self.xyreso), round(self.gy / self.xyreso)
        self.gyawr = round(self.gyaw / self.yawreso)
        ngoal = hyastar.Node_three_trailer(self.vehicle, self.gxr, self.gyr, self.gyawr, 1, \
            [self.gx], [self.gy], [self.pi_2_pi(self.gyaw)], [self.pi_2_pi(self.gyawt1)], [self.pi_2_pi(self.gyawt2)], [self.pi_2_pi(self.gyawt3)], [1], 0.0, 0.0, -1)
        if not self.is_index_ok(ngoal, self.config["collision_check_step"]):
            sys.exit("illegal goal configuration")
        
        
        # calculate heuristic for obstacle
        if self.obs:
            self.hmap = hyastar.calc_holonomic_heuristic_with_obstacle(ngoal, self.P.ox, self.P.oy, self.heuristic_reso, self.heuristic_rr)
        if self.config["plot_heuristic_nonholonomic"]:
            self.visualize_hmap(self.hmap)
        
        
        steer_set, direc_set = self.calc_motion_set()
        # Initialize open_set and closed_set
        open_set, closed_set = {self.calc_index(nstart): nstart}, {}
        
        # reset qp for next using
        self.qp.reset()
        # an indicator whether find the rs path at last(for extract)
        find_rs_path = False
        # an indicator whether find the rl path at last(for extract)
        find_rl_path = False
        # the loop number for analystic expansion(counting number)
        count = 0
        # update parameter
        update = False
        # the main change here will be the heuristic value
        if self.heuristic_type == "traditional":
            find_feasible, path = self.rs_gear(nstart, ngoal)
            # if find feasible, then go to extract
            # else calculate heuristic
            if find_feasible:
                fnode = path.info["final_node"]
                find_rs_path = True
                update = find_feasible
                rs_path = path
                rs_control_list = path.rscontrollist
                if self.config["plot_expand_tree"]:
                    plot_rs_path(rs_path, self.ox, self.oy)
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    plt.savefig("rl_training/savefig.png")
                    plt.close() 
                    # plt.close()
                if verbose:
                    print("find path at first time via rs path")
                closed_set[self.calc_index(nstart)] = nstart
            else:
                self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_simplify(nstart, ngoal, path.rscost)) 
        elif self.heuristic_type == "rl":
            # TODO wait for RL to guide search
            t1 = time.time()
            find_feasible, path = self.new_rl_gear(nstart, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
            # find_feasible, path = self.rl_gear(nstart, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
            t2 = time.time()
            print("time for rl simulation:", t2 - t1)
            if find_feasible:
                fnode = path.info["final_node"]
                find_rl_path = True
                update = find_feasible
                rl_path = path
                rl_control_list = path.rlcontrollist
                if self.config["plot_expand_tree"]:
                    plot_rs_path(rl_path, self.ox, self.oy)
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    plt.savefig("rl_training/savefig.png")
                    plt.close()
                if verbose:
                    print("find path at first time via rl path")
                closed_set[self.calc_index(nstart)] = nstart
            else:
                # cost_qp = self.calc_euclidean_distance(nstart, ngoal)
                # self.qp.put(self.calc_index(nstart), cost_qp)
                self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_simplify(nstart, ngoal, path.rlcost))
        else:
            # Fank: Mixture of two gears, mix
            find_feasible, path = self.new_rl_gear(nstart, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
            # find_feasible, path = self.rl_gear(nstart, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
            if self.config["plot_expand_tree"]:
                plot_rs_path(path, self.ox, self.oy)
                self.plot_expand_tree(start, goal, closed_set, open_set)
                plt.savefig("rl_training/savefig.png")
                plt.close()
            if find_feasible:
                fnode = path.info["final_node"]
                find_rl_path = True
                update = find_feasible
                rl_path = path
                rl_control_list = path.rlcontrollist
                if self.config["plot_expand_tree"]:
                    plot_rs_path(rl_path, self.ox, self.oy)
                    self.plot_expand_tree(start, goal, closed_set, open_set)
                    plt.savefig("rl_training/savefig.png")
                    plt.close()
                if verbose:
                    print("find path at first time via rl path")
                closed_set[self.calc_index(nstart)] = nstart
            else:
                find_feasible, path = self.rs_gear(nstart, ngoal)
                if find_feasible:
                    fnode = path.info["final_node"]
                    find_rs_path = True
                    update = find_feasible
                    rs_path = path
                    rs_control_list = path.rscontrollist
                    if self.config["plot_expand_tree"]:
                        plot_rs_path(rs_path, self.ox, self.oy)
                        self.plot_expand_tree(start, goal, closed_set, open_set)
                        plt.savefig("rl_training/savefig.png")
                        plt.close() 
                        # plt.close()
                    if verbose:
                        print("find path at first time via rs path")
                    closed_set[self.calc_index(nstart)] = nstart
                else:
                    self.qp.put(self.calc_index(nstart), self.calc_hybrid_cost_simplify(nstart, ngoal, path.rscost)) 
                
        # Main Loop
        while True:
            if update:
                # use the flag update to break the main loop
                break
            if not open_set or self.qp.empty():
                print("failed finding a feasible path")
                if self.config["plot_failed_path"]:
                    self.extract_failed_path(closed_set, nstart)
                return None, None, None, None
            count += 1
            # add if the loop's too much
            if count > self.max_iter:
                if verbose:
                    print("waste a long time to find")
                if self.config["plot_failed_path"]:
                    self.extract_failed_path(start, goal, closed_set, nstart)
                return None, None, None, None
            
            ind = self.qp.get()
            n_curr = open_set[ind]
            closed_set[ind] = n_curr
            open_set.pop(ind)

            # expand tree using motion primitive
            for i in range(len(steer_set)):
                node = self.calc_next_node(n_curr, ind, steer_set[i], direc_set[i])
                if not node:
                    # encounter jack_knife
                    continue
                if not self.is_index_ok(node, int(self.config["collision_check_step"]/4)):
                    # check go outside or collision
                    continue
                node_ind = self.calc_index(node)
                if node_ind in closed_set:
                    # we will not calculate twice 
                    # Note that this can be a limitation
                    continue
                if node_ind not in open_set:
                    open_set[node_ind] = node
                    if self.heuristic_type == "traditional":
                        find_feasible, path = self.rs_gear(node, ngoal)
                        if find_feasible:
                            fnode = path.info["final_node"]
                            find_rs_path = True
                            update = find_feasible
                            rs_path = path
                            rs_control_list = path.rscontrollist
                            if self.config["plot_expand_tree"]:
                                plot_rs_path(rs_path, self.ox, self.oy)
                                self.plot_expand_tree(start, goal, closed_set, open_set)
                                # plt.close()
                            if verbose:
                                print("final expansion node number:", count)
                            # Here you need to add node to closed set
                            closed_set[node_ind] = node
                            # break the inner expand_tree loop
                            break
                        else:
                            self.qp.put(node_ind, self.calc_hybrid_cost_simplify(node, ngoal, path.rscost))
                    elif self.heuristic_type == "rl":
                        # wait for RL to guide search
                        t1 = time.time()
                        find_feasible, path = self.rl_gear(node, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
                        t2 = time.time()
                        print("time for rl simulation:", t2 - t1)
                        if find_feasible:
                            fnode = path.info["final_node"]
                            find_rl_path = True
                            update = find_feasible
                            rl_path = path
                            rl_control_list = path.rlcontrollist
                            # if self.config["plot_expand_tree"]:
                            #     plot_rs_path(rl_path, self.ox, self.oy)
                            #     self.plot_expand_tree(start, goal, closed_set, open_set)
                                # plt.close()
                            if verbose:
                                print("final expansion node number:", count)
                            closed_set[node_ind] = node
                            break
                        else:
                            # cost_qp = self.calc_euclidean_distance(node, ngoal)
                            # self.qp.put(node_ind, cost_qp)
                            self.qp.put(node_ind, self.calc_hybrid_cost_simplify(nstart, ngoal, path.rlcost))
                    else:
                        find_feasible, path = self.new_rl_gear(node, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
                        # find_feasible, path = self.rl_gear(node, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
                        if self.config["plot_expand_tree"]:
                            self.plot_expand_tree(start, goal, closed_set, open_set)
                            plot_rl_path(path, self.ox, self.oy)
                            plt.savefig("rl_training/savefig.png")
                            plt.close()
                        if find_feasible:
                            fnode = path.info["final_node"]
                            find_rl_path = True
                            update = find_feasible
                            rl_path = path
                            rl_control_list = path.rlcontrollist
                            # if self.config["plot_expand_tree"]:
                            #     plot_rs_path(rl_path, self.ox, self.oy)
                            #     self.plot_expand_tree(start, goal, closed_set, open_set)
                            #     plt.savefig("rl_training/savefig.png")
                            #     plt.close()
                            if verbose:
                                print("find via rl path")
                                print("final expansion node number:", count)
                            closed_set[node_ind] = node
                            break
                        else: 
                            find_feasible, path = self.rs_gear(node, ngoal)
                            if self.config["plot_expand_tree"]:
                                self.plot_expand_tree(start, goal, closed_set, open_set)
                                plot_rs_path(path, self.ox, self.oy)
                                plt.savefig("rl_training/savefig.png")
                                plt.close()
                            if find_feasible:
                                fnode = path.info["final_node"]
                                find_rs_path = True
                                update = find_feasible
                                rs_path = path
                                rs_control_list = path.rscontrollist
                                
                                if verbose:
                                    print("find via rs path")
                                    print("final expansion node number:", count)
                                # Here you need to add node to closed set
                                closed_set[node_ind] = node
                                # break the inner expand_tree loop
                                break
                            else:
                                self.qp.put(node_ind, self.calc_hybrid_cost_simplify(node, ngoal, path.rscost))
                    # if self.config["plot_expand_tree"]:
                    #     self.plot_expand_tree(start, goal, closed_set, open_set)
                    #     plot_rs_path(path, self.ox, self.oy)
                    #     plt.savefig("rl_training/savefig.png")
                    #     plt.close() 
                else:
                    if open_set[node_ind].cost > node.cost:
                        open_set[node_ind] = node
                        if self.qp_type == "heapdict":  
                            # if using heapdict, here you can modify the value
                            if self.heuristic_type == "traditional":
                                find_feasible, path = self.rs_gear(node, ngoal)
                                if find_feasible:
                                    fnode = path.info["final_node"]
                                    find_rs_path = True
                                    update = find_feasible
                                    rs_path = path
                                    rs_control_list = path.rscontrollist
                                    # if self.config["plot_expand_tree"]:
                                    #     plot_rs_path(rs_path, self.ox, self.oy)
                                    #     self.plot_expand_tree(start, goal, closed_set, open_set)
                                    #     # plt.close()
                                    if verbose:
                                        print("final expansion node number:", count)
                                    closed_set[node_ind] = node
                                    break
                                else:
                                    self.qp.queue[node_ind] = self.calc_hybrid_cost_simplify(node, ngoal, path.rscost)
                                    
                            elif self.heuristic_type == "rl":
                                t1 = time.time()
                                find_feasible, path = self.rl_gear(node, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
                                t2 = time.time()
                                print("time for rl simulation:", t2 - t1)
                                if find_feasible:
                                    fnode = path.info["final_node"]
                                    find_rl_path = True
                                    update = find_feasible
                                    rl_path = path
                                    rl_control_list = path.rlcontrollist
                                    # if self.config["plot_expand_tree"]:
                                    #     plot_rs_path(rl_path, self.ox, self.oy)
                                    #     self.plot_expand_tree(start, goal, closed_set, open_set)
                                    #     # plt.close()
                                    if verbose:
                                        print("find via rl path")
                                        print("final expansion node number:", count)
                                    closed_set[node_ind] = node
                                    break
                                else:
                                    # cost_qp = self.calc_euclidean_distance(node, ngoal)
                                    # self.qp.queue[node_ind] = cost_qp
                                    self.qp.queue[node_ind] = self.calc_hybrid_cost_simplify(node, ngoal, path.rlcost) 
                                    
                            else:
                                # find_feasible, path = self.rl_gear(node, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
                                find_feasible, path = self.new_rl_gear(node, ngoal, obstacles_info=obstacles_info, map_vertices=map_vertices)
                                if self.config["plot_expand_tree"]:
                                    self.plot_expand_tree(start, goal, closed_set, open_set)
                                    plot_rl_path(path, self.ox, self.oy)
                                    plt.savefig("rl_training/savefig.png")
                                    plt.close()
                                if find_feasible:
                                    fnode = path.info["final_node"]
                                    find_rl_path = True
                                    update = find_feasible
                                    rl_path = path
                                    rl_control_list = path.rlcontrollist
                                    if verbose:
                                        print("find via rl path")
                                        print("final expansion node number:", count)
                                    closed_set[node_ind] = node
                                    break
                                else: 
                                    find_feasible, path = self.rs_gear(node, ngoal)
                                    if self.config["plot_expand_tree"]:
                                        self.plot_expand_tree(start, goal, closed_set, open_set)
                                        plot_rs_path(path, self.ox, self.oy)
                                        plt.savefig("rl_training/savefig.png")
                                        plt.close()
                                    if find_feasible:
                                        fnode = path.info["final_node"]
                                        find_rs_path = True
                                        update = find_feasible
                                        rs_path = path
                                        rs_control_list = path.rscontrollist
                                        
                                        if verbose:
                                            print("find via rs path")
                                            print("final expansion node number:", count)
                                        # Here you need to add node to closed set
                                        closed_set[node_ind] = node
                                        # break the inner expand_tree loop
                                        break
                                    else:
                                        self.qp.put(node_ind, self.calc_hybrid_cost_simplify(node, ngoal, path.rscost))
                    # if self.config["plot_expand_tree"]:
                    #     self.plot_expand_tree(start, goal, closed_set, open_set)
                    #     plot_rs_path(path, self.ox, self.oy)
                    #     plt.savefig("rl_training/savefig.png")
                    #     plt.close()         
            if self.config["plot_expand_tree"]:
                self.plot_expand_tree(start, goal, closed_set, open_set)
                plot_rs_path(path, self.ox, self.oy)
                plt.savefig("rl_training/savefig.png")
                plt.close() 
        
        # if self.config["save_final_plot"]:
        #     self.plot_expand_tree(start, goal, closed_set, open_set)
        #     plot_rs_path(path, self.ox, self.oy)
        #     if not os.path.exists("./rl_training/planner_png/tt_meta_reaching"):
        #         os.makedirs("./rl_training/planner_png/tt_meta_reaching")

        #     base_path = "./rl_training/planner_png/tt_meta_reaching/path_simulation"
        #     extension = ".png"

        #     all_files = os.listdir("./rl_training/planner_png/tt_meta_reaching")
        #     matched_files = [re.match(r'path_simulation(\d+)\.png', f) for f in all_files]
        #     numbers = [int(match.group(1)) for match in matched_files if match]

        #     if numbers:
        #         save_index = max(numbers) + 1
        #     else:
        #         save_index = 1
        #     plt.savefig(base_path + str(save_index) + extension)
        #     plt.close()
        
        
        if verbose:
            print("final expand node: ", len(open_set) + len(closed_set) - 1)
        
        if get_control_sequence:
            path, expand_control_list = self.extract_path_and_control(closed_set, fnode, nstart,find_rs_path=find_rs_path, find_rl_path=find_rl_path)
            if self.config["save_final_plot"]:
                self.plot_expand_tree(start, goal, closed_set, open_set)
                plot_rl_path(path, self.ox, self.oy)
                if not os.path.exists("./rl_training/planner_png/tt_meta_reaching"):
                    os.makedirs("./rl_training/planner_png/tt_meta_reaching")

                base_path = "./rl_training/planner_png/tt_meta_reaching/path_simulation"
                extension = ".png"

                all_files = os.listdir("./rl_training/planner_png/tt_meta_reaching")
                matched_files = [re.match(r'path_simulation(\d+)\.png', f) for f in all_files]
                numbers = [int(match.group(1)) for match in matched_files if match]

                if numbers:
                    save_index = max(numbers) + 1
                else:
                    save_index = 1
                plt.savefig(base_path + str(save_index) + extension)
                plt.close()
            if find_rl_path:
                all_control_list = expand_control_list + rl_control_list
                return path, all_control_list, rl_path, len(open_set) + len(closed_set) - 1
            elif find_rs_path:
                all_control_list = expand_control_list + rs_control_list
                return path, all_control_list, rs_path, len(open_set) + len(closed_set) - 1
            else:
                all_control_list = expand_control_list
                return path, all_control_list, None, len(open_set) + len(closed_set) - 1
            # if self.heuristic_type == "rl":
            #     if find_rl_path:
            #         all_control_list = expand_control_list + rl_control_list
            #     else:
            #         rl_path = None
            #         all_control_list = expand_control_list
            #     return path, all_control_list, rl_path
            # else:
            #     if find_rs_path:
            #         all_control_list = expand_control_list + rs_control_list
            #     else:
            #         rs_path = None
            #         all_control_list = expand_control_list
            #     return path, all_control_list, rs_path         
        else:
            if find_rl_path:
                return self.extract_path(closed_set, fnode, nstart), None, rl_path, len(open_set) + len(closed_set) - 1
            elif find_rs_path:
                return self.extract_path(closed_set, fnode, nstart), None, rs_path, len(open_set) + len(closed_set) - 1
            else: 
                return self.extract_path(closed_set, fnode, nstart), None, None, len(open_set) + len(closed_set) - 1
            
            # if self.heuristic_type == "rl":
            #     if find_rl_path: 
            #         return self.extract_path(closed_set, fnode, nstart), None, rl_path
            #     else:
            #         return self.extract_path(closed_set, fnode, nstart), None, None
            # else:
            #     if find_rs_path: 
            #         return self.extract_path(closed_set, fnode, nstart), None, rs_path
            #     else:
            #         return self.extract_path(closed_set, fnode, nstart), None, None
    
    def plot_expand_tree(self, start, goal, closed_set, open_set):
        plt.axis("equal")
        ax = plt.gca() 
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
        
        for key, value in open_set.items():
            self.plot_node(value, color='gray')
        for key, value in closed_set.items():
            self.plot_node(value, color='red')
        # change here last plot goal and start
        self.vehicle.reset(*goal)
        
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        self.vehicle.reset(*start)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
    
    def plot_node(self, node, color):
        xlist = node.x
        ylist = node.y
        plt.plot(xlist, ylist, color=color, markersize=1)
    
    def plot_real_path(self, rx, ry):
        plt.plot(rx, ry, color="blue", markersize=1)
    
    
    def visualize_planning(self, start, goal, path, 
                           gif=True, save_dir='./HybridAstarPlanner/gif'):
        """visuliaze the planning result
        : param path: a path class
        : start & goal: cast as np.ndarray
        """
        print("Start Visulizate the Result")
        x = path.x
        y = path.y
        yaw = path.yaw
        yawt1 = path.yawt1
        yawt2 = path.yawt2
        yawt3 = path.yawt3
        direction = path.direction
        
        if gif:
            fig, ax = plt.subplots()

            def update(num):
                ax.clear()
                plt.axis("equal")
                k = num
                # plot env (obstacle)
                plt.plot(self.ox, self.oy, "sk", markersize=1)
                self.vehicle.reset(*start)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
                # draw_model_three_trailer(TT, gx, gy, gyaw0, gyawt1, gyawt2, gyawt3, 0.0)
                # plot the planning path
                plt.plot(x, y, linewidth=1.5, color='r')
                self.vehicle.reset(*goal)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
                self.vehicle.reset(x[k], y[k], yaw[k], yawt1[k], yawt2[k], yawt3[k])
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0

                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')
                plt.axis("equal")

            ani = FuncAnimation(fig, update, frames=len(x), repeat=True)

            # Save the animation
            writer = PillowWriter(fps=20)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                
            # base_path = "./HybridAstarPlanner/gif/path_animation"
            base_path = os.path.join(save_dir, 'hybrid_astar_path_plan_three_tractor_trailer')
            extension = ".gif"
            
            all_files = os.listdir(save_dir)
            matched_files = [re.match(r'hybrid_astar_path_plan_three_tractor_trailer(\d+)\.gif', f) for f in all_files]
            numbers = [int(match.group(1)) for match in matched_files if match]
            
            if numbers:
                save_index = max(numbers) + 1
            else:
                save_index = 1
            ani.save(base_path + str(save_index) + extension, writer=writer)
            print("Done Plotting")
            
        else:
            # this is when your device has display setting
            fig, ax = plt.subplots()
            # this is when your device has display setting
            plt.pause(5)

            for k in range(len(x)):
                plt.cla()
                plt.axis("equal")
                # plot env (obstacle)
                plt.plot(self.ox, self.oy, "sk", markersize=1)
                # plot the planning path
                plt.plot(x, y, linewidth=1.5, color='r')
                self.vehicle.reset(*start)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'gray')
                self.vehicle.reset(*goal)
                self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')

                # calculate every time step
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    # different from a single car
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0
                # draw goal model
                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')
                plt.pause(0.0001)

            plt.show()

    def extract_failed_path(self, start, goal, closed, nstart):
        
        for value in closed.values():
            plt.plot(value.x, value.y,'.', color='grey', markersize=1)
        plt.plot(nstart.x, nstart.y, 'o', color='r', markersize=3)    
        plt.plot(self.ox, self.oy, 'sk', markersize=1)
        # plt.legend()
        plt.axis("equal")
        ax = plt.gca()
        # Add plot vehicle start and goal
        self.vehicle.reset(*goal)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        self.vehicle.reset(*start)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'black')
        if not os.path.exists("planner_result/failed_trajectory_three_trailer"):
            os.makedirs("planner_result/failed_trajectory_three_trailer")
            
        base_path = "./planner_result/failed_trajectory_three_trailer"
        extension = ".png"
            
        all_files = os.listdir("./planner_result/failed_trajectory_three_trailer")
        matched_files = [re.match(r'explored(\d+)\.png', f) for f in all_files]
        numbers = [int(match.group(1)) for match in matched_files if match]
        
        if numbers:
            save_index = max(numbers) + 1
        else:
            save_index = 0
        plt.savefig(base_path + "/explored" + str(save_index) + extension)
        plt.close()
        # plt.savefig("HybridAstarPlanner/trajectory/explored.png")

    
