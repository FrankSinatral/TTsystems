import numpy as np
import os
import sys
import math
from scipy.spatial import KDTree
import tractor_trailer_envs as tt_envs
import curves_generator.dubins_path as db
# import rrt_planner.planner_base.rrt as rrt
import planner_zoo.rrt_planner.rrt as rrt

import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.animation import FuncAnimation, PillowWriter
import re
def define_map():
    pass



class RRTPlanner():
    
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
            "max_iter": 100000,
            "goal_sample_rate": 0.4,
            "step_size": 0.2,
            "delta": 0.5,
            "d1": 60,
            "d2": 2,
            "is_save_animation": True,
            "is_save_expand_tree": True,
            "visualize_mode": True,
            "dt": 0.1,
        }
        
    @staticmethod
    def pi_2_pi(theta):
        while theta >= np.pi:
            theta -= 2.0 * np.pi

        while theta < -np.pi:
            theta += 2.0 * np.pi
        
        return theta 
    
    @staticmethod
    def nearest_neighbor(node_list, node):
        return node_list[int(np.argmin([math.sqrt((nd.x - node.x)**2 + (nd.y - node.y)**2 + (nd.yaw - node.yaw)**2)
                                        for nd in node_list]))]
        
    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    
    def __init__(self, ox, oy, config: Optional[dict] = None):
        self.config = self.default_config()
        self.configure(config)
        self.ox = ox
        self.oy = oy
        self.minx = min(self.ox)
        self.maxx = max(self.ox)
        self.miny = min(self.oy)
        self.maxy = max(self.oy)
        self.kdtree = KDTree([[x, y] for x, y in zip(self.ox, self.oy)])
    
    def seed(self, seed=None):
        np.random.seed(seed)
        
        
    def configure(self, config: Optional[dict]) -> None:
        if config:
            self.config.update(config)
            
        self.vehicle = tt_envs.SingleTractor(self.config["controlled_vehicle_config"])
        self.iter_max = self.config["max_iter"]
        self.goal_sample_rate = self.config["goal_sample_rate"]
        self.delta = self.config["delta"]
        self.d1 = self.config["d1"]
        self.d2 = self.config["d2"]
        self.safe_d = self.vehicle.SAFE_D
        self.minyaw = -self.vehicle.PI
        self.maxyaw = self.vehicle.PI 
        self.step_size = 0.2
        # self.eta = 138.0 
        self.vertex = []
    
    def is_the_start(self, node1, node2):
        """
        whether the two nodes are all start node
        """
        if len(node1.xlist) == 1 and len(node2.ylist) == 1:
            return True
        return False
    
    def plan(self, ax, start:np.ndarray, goal:np.ndarray):
        #TODO: plan using rrt framework
        start_x, start_y, start_yaw = start
        goal_x, goal_y, goal_yaw = goal
        self.goal = rrt.Node_single_tractor(goal_x, goal_y, goal_yaw, 1,
                                             0.0, [goal_x], [goal_y], [goal_yaw], 0.0, None)
        self.start = rrt.Node_single_tractor(start_x, start_y, start_yaw, 1,
                                             0.0, [start_x], [start_y], [start_yaw], 0.0, None)
        if self.config["visualize_mode"]:
            self.plot_vehicle_from_node(ax, self.goal, color='r')
            self.plot_vehicle_from_node(ax, self.start, color='blue')
        self.steering(self.start, self.goal, d=self.d1, eta=5.0)
        self.vertex.append(self.start)
        for k in range(self.iter_max):
            node_rand = self.generate_random_node()
            # self.plot_vehicle_from_node(ax, args, node_rand)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            # here's when you apply steering
            node_new = self.steering(node_near, node_rand, d=self.d1, eta=10.0)
            if node_new is not None and not self.is_collision(node_new):
                self.vertex.append(node_new)
                if self.config["visualize_mode"]:
                    
                    self.plot_expand_tree(ax, start, goal)
            else:
                continue
            #TODO
            if self.distance_between_node(node_new, self.goal) < self.config["d1"]:   
                node_final = self.analystic_expansion_dubins(ax, node_new, self.goal)
                if node_final is not None:
                    self.vertex.append(node_final)
                    if self.config["visualize_mode"]:
                        self.plot_expand_tree(ax, start, goal)
                    print("final expand number:", k)
                    return self.extract_path(node_final)
        
        return None    

        # index = self.search_goal_parent()
        # self.path = self.extract_path(self.vertex[index])

        # self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))
    
    def plot_vehicle_from_node(self, ax, node, color='black'):
        x = node.x
        y = node.y
        yaw = node.yaw
        # vehicle = tt_envs.SingleTractor(self.config['controlled_vehicle_config'])
        # vehicle.reset(x, y, yaw)
        # action = np.array([0.0, 0.0], dtype=np.float32)
        # vehicle.plot(ax, action, color=color)
        xlist = node.xlist
        ylist = node.ylist
        plt.plot(xlist, ylist, 'gray')
        
    def plot_expand_tree(self, ax, start, goal, color='black'):
        self.vehicle.reset(*goal)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'green')
        self.vehicle.reset(*start)
        self.vehicle.plot(ax, np.array([0.0, 0.0], dtype=np.float32), 'blue')
        for node in self.vertex:
            self.plot_vehicle_from_node(ax, node, color=color)   
        
            
    def extract_path(self, node_goal):
        rx, ry, ryaw, direc = [], [], [], []
        node = node_goal
        while True:
            rx += node.xlist[::-1]
            ry += node.ylist[::-1]
            ryaw += node.yawlist[::-1]
            direc += [1] * len(node.xlist)
            
            if self.is_the_start(node, self.start):
                break
            node = node.parent
        
        plt.plot(rx, ry, 'blue')
        if self.config["is_save_expand_tree"]:
            plt.savefig("rrt_planning.png")
        plt.close()
        rx = rx[::-1]
        ry = ry[::-1]
        ryaw = ryaw[::-1]
        direc = direc[::-1]
        path = rrt.Path_single_tractor(rx, ry, ryaw, direc)
        
        return path

    
    
    def new_state(self, node_start, node_goal):
        #TODO have to change here
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = rrt.Node_single_tractor((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new
        
    
    def generate_random_node(self):
        self.delta = 0.5
        if np.random.random() > self.goal_sample_rate:
            x_coordinate = np.random.uniform(low=self.minx + self.delta, high=self.maxx - self.delta)
            y_coordinate = np.random.uniform(low=self.miny + self.delta, high=self.maxy - self.delta)
            yaw_coordinate = np.random.uniform(low=self.minyaw, high=self.maxyaw)
            return rrt.Node_single_tractor(x_coordinate, y_coordinate, yaw_coordinate,
                                           1, 0.0, [x_coordinate], [y_coordinate],
                                           [yaw_coordinate], 0.0, None)      
        
        return self.goal 
    
    def test_generate_random_node(self, ax, node, args):
        x = node.x
        y = node.y
        yaw = node.yaw
        vehicle = tt_envs.SingleTractor(args)
        vehicle.state = (x, y, yaw)
        action = np.array([0.0, 0.0], dtype=np.float32)
        vehicle.plot(ax, action)
               
    
    def is_collision(self, node):
        '''
        check whether there is collision
        Inputs:
        x, y, yaw: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        x = node.xlist
        y = node.ylist
        yaw = node.yawlist
        for ix, iy, iyaw in zip(x, y, yaw):
            # first trailer test collision
            d = self.safe_d
                        
            # check the tractor collision
            deltal = (self.vehicle.RF - self.vehicle.RB) / 2.0
            rc = (self.vehicle.RF + self.vehicle.RB) / 2.0 + d

            cx = ix + deltal * math.cos(iyaw)
            cy = iy + deltal * math.sin(iyaw)

            ids = self.kdtree.query_ball_point([cx, cy], rc)

            if ids:
                for i in ids:
                    xo = self.ox[i] - cx
                    yo = self.oy[i] - cy

                    dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                    dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                    if abs(dx_car) <= rc and \
                            abs(dy_car) <= self.vehicle.W / 2.0 + d:
                        return True

        return False
    
    def is_collision_dubins(self, dubinspath):
        '''
        check whether there is collision
        Inputs:
        x, y, yaw: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        x = dubinspath.x
        y = dubinspath.y
        yaw = dubinspath.yaw
        for ix, iy, iyaw in zip(x, y, yaw):
            # first trailer test collision
            d = self.safe_d
                        
            # check the tractor collision
            deltal = (self.vehicle.RF - self.vehicle.RB) / 2.0
            rc = (self.vehicle.RF + self.vehicle.RB) / 2.0 + d

            cx = ix + deltal * math.cos(iyaw)
            cy = iy + deltal * math.sin(iyaw)

            ids = self.kdtree.query_ball_point([cx, cy], rc)

            if ids:
                for i in ids:
                    xo = self.ox[i] - cx
                    yo = self.oy[i] - cy

                    dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                    dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                    if abs(dx_car) <= rc and \
                            abs(dy_car) <= self.vehicle.W / 2.0 + d:
                        return True

        return False
    
    def step(self, state, control):
        """One step simulation
        the first dimension of control is #[m]
        """
        sx, sy, syaw0 = state
        next_sx = sx + control[0] * math.cos(syaw0)
        next_sy = sy + control[0] * math.sin(syaw0)
        next_syaw0 = self.pi_2_pi(syaw0 + control[0] / self.vehicle.WB * math.tan(control[1]))
        return np.array([next_sx, next_sy, next_syaw0], dtype=np.float32)
    
    # def calc_motion_set(self):
    #     """
    #     this is much alike motion primitives
    #     """
    #     s = [i for i in np.arange(self.vehicle.MAX_STEER / self.vehicle.N_STEER,
    #                             self.vehicle.MAX_STEER, self.vehicle.MAX_STEER / self.vehicle.N_STEER)]

    #     steer = [0.0] + s + [-i for i in s]
    #     direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    #     steer = steer + steer

    #     return steer, direc
    
    
    def steering(self, node_near, node_rand, d, eta):
        #TODO: steering towards the goal
        node_near_x = node_near.x
        node_near_y = node_near.y
        node_near_yaw = node_near.yaw
        node_rand_x = node_rand.x
        node_rand_y = node_rand.y
        node_rand_yaw = node_rand.yaw
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        path = db.calc_dubins_path(node_near_x, node_near_y, node_near_yaw, node_rand_x, node_rand_y, node_rand_yaw,
                                   maxc, self.vehicle.MAX_STEER, step_size=self.step_size)
        # self.plot_dubins_path(path)
        xlist = [node_near_x]
        ylist = [node_near_y]
        yawlist = [node_near_yaw]
        # steerlist = []
        self.vehicle.reset(node_near_x, node_near_y, node_near_yaw)
        state = np.array([node_near_x, node_near_y, node_near_yaw], 
                         dtype=np.float64)
        if path.L > eta:
            index_end = round(eta / self.step_size)
            for i in range(index_end):
                # state_ = self.step(state, path.controllist[i])
                # state = state_
                # x, y, yaw = state
                self.vehicle.step(np.array([path.controllist[i][0] / self.config["dt"], path.controllist[i][1]], dtype=np.float32), self.config["dt"], scale=False)
                x, y, yaw = self.vehicle.state
                xlist.append(x)
                ylist.append(y)
                yawlist.append(yaw)
                state = np.array([x, y, yaw], dtype=np.float64)
            node_expand = rrt.Node_single_tractor(*state, 1, path.controllist[index_end - 1][1],
                                                  xlist, ylist, yawlist, cost=0.0, parent=node_near)
        else:
            for control in path.controllist:
                self.vehicle.step(np.array([control[0] / self.config["dt"], control[1]], dtype=np.float32), self.config["dt"], scale=False)
                x, y, yaw = self.vehicle.state
                xlist.append(x)
                ylist.append(y)
                yawlist.append(yaw)
                state = np.array([x, y, yaw], dtype=np.float64)
            node_expand = rrt.Node_single_tractor(*state, 1, path.controllist[-1][-1],
                                                  xlist, ylist, yawlist, cost=0.0, parent=node_near)     
        # self.plot_vehicle_from_node(ax, args, node_expand)
        if self.distance_between_node(node_expand, node_rand) < d:
            return node_expand 
        return None
    
    def analystic_expansion_dubins(self, ax, node1, node2):
        node1_x = node1.x
        node1_y = node1.y
        node1_yaw = node1.yaw
        node2_x = node2.x
        node2_y = node2.y
        node2_yaw = node2.yaw
        maxc = math.tan(self.vehicle.MAX_STEER) / self.vehicle.WB
        path = db.calc_dubins_path(node1_x, node1_y, node1_yaw, node2_x, node2_y, node2_yaw,
                                   maxc, self.vehicle.MAX_STEER, step_size=self.step_size)
        if not self.is_collision_dubins(path):
            xlist = path.x.tolist()
            ylist = path.y.tolist()
            yawlist = path.yaw
            x = path.x[-1]
            y = path.y[-1]
            yaw = path.yaw[-1]
            node = rrt.Node_single_tractor(x, y, yaw, 1, path.controllist[-1][1],
                                    xlist, ylist, yawlist, cost=0.0, parent=node1)
            return node
        return None
    
    def plot_dubins_path(self, path):
        x = path.x
        y = path.y
        yaw = path.yaw
        plt.plot(x, y, 'r')
        
    
    def distance_between_node(self, node1, node2):
        node1_x = node1.x
        node1_y = node1.y
        node1_yaw = node1.yaw
        node2_x = node2.x
        node2_y = node2.y
        node2_yaw = node2.yaw
        
        return math.sqrt((node1_x - node2_x)**2 + (node1_y - node2_y)**2 + (node1_yaw - node2_yaw)**2)
    
    def visualize_planning(self, start: np.ndarray, goal: np.ndarray, path, save_dir='./planner_result/gif'):
        """visuliaze the planning result
        : param path: a path class
        : start & goal: cast as np.ndarray
        """
        print("Start Visulizate the Result")
        x = path.x
        y = path.y
        yaw = path.yaw
        # yawt1 = path.yawt1
        # yawt2 = path.yawt2
        # yawt3 = path.yawt3
        direction = path.direction
        
        if self.config["is_save_animation"]:
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
                self.vehicle.reset(x[k], y[k], yaw[k])
                if k < len(x) - 2:
                    dy = (yaw[k + 1] - yaw[k]) / self.step_size
                    steer = self.pi_2_pi(math.atan(self.vehicle.WB * dy / direction[k]))
                else:
                    steer = 0.0
                self.vehicle.plot(ax, np.array([0.0, steer], dtype=np.float32), 'black')

            ani = FuncAnimation(fig, update, frames=len(x), repeat=True)

            # Save the animation
            writer = PillowWriter(fps=20)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # base_path = "./planner_result/gif/rrt_path_plan_single_tractor" 
            base_path = os.path.join(save_dir, 'rrt_path_plan_single_tractor')
            extension = ".gif"
            
            all_files = os.listdir(save_dir)
            matched_files = [re.match(r'rrt_path_plan_single_tractor(\d+)\.gif', f) for f in all_files]
            numbers = [int(match.group(1)) for match in matched_files if match]
            
            if numbers:
                save_index = max(numbers) + 1
            else:
                save_index = 1
            ani.save(base_path + str(save_index) + extension, writer=writer)
            print("Done Plotting")
            
        else:
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