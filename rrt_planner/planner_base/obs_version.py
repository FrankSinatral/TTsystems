import numpy as np
import os
import sys
import math
from scipy.spatial import KDTree
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../../ttsystems/")

import curves_generator.dubins_path as db
import rrt_planner.planner_base.rrt as rrt
import matplotlib.pyplot as plt


class RRTPlanner():
    def __init__(self, ox, oy, args):
        self.vehicle = SingleTractor(args)
        self.iter_max = args.max_iter
        self.goal_sample_rate = args.sample_rate
        self.ox = ox
        self.oy = oy
        self.safe_d = args.safe_d
        self.minx = min(self.ox)
        self.maxx = max(self.ox)
        self.miny = min(self.oy)
        self.maxy = max(self.oy)
        self.kdtree = KDTree([[x, y] for x, y in zip(self.ox, self.oy)])
        self.minyaw = -self.vehicle.PI
        self.maxyaw = self.vehicle.PI 
        self.step_size = 0.2
        # self.eta = 138.0 
        self.vertex = [] 
        self.delta = 0.5
        self.d1 = 60
        self.d2 = 2
        
    
    def calc_motion_set(self):
        """
        this is much alike motion primitives
        """
        s = [i for i in np.arange(self.vehicle.MAX_STEER / self.vehicle.N_STEER,
                                self.vehicle.MAX_STEER, self.vehicle.MAX_STEER / self.vehicle.N_STEER)]

        steer = [0.0] + s + [-i for i in s]
        direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
        steer = steer + steer

        return steer, direc
    
    def is_the_start(self, node1, node2):
        """
        whether the two nodes are all start node
        """
        if len(node1.xlist) == 1 and len(node2.ylist) == 1:
            return True
        return False
    
    def plan(self, ax, args, start:np.ndarray, goal:np.ndarray):
        #TODO: plan using rrt framework
        start_x, start_y, start_yaw = start
        goal_x, goal_y, goal_yaw = goal
        self.goal = rrt.Node_single_tractor(goal_x, goal_y, goal_yaw, 1,
                                             0.0, [goal_x], [goal_y], [goal_yaw], 0.0, None)
        self.start = rrt.Node_single_tractor(start_x, start_y, start_yaw, 1,
                                             0.0, [start_x], [start_y], [start_yaw], 0.0, None)
        self.plot_vehicle_from_node(ax, args, self.goal, color='r')
        self.plot_vehicle_from_node(ax, args, self.start, color='blue')
        self.steering(ax, args, self.start, self.goal, d=self.d1, eta=5.0)
        self.vertex.append(self.start)
        for k in range(self.iter_max):
            node_rand = self.generate_random_node()
            # self.plot_vehicle_from_node(ax, args, node_rand)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            # here's when you apply steering
            node_new = self.steering(ax, args, node_near, node_rand, d=self.d1, eta=10.0)
            if node_new is not None and not self.is_collision(node_new):
                self.vertex.append(node_new)
                self.plot_expand_tree(ax, args)
            else:
                continue
            #TODO
            if self.distance_between_node(node_new, self.goal) < 60:   
                node_final = self.analystic_expansion_dubins(ax, args, node_new, self.goal)
                if node_final is not None:
                    self.vertex.append(node_final)
                    self.plot_expand_tree(ax, args)
                    print("final expand number:", k)
                    return self.extract_path(node_final)
        
        return None    

        # index = self.search_goal_parent()
        # self.path = self.extract_path(self.vertex[index])

        # self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))
    
    def plot_vehicle_from_node(self, ax, args, node, color='black'):
        x = node.x
        y = node.y
        yaw = node.yaw
        vehicle = SingleTractor(args)
        vehicle.state = (x, y, yaw)
        action = np.array([0.0, 0.0], dtype=np.float32)
        vehicle.plot(ax, action, color=color)
        xlist = node.xlist
        ylist = node.ylist
        plt.plot(xlist, ylist, 'gray')
        
    def plot_expand_tree(self, ax, args, color='black'):
        for node in self.vertex:
            self.plot_vehicle_from_node(ax, args, node, color=color)   
        
            
    def extract_path(self, node_goal):
        rx, ry, ryaw, direc = [], [], [], []
        node = node_goal
        while True:
            rx += node.xlist[::-1]
            ry += node.ylist[::-1]
            ryaw += node.ylist[::-1]
            direc += [1] * len(node.xlist)
            
            if self.is_the_start(node, self.start):
                break
            node = node.parent
        
        plt.plot(rx, ry, 'r')
        rx = rx[::-1]
        ry = ry[::-1]
        ryaw = ryaw[::-1]
        direc = direc[::-1]
        path = rrt.Path_single_tractor(rx, ry, ryaw, direc)
        
        return path
    
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
        vehicle = SingleTractor(args)
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
        """One step simulation for single tractor"""
        sx, sy, syaw0 = state
        next_sx = sx + control[0] * math.cos(syaw0)
        next_sy = sy + control[0] * math.sin(syaw0)
        next_syaw0 = self.pi_2_pi(syaw0 + control[0] / self.vehicle.WB * math.tan(control[1]))
        return np.array([next_sx, next_sy, next_syaw0], dtype=np.float32)
    
    def calc_motion_set(self):
        """
        this is much alike motion primitives
        """
        s = [i for i in np.arange(self.vehicle.MAX_STEER / self.vehicle.N_STEER,
                                self.vehicle.MAX_STEER, self.vehicle.MAX_STEER / self.vehicle.N_STEER)]

        steer = [0.0] + s + [-i for i in s]
        direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
        steer = steer + steer

        return steer, direc
    
    def get_nearest_node(self):
        pass
    
    def steering(self, ax, args, node_near, node_rand, d, eta):
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
        state = np.array([node_near_x, node_near_y, node_near_yaw], 
                         dtype=np.float32)
        if path.L > eta:
            index_end = round(eta / self.step_size)
            for i in range(index_end):
                state_ = self.step(state, path.controllist[i])
                state = state_
                x, y, yaw = state
                xlist.append(x)
                ylist.append(y)
                yawlist.append(yaw)
            node_expand = rrt.Node_single_tractor(*state, 1, path.controllist[index_end - 1][1],
                                                  xlist, ylist, yawlist, cost=0.0, parent=node_near)
        else:
            for control in path.controllist:
                state_ = self.step(state, control)
                state = state_
                x, y, yaw = state
                xlist.append(x)
                ylist.append(y)
                yawlist.append(yaw)
            node_expand = rrt.Node_single_tractor(*state, 1, path.controllist[-1][-1],
                                                  xlist, ylist, yawlist, cost=0.0, parent=node_near)     
        # self.plot_vehicle_from_node(ax, args, node_expand)
        if self.distance_between_node(node_expand, node_rand) < d:
            return node_expand 
        return None
    
    def analystic_expansion_dubins(self, ax, args, node1, node2):
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
    
    