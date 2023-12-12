from abc import ABC, abstractmethod
from scipy.spatial import KDTree
import numpy as np
from heapdict import heapdict
import heapq
import math
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
class NewQueuePrior:
    def __init__(self):
        self.queue = heapdict()
        
    def empty(self):
        return len(self.queue) == 0
    
    def put(self, item, priority):
       self.queue[item] = priority
    
    def get(self):
        return self.queue.popitem()[0] 
    
    def reset(self):
        self.queue = heapdict()
        
class QueuePrior:
    def __init__(self):
        self.queue = []
        self.counter = 0

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        # Check if the same cost already exists in the queue
        # if any(pri == priority for pri, _ in self.queue):
        #     return  # Do not insert the new path if cost matches
        heapq.heappush(self.queue, (priority, self.counter, item))  # reorder x using priority
        self.counter += 1

    def get(self):
        return heapq.heappop(self.queue)[-1]  # pop out element with smallest priority

    def reset(self):
        self.queue.clear()
        self.counter = 0

class Node_single_tractor:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind):
        # notice that here I don't assume the input
        # but one has to turn the yaw to [-np.pi, np.pi)
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y 
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind
        
    def __str__(self):
        return f"A Single Tractor Node ({self.x[-1]},{self.y[-1]},{self.yaw[-1]} with cost {self.cost}"  

class Node_one_trailer:
    def __init__(self, vehicle, xind, yind, yawind, direction, x, y,
                 yaw, yawt1, directions, steer, cost, pind):
        '''
        Node class for three trailer vehicle
        xind: x coordinate
        direction: 1 for forward; -1 for backward
        x,y,yaw: list from parent node to this node (parent node not included)
        yawt1: yaw trailer1 list
        directions: direction list
        steer: a value
        cost: node cost(not hybrid cost)
        pind: parent index
        '''
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt1 = yawt1
        self.vehicle = vehicle
        # here I change the calculate of xi4, xi5, xi6
        self.xi4 = [self.pi_2_pi(x - y) for x, y in zip(self.yawt1, self.yaw)]
        self._validate_data(self.xi4)
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind
    
    def __str__(self):
        return f"Node ({self.x[-1]},{self.y[-1]},{self.yaw[-1]},{self.yawt1[-1]}) with cost {self.cost}"
    
    @staticmethod
    def pi_2_pi(theta):
        while theta >= np.pi:
            theta -= 2.0 * np.pi

        while theta < -np.pi:
            theta += 2.0 * np.pi
        
        return theta    
        
    def _validate_data(self, xi4):
        for lst in [xi4]:
            for val in lst:
                if val < -self.vehicle.XI_MAX or val > self.vehicle.XI_MAX:
                    raise ValueError("jack-knife configuration")



class Node_three_trailer:
    def __init__(self, vehicle, xind, yind, yawind, direction, x, y,
                 yaw, yawt1, yawt2, yawt3, directions, steer, cost, pind):
        '''
        Node class for three trailer vehicle
        xind: x coordinate
        direction: 1 for forward; -1 for backward
        x,y,yaw: list from parent node to this node (parent node not included)
        yawt1: yaw trailer1 list
        yawt2: yaw trailer2 list
        yawt3: yaw trailer3 list
        directions: direction list
        steer: a value
        cost: node cost(not hybrid cost)
        pind: parent index
        '''
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt1 = yawt1
        self.yawt2 = yawt2
        self.yawt3 = yawt3
        self.vehicle = vehicle
        # here I change the calculate of xi4, xi5, xi6
        self.xi4 = [self.pi_2_pi(x - y) for x, y in zip(self.yawt1, self.yaw)]
        self.xi5 = [self.pi_2_pi(x - y) for x, y in zip(self.yawt2, self.yawt1)]
        self.xi6 = [self.pi_2_pi(x - y) for x, y in zip(self.yawt3, self.yawt2)]
        self._validate_data(self.xi4, self.xi5, self.xi6)
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind
    
    def __str__(self):
        return f"Node ({self.x[-1]},{self.y[-1]},{self.yaw[-1]},{self.yawt1[-1]}, {self.yawt2[-1]}, {self.yawt3[-1]}) with cost {self.cost}"
    
    @staticmethod
    def pi_2_pi(theta):
        while theta >= np.pi:
            theta -= 2.0 * np.pi

        while theta < -np.pi:
            theta += 2.0 * np.pi
        
        return theta    
        
    def _validate_data(self, xi4, xi5, xi6):
        for lst in [xi4, xi5, xi6]:
            for val in lst:
                if val < -self.vehicle.XI_MAX or val > self.vehicle.XI_MAX:
                    raise ValueError("jack-knife configuration")

class Para_single_tractor:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
        '''
        minx: 在resolution下的最小
        xw: 范围
        ox: 障碍物信息
        '''
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree
        
class Para_one_trailer:
    """
    This class is for storing the parameters
    """
    def __init__(self, minx, miny, minyaw, minyawt1, maxx, maxy, maxyaw, maxyawt1,
                 xw, yw, yaww, yawt1w, xyreso, yawreso, ox, oy, kdtree):
        '''
        ox, oy: obstacle position
        yawt1w, yawt2w: width of yawt1/ yawt2
        '''
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.minyawt1 = minyawt1
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.maxyawt1 = maxyawt1
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.yawt1w = yawt1w
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree

class Para_three_trailer:
    """
    This class is for storing the parameters
    """
    def __init__(self, minx, miny, minyaw, minyawt1, minyawt2, minyawt3, maxx, maxy, maxyaw, maxyawt1, maxyawt2,
                 maxyawt3, xw, yw, yaww, yawt1w, yawt2w, yawt3w, xyreso, yawreso, ox, oy, kdtree):
        '''
        ox, oy: obstacle position
        yawt1w, yawt2w: width of yawt1/ yawt2
        '''
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.minyawt1 = minyawt1
        self.minyawt2 = minyawt2
        self.minyawt3 = minyawt3
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.maxyawt1 = maxyawt1
        self.maxyawt2 = maxyawt2
        self.maxyawt3 = maxyawt3
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.yawt1w = yawt1w
        self.yawt2w = yawt2w
        self.yawt3w = yawt3w
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree

class Path_one_trailer:
    """
    path parameter
    """
    def __init__(self, x, y, yaw, yawt1, direction, cost):
        """
        x, y, yaw, yawt1, direction: list
        cost: value
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt1 = yawt1
        self.direction = direction
        self.cost = cost
        
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        yaw = self.yaw + other.yaw
        yawt1 = self.yawt1 + other.yawt1
        direction = self.direction + other.direction
        cost = self.cost + other.cost
        return Path_one_trailer(x, y, yaw, yawt1, direction, cost)
        
class Path_three_trailer:
    """
    path parameter
    """
    def __init__(self, x, y, yaw, yawt1, yawt2, yawt3, direction, cost):
        """
        x, y, yaw, yawt1, direction: list
        cost: value
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt1 = yawt1
        self.yawt2 = yawt2
        self.yawt3 = yawt3
        self.direction = direction
        self.cost = cost
        
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        yaw = self.yaw + other.yaw
        yawt1 = self.yawt1 + other.yawt1
        yawt2 = self.yawt2 + other.yawt2
        yawt3 = self.yawt3 + other.yawt3
        direction = self.direction + other.direction
        cost = self.cost + other.cost
        return Path_three_trailer(x, y, yaw, yawt1, yawt2, yawt3, direction, cost)

class Path_single_tractor:
    """
    path parameter
    """
    def __init__(self, x, y, yaw, direction, cost):
        """
        x, y, yaw, yawt1, direction: list
        cost: value
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost
        
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        yaw = self.yaw + other.yaw
        direction = self.direction + other.direction
        cost = self.cost + other.cost
        return Path_single_tractor(x, y, yaw, direction, cost)

class BasicHybridAstarPlanner(ABC):
    PI = np.pi
    def __init__(self, ox, oy):
        
        self._ox = ox
        self._oy = oy
        self.kdtree = self._create_kdtree()
        self.P = self.calc_parameters()
        
    @property
    def ox(self):
        return self._ox

    @property
    def oy(self):
        return self._oy

    def set_ox_oy(self, ox, oy):
        self._ox = ox
        self._oy = oy
        self.kdtree = self._create_kdtree()
        self.P = self.calc_parameters()

    def _create_kdtree(self):
        return KDTree([[x, y] for x, y in zip(self._ox, self._oy)])

    @abstractmethod
    def calc_parameters(self):
        pass

    @abstractmethod
    def plan(self, *args, **kwargs):
        pass
    
    @staticmethod
    def pi_2_pi(theta):
        while theta >= BasicHybridAstarPlanner.PI:
            theta -= 2.0 * BasicHybridAstarPlanner.PI

        while theta < -BasicHybridAstarPlanner.PI:
            theta += 2.0 * BasicHybridAstarPlanner.PI
        
        return theta 
    




class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node


class Para:
    def __init__(self, minx, miny, maxx, maxy, xw, yw, reso, motion):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xw = xw
        self.yw = yw
        self.reso = reso  # resolution of grid world
        self.motion = motion  # motion set


def astar_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    return path of A*.
    :param sx: starting node x [m]
    :param sy: starting node y [m]
    :param gx: goal node x [m]
    :param gy: goal node y [m]
    :param ox: obstacles x positions [m]
    :param oy: obstacles y positions [m]
    :param reso: xy grid resolution
    :param rr: robot radius
    :return: path
    """

    n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    n_goal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, rr, reso)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_start, P)] = n_start

    q_priority = []
    heapq.heappush(q_priority,
                   (fvalue(n_start, n_goal), calc_index(n_start, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority,
                                   (fvalue(node, n_goal), calc_index(node, P)))

    pathx, pathy = extract_path(closed_set, n_start, n_goal, P)

    return pathx, pathy


def calc_holonomic_heuristic_with_obstacle(node, ox, oy, reso, rr):
    """
    calculate heuristic considering obstacle
    
    - node: usually the goal node class
    - ox: obstacle x axis (original)
    - oy: obstacle y axis (original)
    - reso: resolution of x, y axis for the holonomic heuristic
    - rr: robot radius for holonomic heuristic
    """
    n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, rr, reso)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_goal, P)] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            curr_node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(curr_node, P, obsmap):
                continue

            n_ind = calc_index(curr_node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > curr_node.cost:
                        open_set[n_ind].cost = curr_node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = curr_node
                    heapq.heappush(q_priority, (curr_node.cost, calc_index(curr_node, P)))

    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]

    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost
    
    # visualize the hmap
    # visualize_hmap(hmap, node, reso)
    return hmap


def calc_holonomic_heuristic_with_obstacle_value(node, hmap, ox, oy, reso):
    """
    Using node hmap to calculate holonomic heuristic with obstacle
    
    Inputs:
    - node: a node class (we only need its position)
    - hmap: a pre-calculated heuristic map
    - ox/ oy: obstacle position before resolution
    - reso: resolution of heuristic, related to hmap
    """
    x = node.x[-1]
    y = node.y[-1]
    xind = round(x / reso)
    yind = round(y / reso)
    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]
    minx, miny = round(min(ox)), round(min(oy))
    return hmap[xind - minx][yind - miny]


def check_node(node, P, obsmap):
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy:
        return False

    if obsmap[node.x - P.minx][node.y - P.miny]:
        return False

    return True


def u_cost(u):
    return math.hypot(u[0], u[1])


def fvalue(node, n_goal):
    return node.cost + h(node, n_goal)


def h(node, n_goal):
    """
    heuristics
    """
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y)


def calc_index(node, P):
    return (node.y - P.miny) * P.xw + (node.x - P.minx)


def calc_parameters(ox, oy, rr, reso):
    minx, miny = round(min(ox)), round(min(oy))
    maxx, maxy = round(max(ox)), round(max(oy))
    xw, yw = maxx - minx + 1, maxy - miny + 1

    motion = get_motion()
    P = Para(minx, miny, maxx, maxy, xw, yw, reso, motion)
    obsmap = calc_obsmap(ox, oy, rr, P)

    return P, obsmap


def calc_obsmap(ox, oy, rr, P):
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]

    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox, oy):
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:
                    obsmap[x][y] = True
                    break

    return obsmap


def extract_path(closed_set, n_start, n_goal, P):
    pathx, pathy = [n_goal.x], [n_goal.y]
    n_ind = calc_index(n_goal, P)

    while True:
        node = closed_set[n_ind]
        pathx.append(node.x)
        pathy.append(node.y)
        n_ind = node.pind

        if node == n_start:
            break

    pathx = [x * P.reso for x in reversed(pathx)]
    pathy = [y * P.reso for y in reversed(pathy)]

    return pathx, pathy


def get_motion():
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion


def get_env():
    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    return ox, oy


def main():
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    robot_radius = 2.0
    grid_resolution = 1.0
    ox, oy = get_env()

    pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)

    plt.plot(ox, oy, 'sk')
    plt.plot(pathx, pathy, '-r')
    plt.plot(sx, sy, 'sg')
    plt.plot(gx, gy, 'sb')
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
