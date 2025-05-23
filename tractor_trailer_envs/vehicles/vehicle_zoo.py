import os
import sys
import numpy as np
from tractor_trailer_envs.vehicles.vehicle import Vehicle
import matplotlib.pyplot as plt
from tractor_trailer_envs.vehicles.config import get_config
from scipy.spatial import cKDTree
import tractor_trailer_envs.map_and_obstacles.settings as settings
from matplotlib.axes import Axes
from typing import Tuple, List, Union

def point_to_rectangle_distance(cx, cy, l, d, yaw, ox, oy):
    # Transform the points to the rectangle's local coordinate system
    ox_local = (ox - cx) * np.cos(yaw) + (oy - cy) * np.sin(yaw)
    oy_local = -(ox - cx) * np.sin(yaw) + (oy - cy) * np.cos(yaw)

    # Initialize the distance array with zeros
    # TODO: this has to be changed
    distances = 100 * np.ones((len(ox), 8))

    # Calculate the distance to each region
    # the i stands for the index of the point
    for i in range(len(ox)):
        # the right corner: yaw's direction
        if ox_local[i] >= l / 2 and oy_local[i] <= d / 2 and oy_local[i] >= - d / 2:
            distances[i, 0] = np.abs(ox_local[i] - l / 2)
        # the top-right corner
        elif ox_local[i] >= l / 2 and oy_local[i] >= d / 2:
            distances[i, 1] = np.sqrt((ox_local[i] - l / 2)**2 + (oy_local[i] - d / 2)**2)
        # the top corner
        elif ox_local[i] <= l / 2 and ox_local[i] >= -l / 2 and oy_local[i] >= d / 2:
            distances[i, 2] = np.abs(oy_local[i] - d / 2)
        # the top-left corner
        elif ox_local[i] <= -l / 2 and oy_local[i] >= d / 2:
            distances[i, 3] = np.sqrt((ox_local[i] + l / 2)**2 + (oy_local[i] - d / 2)**2)
        # the left corner
        elif ox_local[i] <= -l / 2 and oy_local[i] <= d / 2 and oy_local[i] >= -d / 2:
            distances[i, 4] = np.abs(ox_local[i] + l / 2)
        # the bottom-left corner
        elif ox_local[i] <= -l / 2 and oy_local[i] <= -d / 2:
            distances[i, 5] = np.sqrt((ox_local[i] + l / 2)**2 + (oy_local[i] + d / 2)**2)
        # the bottom corner
        elif ox_local[i] <= l / 2 and  ox_local[i] >= -l / 2 and oy_local[i] <= -d / 2:
            distances[i, 6] = np.abs(oy_local[i] + d / 2)
        # the bottom-right corner
        elif ox_local[i] >= l / 2 and oy_local[i] <= -d / 2:
            # The point is above the rectangle
            distances[i, 7] = np.sqrt((ox_local[i] - l / 2)**2 + (oy_local[i] + d / 2)**2)

    return distances.min(axis=0)

def point_to_rectangle_distance_vectorized(cx, cy, l, d, yaw, ox, oy):
    ox = np.array(ox)
    oy = np.array(oy)
    
    # Transform the points to the rectangle's local coordinate system
    ox_local = (ox - cx) * np.cos(yaw) + (oy - cy) * np.sin(yaw)
    oy_local = -(ox - cx) * np.sin(yaw) + (oy - cy) * np.cos(yaw)
    
    
    # Check if any point is inside the rectangle
    mask_inside = (ox_local <= l / 2) & (ox_local >= -l / 2) & (oy_local <= d / 2) & (oy_local >= -d / 2)
    if np.any(mask_inside):
        return np.zeros(8)
    
    
    # If coming to here, there is no collision happened
    # Initialize the distances matrix with infinity
    distances = np.full((len(ox), 8), np.inf)

    # Right side
    mask = (ox_local >= l / 2) & (oy_local <= d / 2) & (oy_local >= -d / 2)
    distances[mask, 0] = np.abs(ox_local[mask] - l / 2)

    # Top-right corner
    mask = (ox_local >= l / 2) & (oy_local >= d / 2)
    distances[mask, 1] = np.sqrt((ox_local[mask] - l / 2)**2 + (oy_local[mask] - d / 2)**2)

    # Top side
    mask = (ox_local <= l / 2) & (ox_local >= -l / 2) & (oy_local >= d / 2)
    distances[mask, 2] = np.abs(oy_local[mask] - d / 2)

    # Top-left corner
    mask = (ox_local <= -l / 2) & (oy_local >= d / 2)
    distances[mask, 3] = np.sqrt((ox_local[mask] + l / 2)**2 + (oy_local[mask] - d / 2)**2)

    # Left side
    mask = (ox_local <= -l / 2) & (oy_local <= d / 2) & (oy_local >= -d / 2)
    distances[mask, 4] = np.abs(ox_local[mask] + l / 2)

    # Bottom-left corner
    mask = (ox_local <= -l / 2) & (oy_local <= -d / 2)
    distances[mask, 5] = np.sqrt((ox_local[mask] + l / 2)**2 + (oy_local[mask] + d / 2)**2)

    # Bottom side
    mask = (ox_local <= l / 2) & (ox_local >= -l / 2) & (oy_local <= -d / 2)
    distances[mask, 6] = np.abs(oy_local[mask] + d / 2)

    # Bottom-right corner
    mask = (ox_local >= l / 2) & (oy_local <= -d / 2)
    distances[mask, 7] = np.sqrt((ox_local[mask] - l / 2)**2 + (oy_local[mask] + d / 2)**2)

    # Return the minimum distance for each of the 8 regions
    return np.min(distances, axis=0)

def lidar_one_hot(cx, cy, l, d, yaw, ox, oy, dist_threshold):
    ox = np.array(ox)
    oy = np.array(oy)
    
    # Transform the points to the rectangle's local coordinate system
    ox_local = (ox - cx) * np.cos(yaw) + (oy - cy) * np.sin(yaw)
    oy_local = -(ox - cx) * np.sin(yaw) + (oy - cy) * np.cos(yaw)
    
    # Check if any point is inside the rectangle
    mask_inside = (ox_local <= l / 2) & (ox_local >= -l / 2) & (oy_local <= d / 2) & (oy_local >= -d / 2)
    regions = np.zeros(9)
    regions[0] = np.any(mask_inside)
    
    # Right side
    mask = (ox_local >= l / 2) & (np.abs(oy_local) <= d / 2) & ((ox_local - l / 2) <= dist_threshold)
    regions[1] = np.any(mask)

    # Top-right corner
    mask = (ox_local >= l / 2) & (oy_local >= d / 2) & (np.sqrt((ox_local - l / 2)**2 + (oy_local - d / 2)**2) <= dist_threshold)
    regions[2] = np.any(mask)

    # Top side
    mask = (np.abs(ox_local) <= l / 2) & (oy_local >= d / 2) & ((oy_local - d / 2) <= dist_threshold)
    regions[3] = np.any(mask)

    # Top-left corner
    mask = (ox_local <= -l / 2) & (oy_local >= d / 2) & (np.sqrt((ox_local + l / 2)**2 + (oy_local - d / 2)**2) <= dist_threshold)
    regions[4] = np.any(mask)

    # Left side
    mask = (ox_local <= -l / 2) & (np.abs(oy_local) <= d / 2) & ((-ox_local - l / 2) <= dist_threshold)
    regions[5] = np.any(mask)

    # Bottom-left corner
    mask = (ox_local <= -l / 2) & (oy_local <= -d / 2) & (np.sqrt((ox_local + l / 2)**2 + (oy_local + d / 2)**2) <= dist_threshold)
    regions[6] = np.any(mask)

    # Bottom side
    mask = (np.abs(ox_local) <= l / 2) & (oy_local <= -d / 2) & ((-oy_local - d / 2) <= dist_threshold)
    regions[7] = np.any(mask)

    # Bottom-right corner
    mask = (ox_local >= l / 2) & (oy_local <= -d / 2) & (np.sqrt((ox_local - l / 2)**2 + (oy_local + d / 2)**2) <= dist_threshold)
    regions[8] = np.any(mask)

    return regions

def plot_lidar_detection(cx, cy, l, d, yaw, dist_threshold):
        # Convert to local coordinate system
        corners = np.array([
            [l/2, d/2],
            [l/2, -d/2],
            [-l/2, -d/2],
            [-l/2, d/2]
        ])
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T)
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        # Define regions based on the description in lidar_one_hot
        regions = [
            [[l/2, -d/2], [l/2, d/2], [l/2 + dist_threshold, d/2], [l/2 + dist_threshold, -d/2]],
            [[-l/2, d/2], [l/2, d/2], [l/2, d/2 + dist_threshold], [-l/2, d/2 + dist_threshold]],
            [[-l/2, -d/2], [-l/2, d/2], [-l/2 - dist_threshold, d/2], [-l/2 - dist_threshold, -d/2]],
            [[-l/2, -d/2], [l/2, -d/2], [l/2, -d/2 - dist_threshold], [-l/2, -d/2 - dist_threshold]]
        ]

        for region in regions:
            region = np.array(region)
            rotated_region = np.dot(region, rotation_matrix.T)
            rotated_region[:, 0] += cx
            rotated_region[:, 1] += cy
            poly = plt.Polygon(rotated_region, fill=None, edgecolor='r', linestyle='--', linewidth=1)
            plt.gca().add_patch(poly)

        # Define the corners
        corners_local = [
            [l/2, d/2],
            [-l/2, d/2],
            [-l/2, -d/2],
            [l/2, -d/2]
        ]
        angles = [
            (0, np.pi/2),
            (np.pi/2, np.pi),
            (np.pi, 3*np.pi/2),
            (3*np.pi/2, 2*np.pi)
        ]

        for i, corner in enumerate(corners_local):
            center_local = np.array(corner)

            # Generate points for the 1/4 circle in the local coordinate system
            theta = np.linspace(*angles[i], 100)
            x_circle_local = center_local[0] + dist_threshold * np.cos(theta)
            y_circle_local = center_local[1] + dist_threshold * np.sin(theta)
            
            # Combine and transform to global coordinate system
            circle_points_local = np.vstack((x_circle_local, y_circle_local)).T
            circle_points_global = np.dot(circle_points_local, rotation_matrix.T)
            circle_points_global[:, 0] += cx
            circle_points_global[:, 1] += cy

            # Plot the 1/4 circle
            plt.plot(circle_points_global[:, 0], circle_points_global[:, 1], 'r--', linewidth=1)


def shift_one_hot_representation(x, y, n):
    "利用这个函数将坐标表示为one-hot-vector"
    # 使用arctan2来计算角度，该函数会返回[-π, π]之间的角度
    angle = np.arctan2(y, x)
    
    # 如果角度为负，将其转换为[0, 2π)范围内的角度
    if angle < 0:
        angle += 2 * np.pi
        
    # 将[0, 2π)范围分成n等分，每份的角度范围
    angle_increment = (2 * np.pi) / n
    
    # 计算角度所在的范围
    index = int(angle // angle_increment)
    
    # 创建一个n维的零向量
    one_hot_vector = np.zeros(n)
    
    # 将对应范围的索引设置为1
    one_hot_vector[index] = 1
    
    return one_hot_vector

def find_one_position(one_hot_vector):
    return np.argmax(one_hot_vector)

def one_hot_or(vector_a, vector_b):
    # 确保两个向量的维度相同
    if len(vector_a) != len(vector_b):
        raise ValueError("The dimensions of the two vectors must be the same.")
    
    # 执行 point-wise 或运算
    or_vector = np.logical_or(vector_a, vector_b).astype(int)
    return or_vector

class SingleTractor(Vehicle):
    def __init__(self, config):
        super().__init__()
        # physical settings
        self.W = config["w"]
        self.WB = config["wb"]
        self.WD = config["wd"]
        self.RF = config["rf"]
        self.RB = config["rb"]
        self.TR = config["tr"]
        self.TW = config["tw"]
        # action_limit
        self.MAX_STEER = config["max_steer"]
        self.V_MAX = config["v_max"]
        
        self.SAFE_D = config["safe_d"]
        
        self.state = (0.0, 0.0, np.deg2rad(0.0))
        
    def reset_equilibrium(self, x, y, yaw):
        self.state = (x, y, yaw)
        
    def reset(self, *args):
        self.state = tuple(arg for arg in args if arg is not None)
        
    def _is_jack_knife(self):
        return False
    
    def step(self, action: np.ndarray, dt: float=0.1, backward: bool = True, scale: bool = True, kinematics_type: str = "velocity"):
        # the action from the action_space, [-1, 1]
        
        if kinematics_type == "velocity":
            x, y, yaw = self.state
            v, steer = action
            if scale:
                v = self.velocity_scale(v)
            if not backward:
                v = abs(v)
            if scale:
                steer = self.steer_scale(steer)
            x_ = x + np.cos(yaw) * v * dt
            y_ = y + np.sin(yaw) * v * dt
            yaw_ = self.pi_2_pi(
                yaw + v * dt * np.tan(steer) / self.WB
                )
            self.state = (x_, y_, yaw_)
        elif kinematics_type == "accelerate":
            pass
    
    
    def observe(self):
        x, y, yaw = self.state
        return np.array([x, y, yaw, 0, 0, 0], dtype=np.float64)
    
    def observe_full(self):
        x, y, yaw = self.state
        return np.array([x, y, yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    
    def velocity_scale(self, velocity: float) -> float:
        return velocity * self.V_MAX
    
    def steer_scale(self, steer: float) -> float:
        return steer * self.MAX_STEER
    
    def plot(self, ax: Axes, action: np.ndarray, color: str = 'black') -> None:
        '''
        Car: three_trailer model class
        x: center of rear wheel
        y: center of rear wheel
        yaw: yaw of rear wheel
        yawt1: yaw of trailer1
        yawt2: yaw of trailer2
        yawt3: yaw of trailer3
        steer: steer of front wheel
        '''
        _, steer = action
        steer = self.steer_scale(steer)
        
        self.plot_tractor_and_four_wheels(ax, steer, color=color)
        self.plot_arrow(ax, l=self.WB * 0.8, color=color)
    
    def plot_arrow(self, ax: Axes, l: float, color: str = 'black') -> None:
        x, y, yaw = self.state
    
        angle = np.deg2rad(30)
        d = 0.3 * l
        w = 1

        x_start = x
        y_start = y
        x_end = x + l * np.cos(yaw)
        y_end = y + l * np.sin(yaw)

        theta_hat_L = yaw + np.pi - angle
        theta_hat_R = yaw + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=color, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=color, linewidth=w)
    
    def plot_tractor_and_four_wheels(self, ax: Axes, steer: float, color: str = 'black') -> None:
        # get current state from the class
        x, y, yaw = self.state
        # plot initial tractor
        car = np.array([[-self.RB, -self.RB, self.RF, self.RF, -self.RB],
                        [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
        wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
                        [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
        frWheel = wheel.copy()
        flWheel = wheel.copy()
        rrWheel = wheel.copy()
        rlWheel = wheel.copy()
        # rotate to yaw
        Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        car = np.dot(Rot1, car)
        # move to the current position
        car += np.array([[x], [y]])
        # plot tractor
        ax.plot(car[0, :], car[1, :], color, linewidth=1)
        Rot2 = np.array([[np.cos(steer), -np.sin(steer)],
                        [np.sin(steer), np.cos(steer)]])
        frWheel = np.dot(Rot2, frWheel)
        flWheel = np.dot(Rot2, flWheel)
        frWheel += np.array([[self.WB], [-self.WD / 2]])
        flWheel += np.array([[self.WB], [self.WD / 2]])
        rrWheel[1, :] -= self.WD / 2
        rlWheel[1, :] += self.WD / 2
        frWheel = np.dot(Rot1, frWheel)
        flWheel = np.dot(Rot1, flWheel)
        rrWheel = np.dot(Rot1, rrWheel)
        rlWheel = np.dot(Rot1, rlWheel)
        frWheel += np.array([[x], [y]])
        flWheel += np.array([[x], [y]])
        rrWheel += np.array([[x], [y]])
        rlWheel += np.array([[x], [y]])
        
        # plot tractor 4 wheels
        ax.plot(frWheel[0, :], frWheel[1, :], color, linewidth=1)
        ax.plot(rrWheel[0, :], rrWheel[1, :], color, linewidth=1)
        ax.plot(flWheel[0, :], flWheel[1, :], color, linewidth=1)
        ax.plot(rlWheel[0, :], rlWheel[1, :], color, linewidth=1)
        
    def is_collision(self, ox: List[float], oy: List[float]) -> bool:
        """
        judge whether the current state crased with obs with sample points given
        """
        points = np.array(list(zip(ox, oy)))
        tree = cKDTree(points)
        x, y, yaw = self.state
        # first trailer test collision
        d = self.SAFE_D
                    
        # check the tractor collision
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0 + d

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        ids = tree.query_ball_point([cx, cy], rc)

        if ids:
            for i in ids:
                xo = ox[i] - cx
                yo = oy[i] - cy

                dx_car = xo * np.cos(yaw) + yo * np.sin(yaw)
                dy_car = -xo * np.sin(yaw) + yo * np.cos(yaw)

                if abs(dx_car) <= rc and \
                        abs(dy_car) <= self.W / 2.0 + d:
                    return True

        return False
    

class OneTrailer(Vehicle):
    def __init__(self, config):
        super().__init__()
        # physical settings
        self.W = config["w"]
        self.WB = config["wb"]
        self.WD = config["wd"]
        self.RF = config["rf"]
        self.RB = config["rb"]
        self.TR = config["tr"]
        self.TW = config["tw"]
        self.RTR = config['rtr']
        self.RTF = config['rtf']
        self.RTB = config['rtb']
        # action_limit
        self.MAX_STEER = config["max_steer"]
        self.V_MAX = config["v_max"]
        
        self.SAFE_D = config["safe_d"]
        self.XI_MAX = config["xi_max"]
        
        self.state = (
            0.0, 0.0, np.deg2rad(0.0), 
            np.deg2rad(0.0)
            )
        
    def reset_equilibrium(self, x, y, yaw):
        self.state = (x, y, yaw, yaw)
        
    def reset(self, *args):
        self.state = tuple(arg for arg in args if arg is not None)
    
    def observe(self):
        x, y, yaw, yawt1 = self.state
        return np.array([x, y, yaw, yawt1, 0, 0], dtype=np.float64)
    
    def observe_full(self):
        x, y, yaw, yawt1 = self.state
        x_trailer1, y_trailer1 = self.get_center_trailer(1)
        return np.array([x, y, yaw, x_trailer1, y_trailer1, yawt1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        
    
    def step(self, action: np.ndarray, dt: float=0.1, backward: bool = True, kinematics_type: str = "velocity"):
        if kinematics_type == "velocity":
            x, y, yaw, yawt1 = self.state
            v, steer = action
            v = self.velocity_scale(v)
            if not backward:
                v = abs(v)
            steer = self.steer_scale(steer)
            x_ = x + np.cos(yaw) * v * dt
            y_ = y + np.sin(yaw) * v * dt
            yaw_ = self.pi_2_pi(
                yaw + v * dt * np.tan(steer) / self.WB
                )
            yawt1_ = self.pi_2_pi(
                yawt1 + v * dt / self.RTR * np.sin(yaw - yawt1)
                )

            self.state = (x_, y_, yaw_, yawt1_)
        elif kinematics_type == "accelerate":
            pass
    
    def _is_jack_knife(self):
        x, y, yaw, yawt1 = self.state
        xi4 = self.pi_2_pi(yawt1 - yaw)
        if abs(xi4) > self.XI_MAX:
            return True
        return False
    
    def velocity_scale(self, velocity: float) -> float:
        return velocity * self.V_MAX
    
    def steer_scale(self, steer: float) -> float:
        return steer * self.MAX_STEER
    
    def plot(self, ax: Axes, action: np.ndarray, color: str = 'black') -> None:
        '''
        Car: three_trailer model class
        x: center of rear wheel
        y: center of rear wheel
        yaw: yaw of rear wheel
        yawt1: yaw of trailer1
        yawt2: yaw of trailer2
        yawt3: yaw of trailer3
        steer: steer of front wheel
        '''
        _, steer = action
        steer = self.steer_scale(steer)
        
        self.plot_tractor_and_four_wheels(ax, steer, color=color)
        for i in range(1, 2):
            self.plot_trailer_and_two_wheels(ax, number=i, color=color)
        for i in range(1, 2):
            self.plot_link(ax, number=i, color=color)
        
        self.plot_arrow(ax, l=self.WB * 0.8, color=color)
    
    def plot_link(self, ax: Axes, number: int, color: str = 'black') -> None:
        x_trailer, y_trailer = self.get_center_trailer(number)
        if number == 1:
            x, y, _, _ = self.state
            ax.plot(np.array([x, x_trailer]), np.array([y, y_trailer]), color, linewidth=1)
        else:
            x_trailer_front, y_trailer_front = self.get_center_trailer(number - 1)
            ax.plot(np.array([x_trailer_front, x_trailer]), np.array([y_trailer_front, y_trailer]), color, linewidth=1)
    
    def plot_arrow(self, ax: Axes, l: float, color: str = 'black') -> None:
        x, y, yaw, _ = self.state
    
        angle = np.deg2rad(30)
        d = 0.3 * l
        w = 1

        x_start = x
        y_start = y
        x_end = x + l * np.cos(yaw)
        y_end = y + l * np.sin(yaw)

        theta_hat_L = yaw + np.pi - angle
        theta_hat_R = yaw + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=color, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=color, linewidth=w)
    
    def plot_tractor_and_four_wheels(self, ax: Axes, steer: float, color: str = 'black') -> None:
        # get current state from the class
        x, y, yaw, _ = self.state
        # plot initial tractor
        tractor = np.array([[-self.RB, -self.RB, self.RF, self.RF, -self.RB],
                        [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
        wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
                        [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
        frWheel = wheel.copy()
        flWheel = wheel.copy()
        rrWheel = wheel.copy()
        rlWheel = wheel.copy()
        # rotate to yaw
        Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        tractor = np.dot(Rot1, tractor)
        # move to the current position
        tractor += np.array([[x], [y]])
        # plot tractor
        ax.plot(tractor[0, :], tractor[1, :], color, linewidth=1)
        Rot2 = np.array([[np.cos(steer), -np.sin(steer)],
                        [np.sin(steer), np.cos(steer)]])
        frWheel = np.dot(Rot2, frWheel)
        flWheel = np.dot(Rot2, flWheel)
        frWheel += np.array([[self.WB], [-self.WD / 2]])
        flWheel += np.array([[self.WB], [self.WD / 2]])
        rrWheel[1, :] -= self.WD / 2
        rlWheel[1, :] += self.WD / 2
        frWheel = np.dot(Rot1, frWheel)
        flWheel = np.dot(Rot1, flWheel)
        rrWheel = np.dot(Rot1, rrWheel)
        rlWheel = np.dot(Rot1, rlWheel)
        frWheel += np.array([[x], [y]])
        flWheel += np.array([[x], [y]])
        rrWheel += np.array([[x], [y]])
        rlWheel += np.array([[x], [y]])
        
        # plot tractor 4 wheels
        ax.plot(frWheel[0, :], frWheel[1, :], color, linewidth=1)
        ax.plot(rrWheel[0, :], rrWheel[1, :], color, linewidth=1)
        ax.plot(flWheel[0, :], flWheel[1, :], color, linewidth=1)
        ax.plot(rlWheel[0, :], rlWheel[1, :], color, linewidth=1)
    
    def get_center_trailer(self, number: int) -> Tuple[float, float]:
        """
        get the center of tractor directly from the self.state
        """
        x, y, yaw, yawt1 = self.state
        if number == 1:
            x_trailer = x - self.RTR * np.cos(yawt1)
            y_trailer = y - self.RTR * np.sin(yawt1)
            
        return (x_trailer, y_trailer)
    
    def get_center_tractor(self) -> Tuple[float, float]:
        """get the center position of tractor"""
        x, y, yaw, _, _, _ = self.state
        x += self.WB / 2 * np.cos(yaw)
        y += self.WB / 2 * np.sin(yaw)
        return (x, y)
        
    def plot_trailer_and_two_wheels(self, ax: Axes, number: int, color: str = 'black') -> None:
        
        x, y, yaw, yawt1 = self.state
        
        trail = np.array([[-self.RTB, -self.RTB, -self.RTF, -self.RTF, -self.RTB],
                        [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
        trail += np.array([[self.RTR],[0]])
        wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
                        [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
        ltWheel = wheel.copy()
        rtWheel = wheel.copy()
        
        if number == 1:
            Rot = np.array([[np.cos(yawt1), -np.sin(yawt1)],
                            [np.sin(yawt1), np.cos(yawt1)]])
            trail = np.dot(Rot, trail)
            trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
            trail += np.array([[x], [y]])
            ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
            
            ltWheel = np.dot(Rot, ltWheel)
            rtWheel = np.dot(Rot, rtWheel)
            x_trailer, y_trailer = self.get_center_trailer(number=1)
            ltWheel += np.array([[x_trailer],[y_trailer]])
            rtWheel += np.array([[x_trailer],[y_trailer]])
            ltWheel -= np.array([[self.WD/2 * np.sin(yawt1)],[-self.WD/2 * np.cos(yawt1)]])
            rtWheel += np.array([[self.WD/2 * np.sin(yawt1)],[-self.WD/2 * np.cos(yawt1)]])
            
            ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
            ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
            
    def is_collision(self, ox: List[float], oy: List[float]) -> bool:
        '''
        check whether there is collision
        Inputs:
        x, y, yaw, yawt1, yawt2, yawt3: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        points = np.array(list(zip(ox, oy)))
        tree = cKDTree(points)
        x, y, yaw, yawt1 = self.state
        d = self.SAFE_D
        
        # first trailer test collision
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0 + d #half length of trailer1 plus d

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)

        idst1 = tree.query_ball_point([ctx1, cty1], rt1)

        if idst1:
            for i in idst1:
                xot1 = ox[i] - ctx1
                yot1 = oy[i] - cty1

                dx_trail1 = xot1 * np.cos(yawt1) + yot1 * np.sin(yawt1)
                dy_trail1 = -xot1 * np.sin(yawt1) + yot1 * np.cos(yawt1)

                if abs(dx_trail1) <= rt1 and \
                        abs(dy_trail1) <= self.W / 2.0 + d:
                    return True
                    
        # check the tractor collision
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0 + d

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        ids = tree.query_ball_point([cx, cy], rc)

        if ids:
            for i in ids:
                xo = ox[i] - cx
                yo = oy[i] - cy

                dx_car = xo * np.cos(yaw) + yo * np.sin(yaw)
                dy_car = -xo * np.sin(yaw) + yo * np.cos(yaw)

                if abs(dx_car) <= rc and \
                        abs(dy_car) <= self.W / 2.0 + d:
                    return True

        return False

class TwoTrailer(Vehicle):
    def __init__(self, config):
        super().__init__()
        # physical settings
        self.W = config["w"]
        self.WB = config["wb"]
        self.WD = config["wd"]
        self.RF = config["rf"]
        self.RB = config["rb"]
        self.TR = config["tr"]
        self.TW = config["tw"]
        self.RTR = config['rtr']
        self.RTF = config['rtf']
        self.RTB = config['rtb']
        self.RTR2 = config['rtr2']
        self.RTF2 = config['rtf2']
        self.RTB2 = config['rtb2']
        # action_limit
        self.MAX_STEER = config["max_steer"]
        self.V_MAX = config["v_max"]
        
        self.SAFE_D = config["safe_d"]
        self.XI_MAX = config["xi_max"]
        
        self.state = (
            0.0, 0.0, np.deg2rad(0.0), 
            np.deg2rad(0.0), np.deg2rad(0.0)
            )
        
    def reset_equilibrium(self, x, y, yaw):
        self.state = (x, y, yaw, yaw, yaw)
        
    def reset(self, *args):
        self.state = tuple(arg for arg in args if arg is not None)
        
    def observe(self):
        x, y, yaw, yawt1, yawt2 = self.state
        return np.array([x, y, yaw, yawt1, yawt2, 0], dtype=np.float64)
    
    def observe_full(self):
        x, y, yaw, yawt1, yawt2 = self.state
        x_trailer1, y_trailer1 = self.get_center_trailer(1)
        x_trailer2, y_trailer2 = self.get_center_trailer(2)
        return np.array([x, y, yaw, x_trailer1, y_trailer1, yawt1, x_trailer2, y_trailer2, yawt2, 0, 0, 0], dtype=np.float64)
    
    def step(self, action: np.ndarray, dt: float=0.1, backward: bool = True, kinematics_type: str = "velocity"):
        if kinematics_type == "velocity":
            x, y, yaw, yawt1, yawt2 = self.state
            v, steer = action
            v = self.velocity_scale(v)
            if not backward:
                v = abs(v)
            steer = self.steer_scale(steer)
            
            x_ = x + np.cos(yaw) * v * dt
            y_ = y + np.sin(yaw) * v * dt
            yaw_ = self.pi_2_pi(
                yaw + v * dt * np.tan(steer) / self.WB
                )
            
            yawt1_ = self.pi_2_pi(
                yawt1 + v * dt / self.RTR * np.sin(yaw - yawt1)
                )
            yawt2_ = self.pi_2_pi(
                yawt2 + v * dt / self.RTR2 * np.sin(yawt1 - yawt2) * np.cos(yaw - yawt1)
                )
            self.state = (x_, y_, yaw_, yawt1_, yawt2_)
        elif kinematics_type == "accelerate":
            pass
    
    def velocity_scale(self, velocity: float) -> float:
        return velocity * self.V_MAX
    
    def steer_scale(self, steer: float) -> float:
        return steer * self.MAX_STEER
    
    def _is_jack_knife(self):
        x, y, yaw, yawt1, yawt2 = self.state
        xi4 = self.pi_2_pi(yawt1 - yaw)
        xi5 = self.pi_2_pi(yawt2 - yawt1)
        xi_list = [xi4, xi5]
        max_abs_xi = abs(max(xi_list, key=abs))
        if max_abs_xi > self.XI_MAX:
            return True
        return False
    
    # def plot(self, ax: Axes, action: np.ndarray, color: str = 'black') -> None:
    #     '''
    #     Car: three_trailer model class
    #     x: center of rear wheel
    #     y: center of rear wheel
    #     yaw: yaw of rear wheel
    #     yawt1: yaw of trailer1
    #     yawt2: yaw of trailer2
    #     yawt3: yaw of trailer3
    #     steer: steer of front wheel
    #     '''
    #     _, steer = action
    #     steer = self.steer_scale(steer)
        
    #     self.plot_tractor_and_four_wheels(ax, steer, color=color)
    #     for i in range(1, 3):
    #         self.plot_trailer_and_two_wheels(ax, number=i, color=color)
    #     for i in range(1, 3):
    #         self.plot_link(ax, number=i, color=color)
        
    #     self.plot_arrow(ax, l=self.WB * 0.8, color=color)
    
    def plot(self, ax: Axes, action: np.ndarray, color: str = 'black', is_full: bool = False) -> None:
        '''
        Car: three_trailer model class
        x: center of rear wheel
        y: center of rear wheel
        yaw: yaw of rear wheel
        yawt1: yaw of trailer1
        yawt2: yaw of trailer2
        yawt3: yaw of trailer3
        steer: steer of front wheel
        '''
        _, steer = action
        steer = self.steer_scale(steer)
        
        self.plot_tractor_and_four_wheels(ax, steer, color=color, is_full=is_full)
        for i in range(1, 3):
            self.plot_trailer_and_two_wheels(ax, number=i, color=color, is_full=is_full)
        for i in range(1, 3):
            self.plot_link(ax, number=i, color=color)
        
        self.plot_arrow(ax, l=self.WB * 0.8, color=color)
    
    def plot_link(self, ax: Axes, number: int, color: str = 'black') -> None:
        x_trailer, y_trailer = self.get_center_trailer(number)
        if number == 1:
            x, y, _, _, _ = self.state
            ax.plot(np.array([x, x_trailer]), np.array([y, y_trailer]), color, linewidth=1)
        else:
            x_trailer_front, y_trailer_front = self.get_center_trailer(number - 1)
            ax.plot(np.array([x_trailer_front, x_trailer]), np.array([y_trailer_front, y_trailer]), color, linewidth=1)
    
    def plot_arrow(self, ax: Axes, l: float, color: str = 'black') -> None:
        x, y, yaw, _, _ = self.state
    
        angle = np.deg2rad(30)
        d = 0.3 * l
        w = 1

        x_start = x
        y_start = y
        x_end = x + l * np.cos(yaw)
        y_end = y + l * np.sin(yaw)

        theta_hat_L = yaw + np.pi - angle
        theta_hat_R = yaw + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=color, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=color, linewidth=w)
    
    # def plot_tractor_and_four_wheels(self, ax: Axes, steer: float, color: str = 'black') -> None:
    #     # get current state from the class
    #     x, y, yaw, _, _ = self.state
    #     # plot initial tractor
    #     tractor = np.array([[-self.RB, -self.RB, self.RF, self.RF, -self.RB],
    #                     [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
    #     wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
    #                     [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
    #     frWheel = wheel.copy()
    #     flWheel = wheel.copy()
    #     rrWheel = wheel.copy()
    #     rlWheel = wheel.copy()
    #     # rotate to yaw
    #     Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
    #                     [np.sin(yaw), np.cos(yaw)]])
    #     tractor = np.dot(Rot1, tractor)
    #     # move to the current position
    #     tractor += np.array([[x], [y]])
    #     # plot tractor
    #     ax.plot(tractor[0, :], tractor[1, :], color, linewidth=1)
    #     Rot2 = np.array([[np.cos(steer), -np.sin(steer)],
    #                     [np.sin(steer), np.cos(steer)]])
    #     frWheel = np.dot(Rot2, frWheel)
    #     flWheel = np.dot(Rot2, flWheel)
    #     frWheel += np.array([[self.WB], [-self.WD / 2]])
    #     flWheel += np.array([[self.WB], [self.WD / 2]])
    #     rrWheel[1, :] -= self.WD / 2
    #     rlWheel[1, :] += self.WD / 2
    #     frWheel = np.dot(Rot1, frWheel)
    #     flWheel = np.dot(Rot1, flWheel)
    #     rrWheel = np.dot(Rot1, rrWheel)
    #     rlWheel = np.dot(Rot1, rlWheel)
    #     frWheel += np.array([[x], [y]])
    #     flWheel += np.array([[x], [y]])
    #     rrWheel += np.array([[x], [y]])
    #     rlWheel += np.array([[x], [y]])
        
    #     # plot tractor 4 wheels
    #     ax.plot(frWheel[0, :], frWheel[1, :], color, linewidth=1)
    #     ax.plot(rrWheel[0, :], rrWheel[1, :], color, linewidth=1)
    #     ax.plot(flWheel[0, :], flWheel[1, :], color, linewidth=1)
    #     ax.plot(rlWheel[0, :], rlWheel[1, :], color, linewidth=1)
    
    def plot_tractor_and_four_wheels(self, ax: Axes, steer: float, color: str = 'black', is_full: bool = False) -> None:
        # get current state from the class
        x, y, yaw, _, _ = self.state
        # plot initial tractor
        tractor = np.array([[-self.RB, -self.RB, self.RF, self.RF, -self.RB],
                            [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
        # rotate to yaw
        Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        tractor = np.dot(Rot1, tractor)
        # move to the current position
        tractor += np.array([[x], [y]])

        if is_full:
            # fill tractor
            ax.fill(tractor[0, :], tractor[1, :], color)
        else:
            # plot tractor
            ax.plot(tractor[0, :], tractor[1, :], color, linewidth=1)
            wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
                            [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
            frWheel = wheel.copy()
            flWheel = wheel.copy()
            rrWheel = wheel.copy()
            rlWheel = wheel.copy()
            Rot2 = np.array([[np.cos(steer), -np.sin(steer)],
                            [np.sin(steer), np.cos(steer)]])
            frWheel = np.dot(Rot2, frWheel)
            flWheel = np.dot(Rot2, flWheel)
            frWheel += np.array([[self.WB], [-self.WD / 2]])
            flWheel += np.array([[self.WB], [self.WD / 2]])
            rrWheel[1, :] -= self.WD / 2
            rlWheel[1, :] += self.WD / 2
            frWheel = np.dot(Rot1, frWheel)
            flWheel = np.dot(Rot1, flWheel)
            rrWheel = np.dot(Rot1, rrWheel)
            rlWheel = np.dot(Rot1, rlWheel)
            frWheel += np.array([[x], [y]])
            flWheel += np.array([[x], [y]])
            rrWheel += np.array([[x], [y]])
            rlWheel += np.array([[x], [y]])

            # plot tractor 4 wheels
            ax.plot(frWheel[0, :], frWheel[1, :], color, linewidth=1)
            ax.plot(rrWheel[0, :], rrWheel[1, :], color, linewidth=1)
            ax.plot(flWheel[0, :], flWheel[1, :], color, linewidth=1)
            ax.plot(rlWheel[0, :], rlWheel[1, :], color, linewidth=1)
    
    def get_center_trailer(self, number: int) -> Tuple[float, float]:
        """
        get the center of tractor directly from the self.state
        """
        x, y, yaw, yawt1, yawt2 = self.state
        if number == 1:
            x_trailer = x - self.RTR * np.cos(yawt1)
            y_trailer = y - self.RTR * np.sin(yawt1)
            
        elif number == 2:
            x_trailer = x - self.RTR * np.cos(yawt1)
            y_trailer = y - self.RTR * np.sin(yawt1)
            x_trailer -= self.RTR2 * np.cos(yawt2)
            y_trailer -= self.RTR2 * np.sin(yawt2)
            
        return (x_trailer, y_trailer)
    
    def get_center_tractor(self) -> Tuple[float, float]:
        """get the center position of tractor"""
        x, y, yaw, _, _, _ = self.state
        x += self.WB / 2 * np.cos(yaw)
        y += self.WB / 2 * np.sin(yaw)
        return (x, y)
        
    # def plot_trailer_and_two_wheels(self, ax: Axes, number: int, color: str = 'black') -> None:
        
    #     x, y, yaw, yawt1, yawt2 = self.state
        
    #     trail = np.array([[-self.RTB, -self.RTB, -self.RTF, -self.RTF, -self.RTB],
    #                     [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
    #     trail += np.array([[self.RTR],[0]])
    #     wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
    #                     [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
    #     ltWheel = wheel.copy()
    #     rtWheel = wheel.copy()
        
    #     if number == 1:
    #         Rot = np.array([[np.cos(yawt1), -np.sin(yawt1)],
    #                         [np.sin(yawt1), np.cos(yawt1)]])
    #         trail = np.dot(Rot, trail)
    #         trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
    #         trail += np.array([[x], [y]])
    #         ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
            
    #         ltWheel = np.dot(Rot, ltWheel)
    #         rtWheel = np.dot(Rot, rtWheel)
    #         x_trailer, y_trailer = self.get_center_trailer(number=1)
    #         ltWheel += np.array([[x_trailer],[y_trailer]])
    #         rtWheel += np.array([[x_trailer],[y_trailer]])
    #         ltWheel -= np.array([[self.WD/2 * np.sin(yawt1)],[-self.WD/2 * np.cos(yawt1)]])
    #         rtWheel += np.array([[self.WD/2 * np.sin(yawt1)],[-self.WD/2 * np.cos(yawt1)]])
            
    #         ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
    #         ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
        
    #     else:
    #         Rot = np.array([[np.cos(yawt2), -np.sin(yawt2)],
    #                         [np.sin(yawt2), np.cos(yawt2)]])
    #         trail = np.dot(Rot, trail)
    #         trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
    #         trail -= np.array([[self.RTR2 * np.cos(yawt2)],[self.RTR2 * np.sin(yawt2)]])
    #         trail += np.array([[x], [y]])
    #         ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
            
    #         ltWheel = np.dot(Rot, ltWheel)
    #         rtWheel = np.dot(Rot, rtWheel)
    #         x_trailer, y_trailer = self.get_center_trailer(number=2)
    #         ltWheel += np.array([[x_trailer],[y_trailer]])
    #         rtWheel += np.array([[x_trailer],[y_trailer]])
    #         ltWheel -= np.array([[self.WD/2 * np.sin(yawt2)],[-self.WD/2 * np.cos(yawt2)]])
    #         rtWheel += np.array([[self.WD/2 * np.sin(yawt2)],[-self.WD/2 * np.cos(yawt2)]])
    #         ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
    #         ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
    
    def plot_trailer_and_two_wheels(self, ax: Axes, number: int, color: str = 'black', is_full: bool = False) -> None:
        # get current state from the class
        x, y, yaw, yawt1, yawt2 = self.state

        # plot initial trailer
        trail = np.array([[-self.RTB, -self.RTB, self.RTF, self.RTF, -self.RTB],
                          [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
        trail += np.array([[self.RTR], [0]])

        # plot initial wheels
        wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
                          [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
        ltWheel = wheel.copy()
        rtWheel = wheel.copy()

        if is_full:
            # fill trailer
            ax.fill(trail[0, :], trail[1, :], color)
        else:
            # plot trailer and its two wheels
            if number == 1:
                # rotate to yawt1
                Rot = np.array([[np.cos(yawt1), -np.sin(yawt1)],
                                [np.sin(yawt1), np.cos(yawt1)]])
                trail = np.dot(Rot, trail)
                trail -= np.array([[self.RTR * np.cos(yawt1)], [self.RTR * np.sin(yawt1)]])
                trail += np.array([[x], [y]])
                ax.plot(trail[0, :], trail[1, :], color, linewidth=1)

                ltWheel = np.dot(Rot, ltWheel)
                rtWheel = np.dot(Rot, rtWheel)
                x_trailer, y_trailer = self.get_center_trailer(number=1)
                ltWheel += np.array([[x_trailer], [y_trailer]])
                rtWheel += np.array([[x_trailer], [y_trailer]])
                ltWheel -= np.array([[self.WD / 2 * np.sin(yawt1)], [-self.WD / 2 * np.cos(yawt1)]])
                rtWheel += np.array([[self.WD / 2 * np.sin(yawt1)], [-self.WD / 2 * np.cos(yawt1)]])

                ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
                ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)

            else:
                # rotate to yawt2
                Rot = np.array([[np.cos(yawt2), -np.sin(yawt2)],
                                [np.sin(yawt2), np.cos(yawt2)]])
                trail = np.dot(Rot, trail)
                trail -= np.array([[self.RTR * np.cos(yawt1)], [self.RTR * np.sin(yawt1)]])
                trail -= np.array([[self.RTR2 * np.cos(yawt2)], [self.RTR2 * np.sin(yawt2)]])
                trail += np.array([[x], [y]])
                ax.plot(trail[0, :], trail[1, :], color, linewidth=1)

                ltWheel = np.dot(Rot, ltWheel)
                rtWheel = np.dot(Rot, rtWheel)
                x_trailer, y_trailer = self.get_center_trailer(number=2)
                ltWheel += np.array([[x_trailer], [y_trailer]])
                rtWheel += np.array([[x_trailer], [y_trailer]])
                ltWheel -= np.array([[self.WD / 2 * np.sin(yawt2)], [-self.WD / 2 * np.cos(yawt2)]])
                rtWheel += np.array([[self.WD / 2 * np.sin(yawt2)], [-self.WD / 2 * np.cos(yawt2)]])

                ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
                ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
            
    def is_collision(self, ox: List[float], oy: List[float]) -> bool:
        '''
        check whether there is collision
        Inputs:
        x, y, yaw, yawt1, yawt2, yawt3: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        points = np.array(list(zip(ox, oy)))
        tree = cKDTree(points)
        x, y, yaw, yawt1, yawt2 = self.state
        d = self.SAFE_D
        
        # first trailer test collision
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0 + d #half length of trailer1 plus d

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)

        idst1 = tree.query_ball_point([ctx1, cty1], rt1)

        if idst1:
            for i in idst1:
                xot1 = ox[i] - ctx1
                yot1 = oy[i] - cty1

                dx_trail1 = xot1 * np.cos(yawt1) + yot1 * np.sin(yawt1)
                dy_trail1 = -xot1 * np.sin(yawt1) + yot1 * np.cos(yawt1)

                if abs(dx_trail1) <= rt1 and \
                        abs(dy_trail1) <= self.W / 2.0 + d:
                    return True
        # check the second trailer collision
        deltal2 = (self.RTF2 + self.RTB2) / 2.0
        rt2 = (self.RTB2 - self.RTF2) / 2.0 + d
        
        ctx2 = ctx1 - deltal2 * np.cos(yawt2)
        cty2 = cty1 - deltal2 * np.sin(yawt2)
        
        idst2 = tree.query_ball_point([ctx2, cty2], rt2)
        
        if idst2:
            for i in idst2:
                xot2 = ox[i] - ctx2
                yot2 = oy[i] - cty2
                
                dx_trail2 = xot2 * np.cos(yawt2) + yot2 * np.sin(yawt2)
                dy_trail2 = -xot2 * np.cos(yawt2) + yot2 * np.cos(yawt2)
                
                if abs(dx_trail2) <= rt2 and \
                    abs(dy_trail2) <= self.W / 2.0 + d:
                        return True
                    
        # check the tractor collision
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0 + d

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        ids = tree.query_ball_point([cx, cy], rc)

        if ids:
            for i in ids:
                xo = ox[i] - cx
                yo = oy[i] - cy

                dx_car = xo * np.cos(yaw) + yo * np.sin(yaw)
                dy_car = -xo * np.sin(yaw) + yo * np.cos(yaw)

                if abs(dx_car) <= rc and \
                        abs(dy_car) <= self.W / 2.0 + d:
                    return True

        return False
    
class ThreeTrailer(Vehicle):
    def __init__(self, config):
        super().__init__()
        # physical settings
        self.W = config["w"]
        self.WB = config["wb"]
        self.WD = config["wd"]
        self.RF = config["rf"]
        self.RB = config["rb"]
        self.TR = config["tr"]
        self.TW = config["tw"]
        self.RTR = config['rtr']
        self.RTF = config['rtf']
        self.RTB = config['rtb']
        self.RTR2 = config['rtr2']
        self.RTF2 = config['rtf2']
        self.RTB2 = config['rtb2']
        self.RTR3 = config['rtr3']
        self.RTF3 = config['rtf3']
        self.RTB3 = config['rtb3']
        # action_limit
        self.MAX_STEER = config["max_steer"]
        self.V_MAX = config["v_max"]
        
        self.SAFE_D = config["safe_d"]
        self.SAFE_METRIC = config["safe_metric"]
        self.XI_MAX = config["xi_max"]
        self.state = (
            0.0, 0.0, np.deg2rad(0.0), 
            np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)
            )
    
    def reset_equilibrium(self, x, y, yaw):
        self.state = (x, y, yaw, yaw, yaw, yaw)
        
    def calculate_configurations_given_equilibrium(self, equilibrium):
        # return the rectangle configuration
        x, y, yaw0, yawt1, yawt2, yawt3 = equilibrium
        assert yaw0 == yawt1 and yawt1 == yawt2 and yawt2 == yawt3, "must be equilibrium state"
        x_center = x - ((self.RF + self.RTR + self.RTR2 + self.RTR3 + self.RTF3)/ 2 - self.RF) * np.cos(yaw0)
        y_center = y - ((self.RF + self.RTR + self.RTR2 + self.RTR3 + self.RTF3)/ 2 - self.RF) * np.sin(yaw0)
        return (x_center, y_center, self.RF + self.RTR + self.RTR2 + self.RTR3 + self.RTF3, self.W, yaw0)
        
    def reset(self, *args):
        """you need to plug the args as reset_equilibrium"""
        self.state = tuple(arg for arg in args if arg is not None)
        
    def observe(self):
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        return np.array([x, y, yaw, yawt1, yawt2, yawt3], dtype=np.float64)
    
    def observe_full(self):
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        x_trailer1, y_trailer1 = self.get_center_trailer(1)
        x_trailer2, y_trailer2 = self.get_center_trailer(2)
        x_trailer3, y_trailer3 = self.get_center_trailer(3)
        return np.array([x, y, yaw, x_trailer1, y_trailer1, yawt1, x_trailer2, y_trailer2, yawt2, x_trailer3, y_trailer3, yawt3], dtype=np.float64)
    
    
    def step(self, action: np.ndarray, dt: float=0.1, backward: bool = True, kinematics_type: str = "velocity"):
        if kinematics_type == "velocity":
            x, y, yaw, yawt1, yawt2, yawt3 = self.state
            v, steer = action
            v = self.velocity_scale(v)
            if not backward:
                v = abs(v)
            # if v < 0 and np.abs(steer) < 1:
            #     steer = 0
            steer = self.steer_scale(steer)
            
            
            x_ = x + np.cos(yaw) * v * dt
            y_ = y + np.sin(yaw) * v * dt
            yaw_ = self.pi_2_pi(
                yaw + v * dt * np.tan(steer) / self.WB
                )
            
            yawt1_ = self.pi_2_pi(
                yawt1 + v * dt / self.RTR * np.sin(yaw - yawt1)
                )
            yawt2_ = self.pi_2_pi(
                yawt2 + v * dt / self.RTR2 * np.sin(yawt1 - yawt2) * np.cos(yaw - yawt1)
                )
            yawt3_ = self.pi_2_pi(
                yawt3 + v * dt / self.RTR3 * np.sin(yawt2 - yawt3) * np.cos(yawt1 - yawt2) * np.cos(yaw - yawt1)
                )
            self.state = (x_, y_, yaw_, yawt1_, yawt2_, yawt3_)
        elif kinematics_type == "accelerate":
            pass
    
    def velocity_scale(self, velocity: float) -> float:
        return velocity * self.V_MAX
    
    def steer_scale(self, steer: float) -> float:
        return steer * self.MAX_STEER
    
    def _is_jack_knife(self):
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        xi4 = self.pi_2_pi(yawt1 - yaw)
        xi5 = self.pi_2_pi(yawt2 - yawt1)
        xi6 = self.pi_2_pi(yawt3 - yawt2)
        xi_list = [xi4, xi5, xi6]
        max_abs_xi = abs(max(xi_list, key=abs))
        if max_abs_xi > self.XI_MAX:
            return True
        return False
    
    # def plot(self, ax: Axes, action: np.ndarray, color: str = 'black') -> None:
    #     '''
    #     Car: three_trailer model class
    #     x: center of rear wheel
    #     y: center of rear wheel
    #     yaw: yaw of rear wheel
    #     yawt1: yaw of trailer1
    #     yawt2: yaw of trailer2
    #     yawt3: yaw of trailer3
    #     steer: steer of front wheel
    #     '''
    #     _, steer = action
    #     steer = self.steer_scale(steer)
        
    #     self.plot_tractor_and_four_wheels(ax, steer, color=color)
    #     for i in range(1, 4):
    #         self.plot_trailer_and_two_wheels(ax, number=i, color=color)
    #     for i in range(1, 4):
    #         self.plot_link(ax, number=i, color=color)
        
    #     self.plot_arrow(ax, l=self.WB * 0.8, color=color)
    
    def plot(self, ax: Axes, action: np.ndarray, color: Union[str, List[str]] = 'black', is_full: bool = False) -> None:
        '''
        Car: three_trailer model class
        x: center of rear wheel
        y: center of rear wheel
        yaw: yaw of rear wheel
        yawt1: yaw of trailer1
        yawt2: yaw of trailer2
        yawt3: yaw of trailer3
        steer: steer of front wheel
        '''
        # Fank: change the color to list
        _, steer = action
        steer = self.steer_scale(steer)
        
        if isinstance(color, str):
            color = [color] * 4
        
        self.plot_tractor_and_four_wheels(ax, steer, color=color[0], is_full=is_full)
        for i in range(1, 4):
            self.plot_trailer_and_two_wheels(ax, number=i, color=color[i], is_full=is_full)
        # for i in range(1, 4):
        #     self.plot_link(ax, number=i, color=color[i])
        
        self.plot_arrow(ax, l=self.WB * 0.8, color=color[0])
    
    
    def plot_link(self, ax: Axes, number: int, color: str = 'black') -> None:
        x_trailer, y_trailer = self.get_center_trailer(number)
        if number == 1:
            x, y, _, _, _, _ = self.state
            ax.plot(np.array([x, x_trailer]), np.array([y, y_trailer]), color, linewidth=1)
        else:
            x_trailer_front, y_trailer_front = self.get_center_trailer(number - 1)
            ax.plot(np.array([x_trailer_front, x_trailer]), np.array([y_trailer_front, y_trailer]), color, linewidth=1)
    
    def plot_arrow(self, ax: Axes, l: float, color: str = 'black') -> None:
        x, y, yaw, _, _, _ = self.state
    
        angle = np.deg2rad(30)
        d = 0.3 * l
        w = 1

        x_start = x
        y_start = y
        x_end = x + l * np.cos(yaw)
        y_end = y + l * np.sin(yaw)

        theta_hat_L = yaw + np.pi - angle
        theta_hat_R = yaw + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=color, linewidth=w)
        ax.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=color, linewidth=w)
    
    # def plot_tractor_and_four_wheels(self, ax: Axes, steer: float, color: str = 'black') -> None:
    #     # get current state from the class
    #     x, y, yaw, _, _, _ = self.state
    #     # plot initial tractor
    #     tractor = np.array([[-self.RB, -self.RB, self.RF, self.RF, -self.RB],
    #                     [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
    #     wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
    #                     [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
    #     frWheel = wheel.copy()
    #     flWheel = wheel.copy()
    #     rrWheel = wheel.copy()
    #     rlWheel = wheel.copy()
    #     # rotate to yaw
    #     Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
    #                     [np.sin(yaw), np.cos(yaw)]])
    #     tractor = np.dot(Rot1, tractor)
    #     # move to the current position
    #     tractor += np.array([[x], [y]])
    #     # plot tractor
    #     ax.plot(tractor[0, :], tractor[1, :], color, linewidth=1)
    #     Rot2 = np.array([[np.cos(steer), -np.sin(steer)],
    #                     [np.sin(steer), np.cos(steer)]])
    #     frWheel = np.dot(Rot2, frWheel)
    #     flWheel = np.dot(Rot2, flWheel)
    #     frWheel += np.array([[self.WB], [-self.WD / 2]])
    #     flWheel += np.array([[self.WB], [self.WD / 2]])
    #     rrWheel[1, :] -= self.WD / 2
    #     rlWheel[1, :] += self.WD / 2
    #     frWheel = np.dot(Rot1, frWheel)
    #     flWheel = np.dot(Rot1, flWheel)
    #     rrWheel = np.dot(Rot1, rrWheel)
    #     rlWheel = np.dot(Rot1, rlWheel)
    #     frWheel += np.array([[x], [y]])
    #     flWheel += np.array([[x], [y]])
    #     rrWheel += np.array([[x], [y]])
    #     rlWheel += np.array([[x], [y]])
        
    #     # plot tractor 4 wheels
    #     ax.plot(frWheel[0, :], frWheel[1, :], color, linewidth=1)
    #     ax.plot(rrWheel[0, :], rrWheel[1, :], color, linewidth=1)
    #     ax.plot(flWheel[0, :], flWheel[1, :], color, linewidth=1)
    #     ax.plot(rlWheel[0, :], rlWheel[1, :], color, linewidth=1)
        
    def plot_tractor_and_four_wheels(self, ax: Axes, steer: float, color: str = 'black', is_full: bool = False) -> None:
        # get current state from the class
        x, y, yaw, _, _, _ = self.state
        # plot initial tractor
        tractor = np.array([[-self.RB, -self.RB, self.RF, self.RF, -self.RB],
                        [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
        wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
                        [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
        frWheel = wheel.copy()
        flWheel = wheel.copy()
        rrWheel = wheel.copy()
        rlWheel = wheel.copy()
        # rotate to yaw
        Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        tractor = np.dot(Rot1, tractor)
        # move to the current position
        tractor += np.array([[x], [y]])
        # plot tractor
        if is_full:
            ax.fill(tractor[0, :], tractor[1, :], color)
        else:
            ax.plot(tractor[0, :], tractor[1, :], color, linewidth=1)
        Rot2 = np.array([[np.cos(steer), -np.sin(steer)],
                        [np.sin(steer), np.cos(steer)]])
        frWheel = np.dot(Rot2, frWheel)
        flWheel = np.dot(Rot2, flWheel)
        frWheel += np.array([[self.WB], [-self.WD / 2]])
        flWheel += np.array([[self.WB], [self.WD / 2]])
        rrWheel[1, :] -= self.WD / 2
        rlWheel[1, :] += self.WD / 2
        frWheel = np.dot(Rot1, frWheel)
        flWheel = np.dot(Rot1, flWheel)
        rrWheel = np.dot(Rot1, rrWheel)
        rlWheel = np.dot(Rot1, rlWheel)
        frWheel += np.array([[x], [y]])
        flWheel += np.array([[x], [y]])
        rrWheel += np.array([[x], [y]])
        rlWheel += np.array([[x], [y]])
        
        # plot tractor 4 wheels
        if not is_full:
            ax.plot(frWheel[0, :], frWheel[1, :], color, linewidth=1)
            ax.plot(rrWheel[0, :], rrWheel[1, :], color, linewidth=1)
            ax.plot(flWheel[0, :], flWheel[1, :], color, linewidth=1)
            ax.plot(rlWheel[0, :], rlWheel[1, :], color, linewidth=1)
    
    def get_bounding_box_list(self, state):
        """get the bounding box list of the vehicle
        - input: current state actually
        - ouput: list of bounding box of which is (x, y, l, d, yaw) style
        """
        x, y, yaw, yawt1, yawt2, yawt3 = state
        bounding_box_list = []
        x_tractor, y_tractor = self.get_center_tractor(state)
        x_tractor_l = self.RF + self.RB
        x_tractor_w = self.W
        bounding_box_list.append((x_tractor, y_tractor, x_tractor_l, x_tractor_w, yaw))
        x_trailer1, y_trailer1 = self.get_center_trailer(1, state)
        x_trailer1_l = self.RTB - self.RTF
        x_trailer1_w = self.W
        bounding_box_list.append((x_trailer1, y_trailer1, x_trailer1_l, x_trailer1_w, yawt1))
        x_trailer2, y_trailer2 = self.get_center_trailer(2, state)
        x_trailer2_l = self.RTB2 - self.RTF2
        x_trailer2_w = self.W
        bounding_box_list.append((x_trailer2, y_trailer2, x_trailer2_l, x_trailer2_w, yawt2))
        x_trailer3, y_trailer3 = self.get_center_trailer(3, state)
        x_trailer3_l = self.RTB3 - self.RTF3
        x_trailer3_w = self.W
        bounding_box_list.append((x_trailer3, y_trailer3, x_trailer3_l, x_trailer3_w, yawt3))
        return bounding_box_list
    
    def get_center_trailer(self, number: int, state=None) -> Tuple[float, float]:
        """
        get the center of tractor directly from the self.state
        """
        if state is None:
            x, y, yaw, yawt1, yawt2, yawt3 = self.state
        else:
            x, y, yaw, yawt1, yawt2, yawt3 = state
        if number == 1:
            x_trailer = x - self.RTR * np.cos(yawt1)
            y_trailer = y - self.RTR * np.sin(yawt1)
            
        elif number == 2:
            x_trailer = x - self.RTR * np.cos(yawt1)
            y_trailer = y - self.RTR * np.sin(yawt1)
            x_trailer -= self.RTR2 * np.cos(yawt2)
            y_trailer -= self.RTR2 * np.sin(yawt2)
            
        else:
            x_trailer = x - self.RTR * np.cos(yawt1)
            y_trailer = y - self.RTR * np.sin(yawt1)
            x_trailer -= self.RTR2 * np.cos(yawt2)
            y_trailer -= self.RTR2 * np.sin(yawt2)
            x_trailer -= self.RTR3 * np.cos(yawt3)
            y_trailer -= self.RTR3 * np.sin(yawt3) 
            
        return (x_trailer, y_trailer)
    
    def get_center_tractor(self, state=None) -> Tuple[float, float]:
        """get the center position of tractor
        if you input state, then will return the input state center tractor coordinates
        if you don't input state, 
        """
        if state is None:
            x, y, yaw, _, _, _ = self.state
        else:
            x, y, yaw, _, _, _ = state
        x += self.WB / 2 * np.cos(yaw)
        y += self.WB / 2 * np.sin(yaw)
        return (x, y)
        
    # def plot_trailer_and_two_wheels(self, ax: Axes, number: int, color: str = 'black') -> None:
        
    #     x, y, yaw, yawt1, yawt2, yawt3 = self.state
        
    #     trail = np.array([[-self.RTB, -self.RTB, -self.RTF, -self.RTF, -self.RTB],
    #                     [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
    #     trail += np.array([[self.RTR],[0]])
    #     wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
    #                     [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
    #     ltWheel = wheel.copy()
    #     rtWheel = wheel.copy()
        
    #     if number == 1:
    #         Rot = np.array([[np.cos(yawt1), -np.sin(yawt1)],
    #                         [np.sin(yawt1), np.cos(yawt1)]])
    #         trail = np.dot(Rot, trail)
    #         trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
    #         trail += np.array([[x], [y]])
    #         ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
            
    #         ltWheel = np.dot(Rot, ltWheel)
    #         rtWheel = np.dot(Rot, rtWheel)
    #         x_trailer, y_trailer = self.get_center_trailer(number=1)
    #         ltWheel += np.array([[x_trailer],[y_trailer]])
    #         rtWheel += np.array([[x_trailer],[y_trailer]])
    #         ltWheel -= np.array([[self.WD/2 * np.sin(yawt1)],[-self.WD/2 * np.cos(yawt1)]])
    #         rtWheel += np.array([[self.WD/2 * np.sin(yawt1)],[-self.WD/2 * np.cos(yawt1)]])
            
    #         ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
    #         ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
        
    #     elif number == 2:
    #         Rot = np.array([[np.cos(yawt2), -np.sin(yawt2)],
    #                         [np.sin(yawt2), np.cos(yawt2)]])
    #         trail = np.dot(Rot, trail)
    #         trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
    #         trail -= np.array([[self.RTR2 * np.cos(yawt2)],[self.RTR2 * np.sin(yawt2)]])
    #         trail += np.array([[x], [y]])
    #         ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
            
    #         ltWheel = np.dot(Rot, ltWheel)
    #         rtWheel = np.dot(Rot, rtWheel)
    #         x_trailer, y_trailer = self.get_center_trailer(number=2)
    #         ltWheel += np.array([[x_trailer],[y_trailer]])
    #         rtWheel += np.array([[x_trailer],[y_trailer]])
    #         ltWheel -= np.array([[self.WD/2 * np.sin(yawt2)],[-self.WD/2 * np.cos(yawt2)]])
    #         rtWheel += np.array([[self.WD/2 * np.sin(yawt2)],[-self.WD/2 * np.cos(yawt2)]])
    #         ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
    #         ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
            
            
    #     if number == 3:
    #         Rot = np.array([[np.cos(yawt3), -np.sin(yawt3)],
    #                         [np.sin(yawt3), np.cos(yawt3)]])
    #         trail = np.dot(Rot, trail)
    #         trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
    #         trail -= np.array([[self.RTR2 * np.cos(yawt2)],[self.RTR2 * np.sin(yawt2)]])
    #         trail -= np.array([[self.RTR3 * np.cos(yawt3)],[self.RTR3 * np.sin(yawt3)]])
    #         trail += np.array([[x], [y]])
    #         ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
            
    #         ltWheel = np.dot(Rot, ltWheel)
    #         rtWheel = np.dot(Rot, rtWheel)
    #         x_trailer, y_trailer = self.get_center_trailer(number=3)
            
    #         ltWheel += np.array([[x_trailer],[y_trailer]])
    #         rtWheel += np.array([[x_trailer],[y_trailer]])
    #         ltWheel -= np.array([[self.WD/2 * np.sin(yawt3)],[-self.WD/2 * np.cos(yawt3)]])
    #         rtWheel += np.array([[self.WD/2 * np.sin(yawt3)],[-self.WD/2 * np.cos(yawt3)]])
    #         ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
    #         ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
    
    def plot_trailer_and_two_wheels(self, ax: Axes, number: int, color: str = 'black', is_full: bool = False) -> None:
            
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        
        trail = np.array([[-self.RTB, -self.RTB, -self.RTF, -self.RTF, -self.RTB],
                        [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]])
        trail += np.array([[self.RTR],[0]])
        wheel = np.array([[-self.TR, -self.TR, self.TR, self.TR, -self.TR],
                        [self.TW / 4, -self.TW / 4, -self.TW / 4, self.TW / 4, self.TW / 4]])
        ltWheel = wheel.copy()
        rtWheel = wheel.copy()
        
        if number == 1:
            Rot = np.array([[np.cos(yawt1), -np.sin(yawt1)],
                            [np.sin(yawt1), np.cos(yawt1)]])
            trail = np.dot(Rot, trail)
            trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
            trail += np.array([[x], [y]])
            if is_full:
                ax.fill(trail[0, :], trail[1, :], color)
            else:
                ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
                
                ltWheel = np.dot(Rot, ltWheel)
                rtWheel = np.dot(Rot, rtWheel)
                x_trailer, y_trailer = self.get_center_trailer(number=1)
                ltWheel += np.array([[x_trailer],[y_trailer]])
                rtWheel += np.array([[x_trailer],[y_trailer]])
                ltWheel -= np.array([[self.WD/2 * np.sin(yawt1)],[-self.WD/2 * np.cos(yawt1)]])
                rtWheel += np.array([[self.WD/2 * np.sin(yawt1)],[-self.WD/2 * np.cos(yawt1)]])
                
                ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
                ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
        
        elif number == 2:
            Rot = np.array([[np.cos(yawt2), -np.sin(yawt2)],
                            [np.sin(yawt2), np.cos(yawt2)]])
            trail = np.dot(Rot, trail)
            trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
            trail -= np.array([[self.RTR2 * np.cos(yawt2)],[self.RTR2 * np.sin(yawt2)]])
            trail += np.array([[x], [y]])
            if is_full:
                ax.fill(trail[0, :], trail[1, :], color)
            else:
                ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
                
                ltWheel = np.dot(Rot, ltWheel)
                rtWheel = np.dot(Rot, rtWheel)
                x_trailer, y_trailer = self.get_center_trailer(number=2)
                ltWheel += np.array([[x_trailer],[y_trailer]])
                rtWheel += np.array([[x_trailer],[y_trailer]])
                ltWheel -= np.array([[self.WD/2 * np.sin(yawt2)],[-self.WD/2 * np.cos(yawt2)]])
                rtWheel += np.array([[self.WD/2 * np.sin(yawt2)],[-self.WD/2 * np.cos(yawt2)]])
                ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
                ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
            
        elif number == 3:
            Rot = np.array([[np.cos(yawt3), -np.sin(yawt3)],
                            [np.sin(yawt3), np.cos(yawt3)]])
            trail = np.dot(Rot, trail)
            trail -= np.array([[self.RTR * np.cos(yawt1)],[self.RTR * np.sin(yawt1)]])
            trail -= np.array([[self.RTR2 * np.cos(yawt2)],[self.RTR2 * np.sin(yawt2)]])
            trail -= np.array([[self.RTR3 * np.cos(yawt3)],[self.RTR3 * np.sin(yawt3)]])
            trail += np.array([[x], [y]])
            if is_full:
                ax.fill(trail[0, :], trail[1, :], color)
            else:
                ax.plot(trail[0, :], trail[1, :], color, linewidth=1)
                
                ltWheel = np.dot(Rot, ltWheel)
                rtWheel = np.dot(Rot, rtWheel)
                x_trailer, y_trailer = self.get_center_trailer(number=3)
                
                ltWheel += np.array([[x_trailer],[y_trailer]])
                rtWheel += np.array([[x_trailer],[y_trailer]])
                ltWheel -= np.array([[self.WD/2 * np.sin(yawt3)],[-self.WD/2 * np.cos(yawt3)]])
                rtWheel += np.array([[self.WD/2 * np.sin(yawt3)],[-self.WD/2 * np.cos(yawt3)]])
                ax.plot(ltWheel[0, :], ltWheel[1, :], color, linewidth=1)
                ax.plot(rtWheel[0, :], rtWheel[1, :], color, linewidth=1)
    
            
    def is_collision(self, ox: List[float], oy: List[float]) -> bool:
        '''
        check whether there is collision
        Inputs:
        x, y, yaw, yawt1, yawt2, yawt3: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        '''
        points = np.array(list(zip(ox, oy)))
        tree = cKDTree(points)
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        d = self.SAFE_D
        
        # first trailer test collision
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0 + d #half length of trailer1 plus d

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)

        idst1 = tree.query_ball_point([ctx1, cty1], rt1)

        if idst1:
            for i in idst1:
                xot1 = ox[i] - ctx1
                yot1 = oy[i] - cty1

                dx_trail1 = xot1 * np.cos(yawt1) + yot1 * np.sin(yawt1)
                dy_trail1 = -xot1 * np.sin(yawt1) + yot1 * np.cos(yawt1)

                if abs(dx_trail1) <= rt1 and \
                        abs(dy_trail1) <= self.W / 2.0 + d:
                    return True
        # check the second trailer collision
        deltal2 = (self.RTF2 + self.RTB2) / 2.0
        rt2 = (self.RTB2 - self.RTF2) / 2.0 + d
        
        ctx2 = ctx1 - deltal2 * np.cos(yawt2)
        cty2 = cty1 - deltal2 * np.sin(yawt2)
        
        idst2 = tree.query_ball_point([ctx2, cty2], rt2)
        
        if idst2:
            for i in idst2:
                xot2 = ox[i] - ctx2
                yot2 = oy[i] - cty2
                
                dx_trail2 = xot2 * np.cos(yawt2) + yot2 * np.sin(yawt2)
                dy_trail2 = -xot2 * np.cos(yawt2) + yot2 * np.cos(yawt2)
                
                if abs(dx_trail2) <= rt2 and \
                    abs(dy_trail2) <= self.W / 2.0 + d:
                        return True
                    
        # check the third trailer collision
        deltal3 = (self.RTF3 + self.RTB3) / 2.0
        rt3 = (self.RTB3 - self.RTF3) / 2.0 + d
        
        ctx3 = ctx2 - deltal3 * np.cos(yawt3)
        cty3 = cty2 - deltal3 * np.sin(yawt3)
        
        idst3 = tree.query_ball_point([ctx3, cty3], rt3)
        
        if idst3:
            for i in idst3:
                xot3 = ox[i] - ctx3
                yot3 = oy[i] - cty3
                
                dx_trail3 = xot3 * np.cos(yawt3) + yot3 * np.sin(yawt3)
                dy_trail3 = -xot3 * np.cos(yawt3) + yot3 * np.cos(yawt3)
                
                if abs(dx_trail3) <= rt3 and \
                    abs(dy_trail3) <= self.W / 2.0 + d:
                        return True
                    
        # check the tractor collision
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0 + d

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        ids = tree.query_ball_point([cx, cy], rc)

        if ids:
            for i in ids:
                xo = ox[i] - cx
                yo = oy[i] - cy

                dx_car = xo * np.cos(yaw) + yo * np.sin(yaw)
                dy_car = -xo * np.sin(yaw) + yo * np.cos(yaw)

                if abs(dx_car) <= rc and \
                        abs(dy_car) <= self.W / 2.0 + d:
                    return True

        return False
    
    
    def collision_metric(self, ox: List[float], oy: List[float]):
        '''
        give a collision metric for each tractor and trailer
        Inputs:
        x, y, yaw, yawt1, yawt2, yawt3: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        
        tractor_collision_metric: 1 means there are no risk, the lower means there are more risk for colliding
        '''
        tractor_collision_metric = 1
        trailer1_collision_metric = 1
        trailer2_collision_metric = 1
        trailer3_collision_metric = 1
        points = np.array(list(zip(ox, oy)))
        tree = cKDTree(points)
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        d = self.SAFE_METRIC
        
        # first trailer test collision
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0 + d #half length of trailer1 plus d

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)

        idst1 = tree.query_ball_point([ctx1, cty1], rt1)

        if idst1:
            min_metric = rt1
            for i in idst1:
                xot1 = ox[i] - ctx1
                yot1 = oy[i] - cty1

                dx_trail1 = xot1 * np.cos(yawt1) + yot1 * np.sin(yawt1)
                dy_trail1 = -xot1 * np.sin(yawt1) + yot1 * np.cos(yawt1)
                d_trail1 = np.sqrt((dx_trail1)**2 + (dy_trail1)**2)
                min_metric = min(min_metric, d_trail1)

            trailer1_collision_metric = min_metric / rt1
        # check the second trailer collision
        deltal2 = (self.RTF2 + self.RTB2) / 2.0
        rt2 = (self.RTB2 - self.RTF2) / 2.0 + d
        
        ctx2 = ctx1 - deltal2 * np.cos(yawt2)
        cty2 = cty1 - deltal2 * np.sin(yawt2)
        
        idst2 = tree.query_ball_point([ctx2, cty2], rt2)
        
        if idst2:
            min_metric = rt2
            for i in idst2:
                xot2 = ox[i] - ctx2
                yot2 = oy[i] - cty2
                
                dx_trail2 = xot2 * np.cos(yawt2) + yot2 * np.sin(yawt2)
                dy_trail2 = -xot2 * np.cos(yawt2) + yot2 * np.cos(yawt2)
                d_trail2 = np.sqrt((dx_trail2)**2 + (dy_trail2)**2)
                min_metric = min(min_metric, d_trail2)
            trailer2_collision_metric = min_metric / rt2
                    
        # check the third trailer collision
        deltal3 = (self.RTF3 + self.RTB3) / 2.0
        rt3 = (self.RTB3 - self.RTF3) / 2.0 + d
        
        ctx3 = ctx2 - deltal3 * np.cos(yawt3)
        cty3 = cty2 - deltal3 * np.sin(yawt3)
        
        idst3 = tree.query_ball_point([ctx3, cty3], rt3)
        
        if idst3:
            min_metric = rt3
            for i in idst3:
                xot3 = ox[i] - ctx3
                yot3 = oy[i] - cty3
                
                dx_trail3 = xot3 * np.cos(yawt3) + yot3 * np.sin(yawt3)
                dy_trail3 = -xot3 * np.cos(yawt3) + yot3 * np.cos(yawt3)
                d_trail3 = np.sqrt((dx_trail3)**2 + (dy_trail3)**2)
                min_metric = min(min_metric, d_trail3)
            trailer3_collision_metric = min_metric / rt3
                    
        # check the tractor collision
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0 + d

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        ids = tree.query_ball_point([cx, cy], rc)

        if ids:
            min_metric = rc
            for i in ids:
                xo = ox[i] - cx
                yo = oy[i] - cy

                dx_car = xo * np.cos(yaw) + yo * np.sin(yaw)
                dy_car = -xo * np.sin(yaw) + yo * np.cos(yaw)
                d_car = np.sqrt((dx_car)**2 + (dy_car)**2)
                min_metric = min(min_metric, d_car)
            tractor_collision_metric = min_metric / rc
        

        return np.array([tractor_collision_metric, trailer1_collision_metric, trailer2_collision_metric, trailer3_collision_metric], dtype=np.float32)
    
    def one_hot_representation(self, d, number, ox: List[float], oy: List[float]):
        '''
        give a collision metric for each tractor and trailer
        Inputs:
        x, y, yaw, yawt1, yawt2, yawt3: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        
        one-hot-representation: d: the safe distance
        number: the size of number
        '''
        
        points = np.array(list(zip(ox, oy)))
        tree = cKDTree(points)
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        
        # first trailer test collision
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0 + d #half length of trailer1 plus d

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)

        idst1 = tree.query_ball_point([ctx1, cty1], rt1)
        trailer1_one_hot = np.zeros(number)
        if idst1:
            for i in idst1:
                xot1 = ox[i] - ctx1
                yot1 = oy[i] - cty1

                dx_trail1 = xot1 * np.cos(yawt1) + yot1 * np.sin(yawt1)
                dy_trail1 = -xot1 * np.sin(yawt1) + yot1 * np.cos(yawt1)
                new_trailer1_one_hot = shift_one_hot_representation(dx_trail1, dy_trail1, number)
                trailer1_one_hot = one_hot_or(new_trailer1_one_hot, trailer1_one_hot)

            
        # check the second trailer collision
        deltal2 = (self.RTF2 + self.RTB2) / 2.0
        rt2 = (self.RTB2 - self.RTF2) / 2.0 + d
        
        ctx2 = ctx1 - deltal2 * np.cos(yawt2)
        cty2 = cty1 - deltal2 * np.sin(yawt2)
        
        idst2 = tree.query_ball_point([ctx2, cty2], rt2)
        trailer2_one_hot = np.zeros(number)
        if idst2:
            for i in idst2:
                xot2 = ox[i] - ctx2
                yot2 = oy[i] - cty2
                
                dx_trail2 = xot2 * np.cos(yawt2) + yot2 * np.sin(yawt2)
                dy_trail2 = -xot2 * np.cos(yawt2) + yot2 * np.cos(yawt2)
                new_trailer2_one_hot = shift_one_hot_representation(dx_trail2, dy_trail2, number)
                trailer2_one_hot = one_hot_or(new_trailer2_one_hot, trailer2_one_hot)
                    
        # check the third trailer collision
        deltal3 = (self.RTF3 + self.RTB3) / 2.0
        rt3 = (self.RTB3 - self.RTF3) / 2.0 + d
        
        ctx3 = ctx2 - deltal3 * np.cos(yawt3)
        cty3 = cty2 - deltal3 * np.sin(yawt3)
        
        idst3 = tree.query_ball_point([ctx3, cty3], rt3)
        trailer3_one_hot = np.zeros(number)
        if idst3:
            for i in idst3:
                xot3 = ox[i] - ctx3
                yot3 = oy[i] - cty3
                
                dx_trail3 = xot3 * np.cos(yawt3) + yot3 * np.sin(yawt3)
                dy_trail3 = -xot3 * np.cos(yawt3) + yot3 * np.cos(yawt3)
                new_trailer3_one_hot = shift_one_hot_representation(dx_trail3, dy_trail3, number)
                trailer3_one_hot = one_hot_or(new_trailer3_one_hot, trailer3_one_hot)
                    
        # check the tractor collision
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0 + d

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        ids = tree.query_ball_point([cx, cy], rc)
        tractor_one_hot = np.zeros(number)
        if ids:
            for i in ids:
                xo = ox[i] - cx
                yo = oy[i] - cy

                dx_car = xo * np.cos(yaw) + yo * np.sin(yaw)
                dy_car = -xo * np.sin(yaw) + yo * np.cos(yaw)
                new_tractor_one_hot = shift_one_hot_representation(dx_car, dy_car, number)
                tractor_one_hot = one_hot_or(tractor_one_hot, new_tractor_one_hot)
        return np.concatenate([tractor_one_hot, trailer1_one_hot, trailer2_one_hot, trailer3_one_hot], axis=0).astype(np.float32)
    
    def one_hot_representation_enhanced(self, d, number, ox: List[float], oy: List[float]):
        '''
        give a collision metric for each tractor and trailer
        Inputs:
        x, y, yaw, yawt1, yawt2, yawt3: list
        first use kdtree to find obstacle index
        then use a more complicated way to test whether to collide
        
        one-hot-representation: d: the safe distance
        number: the size of number
        '''
        
        points = np.array(list(zip(ox, oy)))
        tree = cKDTree(points)
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        
        # first trailer test collision
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0 + d #half length of trailer1 plus d

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)

        idst1 = tree.query_ball_point([ctx1, cty1], rt1)
        trailer1_one_hot = np.zeros(number)
        min_squared_distance_trailer1 = {i: (float('inf'), None) for i in range(number)}
        if idst1:
            for i in idst1:
                xot1 = ox[i] - ctx1
                yot1 = oy[i] - cty1

                dx_trail1 = xot1 * np.cos(yawt1) + yot1 * np.sin(yawt1)
                dy_trail1 = -xot1 * np.sin(yawt1) + yot1 * np.cos(yawt1)
                new_trailer1_one_hot = shift_one_hot_representation(dx_trail1, dy_trail1, number)
                trailer1_one_hot = one_hot_or(new_trailer1_one_hot, trailer1_one_hot)
                region_number = find_one_position(new_trailer1_one_hot)
                distance = dx_trail1**2 + dy_trail1**2
                if distance < min_squared_distance_trailer1[region_number][0]:
                    min_squared_distance_trailer1[region_number] = (distance, (dx_trail1, dy_trail1))
        for region_number in min_squared_distance_trailer1:
            if min_squared_distance_trailer1[region_number][0] == float('inf'):
                min_squared_distance_trailer1[region_number] = (0, (0, 0))
            else:
                min_squared_distance_trailer1[region_number] = (1, min_squared_distance_trailer1[region_number][1])
        # check the second trailer collision
        deltal2 = (self.RTF2 + self.RTB2) / 2.0
        rt2 = (self.RTB2 - self.RTF2) / 2.0 + d

        ctx2 = ctx1 - deltal2 * np.cos(yawt2)
        cty2 = cty1 - deltal2 * np.sin(yawt2)

        idst2 = tree.query_ball_point([ctx2, cty2], rt2)
        trailer2_one_hot = np.zeros(number)
        min_squared_distance_trailer2 = {i: (float('inf'), None) for i in range(number)}
        if idst2:
            for i in idst2:
                xot2 = ox[i] - ctx2
                yot2 = oy[i] - cty2

                dx_trail2 = xot2 * np.cos(yawt2) + yot2 * np.sin(yawt2)
                dy_trail2 = -xot2 * np.sin(yawt2) + yot2 * np.cos(yawt2)
                new_trailer2_one_hot = shift_one_hot_representation(dx_trail2, dy_trail2, number)
                trailer2_one_hot = one_hot_or(new_trailer2_one_hot, trailer2_one_hot)
                region_number = find_one_position(new_trailer2_one_hot)
                distance = dx_trail2**2 + dy_trail2**2
                if distance < min_squared_distance_trailer2[region_number][0]:
                    min_squared_distance_trailer2[region_number] = (distance, (dx_trail2, dy_trail2))
        for region_number in min_squared_distance_trailer2:
            if min_squared_distance_trailer2[region_number][0] == float('inf'):
                min_squared_distance_trailer2[region_number] = (0, (0, 0))
            else:
                min_squared_distance_trailer2[region_number] = (1, min_squared_distance_trailer2[region_number][1])
        
                    
        # check the third trailer collision
        deltal3 = (self.RTF3 + self.RTB3) / 2.0
        rt3 = (self.RTB3 - self.RTF3) / 2.0 + d

        ctx3 = ctx2 - deltal3 * np.cos(yawt3)
        cty3 = cty2 - deltal3 * np.sin(yawt3)

        idst3 = tree.query_ball_point([ctx3, cty3], rt3)
        trailer3_one_hot = np.zeros(number)
        min_squared_distance_trailer3 = {i: (float('inf'), None) for i in range(number)}
        if idst3:
            for i in idst3:
                xot3 = ox[i] - ctx3
                yot3 = oy[i] - cty3

                dx_trail3 = xot3 * np.cos(yawt3) + yot3 * np.sin(yawt3)
                dy_trail3 = -xot3 * np.sin(yawt3) + yot3 * np.cos(yawt3)
                new_trailer3_one_hot = shift_one_hot_representation(dx_trail3, dy_trail3, number)
                trailer3_one_hot = one_hot_or(new_trailer3_one_hot, trailer3_one_hot)
                region_number = find_one_position(new_trailer3_one_hot)
                distance = dx_trail3**2 + dy_trail3**2
                if distance < min_squared_distance_trailer3[region_number][0]:
                    min_squared_distance_trailer3[region_number] = (distance, (dx_trail3, dy_trail3))
        for region_number in min_squared_distance_trailer3:
            if min_squared_distance_trailer3[region_number][0] == float('inf'):
                min_squared_distance_trailer3[region_number] = (0, (0, 0))
            else:
                min_squared_distance_trailer3[region_number] = (1, min_squared_distance_trailer3[region_number][1])
                    
        # check the tractor collision
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0 + d

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        ids = tree.query_ball_point([cx, cy], rc)
        tractor_one_hot = np.zeros(number)
        min_squared_distance_tractor = {i: (float('inf'), None) for i in range(number)}
        if ids:
            for i in ids:
                xo = ox[i] - cx
                yo = oy[i] - cy

                dx_car = xo * np.cos(yaw) + yo * np.sin(yaw)
                dy_car = -xo * np.sin(yaw) + yo * np.cos(yaw)
                new_tractor_one_hot = shift_one_hot_representation(dx_car, dy_car, number)
                tractor_one_hot = one_hot_or(tractor_one_hot, new_tractor_one_hot)
                region_number = find_one_position(new_tractor_one_hot)
                distance = dx_car**2 + dy_car**2
                if distance < min_squared_distance_tractor[region_number][0]:
                    min_squared_distance_tractor[region_number] = (distance, (dx_car, dy_car))
        for region_number in min_squared_distance_tractor:
            if min_squared_distance_tractor[region_number][0] == float('inf'):
                min_squared_distance_tractor[region_number] = (0, (0, 0))
            else:
                min_squared_distance_tractor[region_number] = (1, min_squared_distance_tractor[region_number][1])
        
        # Create the dictionary as before
        result_dict = {
            "d": np.float32(d),
            "number": np.float32(number),
            "distances": {
                "tractor": {k: (np.float32(v[0]), np.float32(v[1][0]), np.float32(v[1][1])) for k, v in min_squared_distance_tractor.items()},
                "trailer1": {k: (np.float32(v[0]), np.float32(v[1][0]), np.float32(v[1][1])) for k, v in min_squared_distance_trailer1.items()},
                "trailer2": {k: (np.float32(v[0]), np.float32(v[1][0]), np.float32(v[1][1])) for k, v in min_squared_distance_trailer2.items()},
                "trailer3": {k: (np.float32(v[0]), np.float32(v[1][0]), np.float32(v[1][1])) for k, v in min_squared_distance_trailer3.items()}
            }
        }

        # Create a one-dimensional vector of the float values
        result_vector = np.array([result_dict["d"], result_dict["number"]] +
                                [v for sublist in result_dict["distances"]["tractor"].values() for v in sublist] +
                                [v for sublist in result_dict["distances"]["trailer1"].values() for v in sublist] +
                                [v for sublist in result_dict["distances"]["trailer2"].values() for v in sublist] +
                                [v for sublist in result_dict["distances"]["trailer3"].values() for v in sublist])

        return result_vector.astype(np.float32)
    
    def lidar_detection(self, ox: List[float], oy: List[float]):
        """
        use the distance to rectangle
        to give the exact distance of each erea,
        since there are 3trialers, vector is a 32-dim vector astype=np.float32
        """
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        
        # first trailer lidar detection
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0  #half length of trailer1

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)
        
        trailer1_lidar_detection = point_to_rectangle_distance_vectorized(ctx1, cty1, 2 * rt1, self.W, yawt1, ox, oy)
        # trailer1_lidar_detection_v = point_to_rectangle_distance_vectorized(ctx1, cty1, 2 * rt1, self.W, yawt1, ox, oy)
        
        # the second trailer lidar detection
        deltal2 = (self.RTF2 + self.RTB2) / 2.0
        rt2 = (self.RTB2 - self.RTF2) / 2.0
        
        ctx2 = ctx1 - deltal2 * np.cos(yawt2)
        cty2 = cty1 - deltal2 * np.sin(yawt2)
        
        trailer2_lidar_detection = point_to_rectangle_distance_vectorized(ctx2, cty2,  2 * rt2, self.W, yawt2, ox, oy)
        # trailer2_lidar_detection_v = point_to_rectangle_distance_vectorized(ctx2, cty2, 2 * rt2, self.W, yawt2, ox, oy)
                 
        # the third trailer lidar detection
        deltal3 = (self.RTF3 + self.RTB3) / 2.0
        rt3 = (self.RTB3 - self.RTF3) / 2.0
        
        ctx3 = ctx2 - deltal3 * np.cos(yawt3)
        cty3 = cty2 - deltal3 * np.sin(yawt3)
        
        trailer3_lidar_detection = point_to_rectangle_distance_vectorized(ctx3, cty3, 2 * rt3, self.W, yawt3, ox, oy)
        # trailer3_lidar_detection_v = point_to_rectangle_distance_vectorized(ctx3, cty3, 2 * rt3, self.W, yawt3, ox, oy)
                    
        # the tractor lidar detection
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        tractor_lidar_detection = point_to_rectangle_distance_vectorized(cx, cy, 2 * rc, self.W, yaw, ox, oy)
        # tractor_lidar_detection_v = point_to_rectangle_distance_vectorized(cx, cy, 2 * rc, self.W, yaw, ox, oy)
        return np.concatenate([tractor_lidar_detection, trailer1_lidar_detection, trailer2_lidar_detection, trailer3_lidar_detection], axis=0).astype(np.float32)

    def lidar_detection_one_hot(self, d, ox: List[float], oy: List[float]):
        """
        use the distance to rectangle
        to give the exact distance of each erea,
        since there are 3trialers, vector is a 32-dim vector astype=np.float32
        """
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        
        # first trailer lidar detection
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0  #half length of trailer1

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)
        
        trailer1_lidar_detection = lidar_one_hot(ctx1, cty1, 2 * rt1, self.W, yawt1, ox, oy, d)
        
        
        # the second trailer lidar detection
        deltal2 = (self.RTF2 + self.RTB2) / 2.0
        rt2 = (self.RTB2 - self.RTF2) / 2.0
        
        ctx2 = ctx1 - deltal2 * np.cos(yawt2)
        cty2 = cty1 - deltal2 * np.sin(yawt2)
        
        trailer2_lidar_detection = lidar_one_hot(ctx2, cty2,  2 * rt2, self.W, yawt2, ox, oy, d)
        
                 
        # the third trailer lidar detection
        deltal3 = (self.RTF3 + self.RTB3) / 2.0
        rt3 = (self.RTB3 - self.RTF3) / 2.0
        
        ctx3 = ctx2 - deltal3 * np.cos(yawt3)
        cty3 = cty2 - deltal3 * np.sin(yawt3)
        
        trailer3_lidar_detection = lidar_one_hot(ctx3, cty3, 2 * rt3, self.W, yawt3, ox, oy, d)
        
                    
        # the tractor lidar detection
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        tractor_lidar_detection = lidar_one_hot(cx, cy, 2 * rc, self.W, yaw, ox, oy, d)
        return np.concatenate([tractor_lidar_detection, trailer1_lidar_detection, trailer2_lidar_detection, trailer3_lidar_detection], axis=0).astype(np.float32)
    
    def plot_lidar_detection_one_hot(self, d):
        """
        use the distance to rectangle
        to give the exact distance of each erea,
        since there are 3trialers, vector is a 32-dim vector astype=np.float32
        """
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
        
        # first trailer lidar detection
        deltal1 = (self.RTF + self.RTB) / 2.0 #which is exactly C.RTR
        rt1 = (self.RTB - self.RTF) / 2.0  #half length of trailer1

        ctx1 = x - deltal1 * np.cos(yawt1)
        cty1 = y - deltal1 * np.sin(yawt1)
        
        plot_lidar_detection(ctx1, cty1, 2 * rt1, self.W, yawt1, d)
        
        
        # the second trailer lidar detection
        deltal2 = (self.RTF2 + self.RTB2) / 2.0
        rt2 = (self.RTB2 - self.RTF2) / 2.0
        
        ctx2 = ctx1 - deltal2 * np.cos(yawt2)
        cty2 = cty1 - deltal2 * np.sin(yawt2)
        
        plot_lidar_detection(ctx2, cty2,  2 * rt2, self.W, yawt2, d)
        
                 
        # the third trailer lidar detection
        deltal3 = (self.RTF3 + self.RTB3) / 2.0
        rt3 = (self.RTB3 - self.RTF3) / 2.0
        
        ctx3 = ctx2 - deltal3 * np.cos(yawt3)
        cty3 = cty2 - deltal3 * np.sin(yawt3)
        
        plot_lidar_detection(ctx3, cty3, 2 * rt3, self.W, yawt3, d)
        
                    
        # the tractor lidar detection
        deltal = (self.RF - self.RB) / 2.0
        rc = (self.RF + self.RB) / 2.0

        cx = x + deltal * np.cos(yaw)
        cy = y + deltal * np.sin(yaw)

        plot_lidar_detection(cx, cy, 2 * rc, self.W, yaw, d)
    
if __name__ == "__main__":
    parser = get_config()
    args = parser.parse_args()
    singletractor = SingleTractor(args)
    fig, ax = plt.subplots()
    action = np.array([0.2, 0.3], dtype=np.float32)
    ox, oy = settings.map_plain_high_resolution()
    ax.plot(ox, oy, 'sk', linewidth=1)
    ax.set_aspect('equal', adjustable='datalim')
    singletractor.plot(ax, action)
    plt.show()
    singletractor.is_collision(ox, oy)
    print(1)
    
        
        
        