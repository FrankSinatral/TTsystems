import os
import sys
import numpy as np
from tractor_trailer_envs.vehicles.vehicle import Vehicle
import matplotlib.pyplot as plt
from tractor_trailer_envs.vehicles.config import get_config
from scipy.spatial import cKDTree
import tractor_trailer_envs.map_and_obstacles.settings as settings
from matplotlib.axes import Axes
from typing import Tuple, List

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
            steer = self.steer_scale(steer)
            x_ = x + np.cos(yaw) * v * dt
            y_ = y + np.sin(yaw) * v * dt
            yaw_ = self.pi_2_pi(
                yaw + v * dt * np.tan(steer) / self.WB
                )
            self.state = (x_, y_, yaw_)
            
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
        for i in range(1, 4):
            self.plot_trailer_and_two_wheels(ax, number=i, color=color, is_full=is_full)
        for i in range(1, 4):
            self.plot_link(ax, number=i, color=color)
        
        self.plot_arrow(ax, l=self.WB * 0.8, color=color)
    
    
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
    
    def get_center_trailer(self, number: int) -> Tuple[float, float]:
        """
        get the center of tractor directly from the self.state
        """
        x, y, yaw, yawt1, yawt2, yawt3 = self.state
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
    
    def get_center_tractor(self) -> Tuple[float, float]:
        """get the center position of tractor"""
        x, y, yaw, _, _, _ = self.state
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
    
    
    def collision_metric(self, ox: List[float], oy: List[float]) -> bool:
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
    
        
        
        