from abc import ABC, abstractmethod
import numpy as np

class Vehicle(ABC):
    PI = np.pi
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self, x, y, yaw):
        NotImplementedError
    
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def plot(self):
        pass
    
    @abstractmethod
    def is_collision(self, ox, oy):
        pass
    
    @staticmethod
    def pi_2_pi(theta):
        while theta >= Vehicle.PI:
            theta -= 2.0 * Vehicle.PI

        while theta < -Vehicle.PI:
            theta += 2.0 * Vehicle.PI
        
        return theta