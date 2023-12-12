import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def is_convex_quadrilateral(vertices: List[Tuple[float, float]]) -> bool:
    if len(vertices) != 4:
        return False

    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    def vector(p1, p2):
        return p2[0] - p1[0], p2[1] - p1[1]

    # calculate cross product
    cross_products = []
    for i in range(4):
        v1 = vector(vertices[i], vertices[(i + 1) % 4])
        v2 = vector(vertices[(i + 1) % 4], vertices[(i + 2) % 4])
        cross = cross_product(v1, v2)
        cross_products.append(cross)

    return all(c > 0 for c in cross_products) or all(c < 0 for c in cross_products)

def remove_duplicates(ox, oy):
    """
    Removes duplicate points from the lists ox and oy.

    Args:
    - ox (list): List of x-coordinates.
    - oy (list): List of y-coordinates.

    Returns:
    - tuple: Two lists without duplicate points.
    """
    
    # Combine the two lists into a list of tuples
    points = list(zip(ox, oy))
    
    # Convert the list of tuples into a set to remove duplicates
    unique_points = list(set(points))
    
    # Unzip the unique points back into two separate lists
    new_ox, new_oy = zip(*unique_points)
    
    return list(new_ox), list(new_oy)

class MapBound:
    def __init__(self, vertices: List[Tuple[float, float]]):
        #TODO: may need to add logic to 
        self.vertices = self.sort_vertices(vertices)
        
    def sort_vertices(self, vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        
        center = np.mean(vertices, axis=0)
        def angle_from_center(vertex):
            return np.arctan2(vertex[1] - center[1], vertex[0] - center[0])

        return sorted(vertices, key=angle_from_center)
        
    def sample_surface(self, sample_density: float) -> Tuple[List[float], List[float]]:
        ox, oy = [], []
        for i in range(4):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % 4]
            dist = np.hypot(end[0] - start[0], end[1] - start[1])
            samples = int(dist / sample_density)
            for s in range(samples + 1):
                t = s / samples
                x = start[0] * (1 - t) + end[0] * t
                y = start[1] * (1 - t) + end[1] * t
                ox.append(x)
                oy.append(y)
        return ox, oy
    
    def plot(self, ax, sample_density: float, markersize: float = 1):
        ox, oy = self.sample_surface(sample_density)
        ax.plot(ox, oy, 'sk', markersize=markersize)
        
# TODO: may need to include dynamic model       
class QuadrilateralObstacle:
    def __init__(self, vertices: List[Tuple[float, float]]):
        #TODO: may need to add logic to 
        self.vertices = self.sort_vertices(vertices)
    
    def cal_center(self):
        return np.mean(self.vertices, axis=0)
    
    def sort_vertices(self, vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        
        center = np.mean(vertices, axis=0)
        def angle_from_center(vertex):
            return np.arctan2(vertex[1] - center[1], vertex[0] - center[0])

        return sorted(vertices, key=angle_from_center)
        
    def sample_surface(self, sample_density: float) -> Tuple[List[float], List[float]]:
        ox, oy = [], []
        for i in range(4):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % 4]
            dist = np.hypot(end[0] - start[0], end[1] - start[1])
            samples = int(dist / sample_density)
            for s in range(samples + 1):
                t = s / samples
                x = start[0] * (1 - t) + end[0] * t
                y = start[1] * (1 - t) + end[1] * t
                ox.append(x)
                oy.append(y)
        return ox, oy
    
    def plot(self, ax, sample_density: float, markersize: float = 1):
        ox, oy = self.sample_surface(sample_density)
        ax.plot(ox, oy, 'sk', markersize=markersize)
        # plt.show()
        
    def step(self, action: np.ndarray, dt: float):
        v, theta = action
        self.vertices = [(x + v * dt * np.cos(theta), y + v * dt * np.sin(theta)) for x, y in self.vertices]
        
class EllipticalObstacle:
    def __init__(self, cx: float, cy: float, a: float, b: float):
        self.cx = cx
        self.cy = cy
        self.a = a
        self.b = b

    def sample_surface(self, sample_density: float) -> Tuple[List[float], List[float]]:
        circumference = np.pi * (3 * (self.a + self.b) - np.sqrt((3 * self.a + self.b) * (self.a + 3 * self.b)))
        num_samples = int(circumference / sample_density)
        theta = np.linspace(0, 2 * np.pi, num_samples)

        ox = self.cx + self.a * np.cos(theta)
        oy = self.cy + self.b * np.sin(theta)

        return ox.tolist(), oy.tolist()

    def plot(self, sample_density: float, markersize: float = 1):
        ox, oy = self.sample_surface(sample_density)
        plt.plot(ox, oy, 'sk', markersize=markersize)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.show()
        
    def step(self, action: np.ndarray, dt: float):
        v, theta = action
        self.cx += v * dt * np.cos(theta)
        self.cy += v * dt * np.sin(theta)