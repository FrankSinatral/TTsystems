class Para_single_tractor:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, ox, oy, kdtree):
        '''
        rrt parameters take out resolution
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
        
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree

class Path_single_tractor:
    """
    path parameter
    """
    def __init__(self, x, y, yaw, direction):
        """
        x, y, yaw, yawt1, direction: list
        cost: value
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        yaw = self.yaw + other.yaw
        direction = self.direction + other.direction
        return Path_single_tractor(x, y, yaw, direction)



class Node_single_tractor:
    def __init__(self, x, y, yaw, direction, steer, 
                 xlist, ylist, yawlist, cost, parent=None):
        # notice that here I don't assume the input
        # but one has to turn the yaw to [-np.pi, np.pi)
        self.x = x
        self.y = y 
        self.yaw = yaw
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.direction = direction
        self.steer = steer
        self.cost = cost
        self.parent = parent
        
    def __str__(self):
        return f"A Single Tractor Node ({self.x[-1]},{self.y[-1]},{self.yaw[-1]} with cost {self.cost}" 