import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import curves_generator
import numpy as np
import math
from scripts.generate_trajectory import action_recover_from_planner, forward_simulation_one_trailer
q0 = [0.0, 0.0, 0.0]
q1 = [10.0, 10.0, np.deg2rad(90.0)]
input = np.array([0, 0, np.deg2rad(0.0), np.deg2rad(0.0)])
goal = np.array([10.0, 10.0, np.deg2rad(90.0), np.deg2rad(90.0)])
maxc = math.tan(0.6) / 3.5
paths = curves_generator.generate_path(q0, q1, maxc)
for path in paths:
    rscontrol_list = []
    for ctype, length in zip(path.ctypes, path.lengths):
        if ctype == 'S':
            steer = 0
        elif ctype == 'WB':
            steer = 0.6
        else:
            steer = -0.6
        step_number = math.floor((np.abs(length / maxc) / (10 * 0.2))) + 1
        
        action_step_size = (length / maxc) / (step_number * 10)
        rscontrol_list += [np.array([action_step_size, steer])] * (step_number * 10)
        
    action_list = action_recover_from_planner(rscontrol_list, 10, 2, 0.6)
    transition_list = forward_simulation_one_trailer(input, goal, action_list, 10)
print(1)