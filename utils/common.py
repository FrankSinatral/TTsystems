import numpy as np
def cyclic_angle_distance(angle1, angle2):
    return min(abs(angle1 - angle2), 2 * np.pi - abs(angle1 - angle2))

def mixed_norm(goal, final_state):
    # the input is 6-dim np_array
    # calculate position sum of square
    position_diff_square = np.sum((goal[:2] - final_state[:2]) ** 2)
    
    # calculate angle distance
    angle_diff_square = sum([cyclic_angle_distance(goal[i], final_state[i]) ** 2 for i in range(2, 6)])
    
    # combine the two distances
    total_distance = np.sqrt(position_diff_square + angle_diff_square)
    return total_distance

def action_recover_from_planner(control_list, simulation_freq, v_max, max_steer):
    # this shift is for rl api
    new_control_list = []
    for control in control_list:
        new_control = np.array([control[0] * simulation_freq / v_max, control[1] / max_steer])
        new_control_list.append(new_control)
    
    return new_control_list