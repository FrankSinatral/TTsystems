import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import tractor_trailer_envs as tt_envs
def random_generate_state(seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    x_coordinates = 0
    y_coordinates = 0
    yaw_state = 0
    xi_max = ((np.pi) / 4) * (1/2)
    delta_1 = np.random.uniform(-xi_max, xi_max)
    # delta_1 = xi_max
    yawt1_state = yaw_state + delta_1
    
    delta_2 = np.random.uniform(-xi_max, xi_max)
    # delta_2 = xi_max
    yawt2_state = yawt1_state + delta_2
    
    delta_3 = np.random.uniform(-xi_max, xi_max)
    # delta_3 = xi_max
    yawt3_state = yawt2_state + delta_3
    
    return x_coordinates, y_coordinates, yaw_state, yawt1_state, yawt2_state, yawt3_state

def test_to_equil(input, control_list, simulation_freq):
    config_dict = {
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
        "safe_metric": 3.0, #[m] the safe distance from the vehicle to obstacle
        "xi_max": (np.pi) / 4, # jack-knife constraint  
    }
    controlled_vehicle = tt_envs.ThreeTrailer(config_dict)
    controlled_vehicle.reset(*input)
    state_list = [np.array(controlled_vehicle.state).astype(np.float32)]
    x_list = [controlled_vehicle.state[0]]
    y_list = [controlled_vehicle.state[1]]
    yaw_list = [controlled_vehicle.state[2]]
    yawt1_list = [controlled_vehicle.state[3]]
    yawt2_list = [controlled_vehicle.state[4]]
    yawt3_list = [controlled_vehicle.state[5]]
    
    for action_clipped in control_list:
        controlled_vehicle.step(action_clipped, 1 / simulation_freq)
        state_list.append(np.array(controlled_vehicle.state).astype(np.float32))
        x_list.append(controlled_vehicle.state[0])
        y_list.append(controlled_vehicle.state[1])
        yaw_list.append(controlled_vehicle.state[2])
        yawt1_list.append(controlled_vehicle.state[3])
        yawt2_list.append(controlled_vehicle.state[4])
        yawt3_list.append(controlled_vehicle.state[5])
        if controlled_vehicle._is_jack_knife():
            return False
    final_state = np.array(controlled_vehicle.state)
    yawt1  = final_state[3]
    yawt2 = final_state[4]
    yawt3 = final_state[5]
    # plt.figure(figsize=(15, 10))
    
    # plt.subplot(4, 1, 1)
    # plt.plot(x_list, label='x', color='orange')
    # plt.xlabel('Time Steps')
    # plt.ylabel('X Coordinates')
    # plt.title('X Coordinates vs Time Steps')
    # plt.legend()
    # plt.grid(True)
    
    # plt.subplot(4, 1, 2)
    # plt.plot(yawt1_list, label='yawt1', color='blue')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Yawt1 Angle')
    # plt.title('Yawt1 Angle vs Time Steps')
    # plt.legend()
    # plt.grid(True)
    
    # plt.subplot(4, 1, 3)
    # plt.plot(yawt2_list, label='yawt2', color='green')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Yawt2 Angle')
    # plt.title('Yawt2 Angle vs Time Steps')
    # plt.legend()
    # plt.grid(True)
    
    # plt.subplot(4, 1, 4)
    # plt.plot(yawt3_list, label='yawt3', color='red')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Yawt3 Angle')
    # plt.title('Yawt3 Angle vs Time Steps')
    # plt.legend()
    # plt.grid(True)
    
    # # 保存图像
    # plt.tight_layout()
    # plt.savefig('yaw_angles_vs_time_steps.png')
    if np.abs(yawt1) < 1e-3 and np.abs(yawt2) < 1e-3 and np.abs(yawt3) < 1e-3:
        return True
    return False

def main():
    count = 0
    
    # input = np.array(random_generate_state()).astype(np.float32)
    # number_1 = 26
    # number_2 = 60
    # control_list = []
    # for i in range(number_2):
    #     control_list += [np.array([1, 0])] * number_1 + [np.array([-1, 0])] * number_1
    # is_success = test_to_equil(input, control_list, 10)
    # print(is_success)
    for j in range(1000, 5000):
        input = np.array(random_generate_state(seed=j)).astype(np.float32)
        number_1 = 26
        number_2 = 60
        control_list = []
        for i in range(number_2):
            control_list += [np.array([1, 0])] * number_1 + [np.array([-1, 0])] * number_1
        is_success = test_to_equil(input, control_list, 10)
        # print(is_success)
        if is_success:
            count += 1
        print(is_success)
    print(count)
    
if __name__ == "__main__":
    main()
