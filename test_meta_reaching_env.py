import sys
import os
import torch.nn as nn
import gymnasium as gym
import pprint


# import rl_training.tt_system_implement as tt_envs
import tractor_trailer_envs as tt_envs

from tractor_trailer_envs import register_tt_envs
register_tt_envs()
from config import get_config
import numpy as np
import yaml
import pickle
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def save_image_from_array(array, filename):
    # Convert the numpy array to an Image object
    img = Image.fromarray(array.transpose(1, 2, 0))

    # Save the image
    img.save(filename)

def process_and_save_image(rgb_image, filename):
    # Transpose the image
    rgb_image = np.transpose(rgb_image, axes=(1, 2, 0))

    # Convert the list to a numpy array
    rgb_image = np.array(rgb_image, dtype=np.float32)

    # Normalize the image data to the range [-1, 1]
    # rgb_image = rgb_image / 255.0 * 2.0 - 1

    print(rgb_image.shape)

    # Convert the numpy array to an Image object
    img = Image.fromarray((rgb_image * 255).astype(np.uint8))

    # Save the image
    img.save(filename)

    # Plot the image
    plt.imshow(rgb_image)
    plt.savefig("plot.png")

def main():
    
    
    with open("configs/envs/meta_reaching_v0_eval.yaml", 'r') as file:
        config = yaml.safe_load(file)
    # env = tt_envs.TractorTrailerParkingEnv(config)
    env = gym.make("tt-meta-reaching-v0", config=config)
    
    
    # for j in range(10000):
    #     t1 = time.time()
    #     obs, _ = env.reset(seed=(40 + j))
    #     env.action_space.seed(seed=(40 + j))
    #     terminated, truncated = False, False
    #     ep_ret = 0.0
    #     while (not terminated) and (not truncated):
    #         action = env.action_space.sample()
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         # env.unwrapped.render()
    #         ep_ret += reward   
    #     # env.unwrapped.run_simulation()
    #     t2 = time.time()
    #     print("Time: ", t2 - t1)
    for j in range(100):
        start_time = time.time()
        obs, info = env.reset(seed=j + 1)
        # env.unwrapped.real_render()
        # gray_image = env.render_obstacles()
        # rgb_image = env.render_jingyu()
        
        # rgb_image = env.render_obstacles()
        # v_image = env.render_vehicles()
        terminated, truncated = False, False
        while (not terminated) and (not truncated):
            action = env.action_space.sample()
            # env.unwrapped.real_render()
            obs, reward, terminated, truncated, info = env.step(action)
            # rgb_image = env.render_vehicles()
        end_time = time.time()
        print("time:", end_time - start_time)
        
    # env.unwrapped.real_render()
    print(1)
    
    

if __name__ == "__main__":
    main()
    
