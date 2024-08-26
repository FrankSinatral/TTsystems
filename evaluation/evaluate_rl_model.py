# The script is for testing rl model
# to see the effects of the trained model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import tractor_trailer_envs as tt_envs
import gymnasium as gym
import rl_agents as agents
import yaml
import numpy as np
from datetime import datetime
from tractor_trailer_envs import register_tt_envs
register_tt_envs()
def gym_tt_planning_env_fn(config: dict):
    import gymnasium as gym
    import tractor_trailer_envs as tt_envs
    if not hasattr(gym_tt_planning_env_fn, "envs_registered"):
        tt_envs.register_tt_envs()
        gym_tt_planning_env_fn.envs_registered = True
    return gym.make("tt-planning-v0", config=config)

def get_current_time_format():
    # get current time
    current_time = datetime.now()
    # demo: 20230130_153042
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_time

def process_obstacles_properties_to_array(input_list):
    """process for mlp with obstacles properties"""
    array_length = 40
    result_array = np.zeros(array_length, dtype=np.float32)
    
    # 将input_list中的元素顺次填入result_array中
    for i, (x, y, l, d) in enumerate(input_list):
        if i >= 10:
            break
        result_array[i*4:i*4+4] = [x, y, l, d]
    
    return result_array

def main():    
    # config_filename = "configs/agents/eval/rl1_obs_attention.yaml"
    # modelpath = "datasets/models/obs_attention_5.pth"
    
    # config_filename = "configs/agents/eval/rl0.yaml"
    # modelpath = "datasets/models/original_model.pth"
    
    # config_filename = "configs/agents/eval/rl1_lidar_detection_one_hot_triple.yaml"
    # modelpath = "datasets/models/lidar_detection_one_hot_triple_5.pth"
    
    config_filename = "configs/agents/eval/rl1_obs_mlp.yaml"
    modelpath = "runs_rl/planning-v0_rl1_obs_mlp_three_trailer_10_20240815_232935/model_749999.pth"
    
    config_filename = "configs/agents/eval/rl1_obs_mlp.yaml"
    modelpath = "runs_rl/planning-v0_rl1_obs_mlp_three_trailer_10_20240815_232935/model_3549999.pth"
    # modelpath = "runs_rl/planning-v0_rl1_obs_mlp_three_trailer_10_20240815_234855/model_799999.pth"
    with open(config_filename, 'r') as file:
        config_algo = yaml.safe_load(file)
    observation_type = config_algo['env_config']['observation']
    whether_attention = config_algo['env_config']['with_obstacles_info']
    agent = agents.SAC_ASTAR_META_NEW(env_fn=gym_tt_planning_env_fn,
        config=config_algo)
    agent.load(modelpath, whether_load_buffer=False)
    config = config_algo["env_config"]
    # with open("configs/envs/tt_planning_v0_eval.yaml", 'r') as file:
    #     config = yaml.safe_load(file)
    env = gym.make("tt-planning-v0", config=config)
    if config_algo.get("whether_fixed_obstacles", False):
        task_list = [{
                    "goal": np.array([-3, -30, 0, 0, 0, 0], dtype=np.float32),
                    # "goal": np.array([3.05, -33.13, -0.82 , -0.82, -0.82, -0.82], dtype=np.float32),
                    "obstacles_info": config_algo.get("obstacles_info"),
                }]
        env.unwrapped.update_task_list(task_list)
    if config_algo.get("whether_fixed_goal", False):
        task_list = [
            {
                "goal": np.array([-30, 30, np.pi/6, np.pi/6, np.pi/6, np.pi/6], dtype=np.float32)
            }
        ]
        env.unwrapped.update_task_list(task_list)
    failed_cases = []
    failed_number = 0
    for i in range(100):
        o, info = env.reset(seed=i)
        env.unwrapped.real_render()
        terminated, truncated, ep_ret, ep_len = False, False, 0, 0
        while not(terminated or truncated):
            obs_list = [o['observation'], o['achieved_goal'], o['desired_goal']]
            if not observation_type.startswith("original"):
                obs_list.append(o[observation_type])
            if observation_type == "original_with_obstacles_info":
                obs_list.append(process_obstacles_properties_to_array(info['obstacles_properties']))
            action_input = np.concatenate(obs_list)
            if whether_attention:
                action = agent.get_action(action_input, info, deterministic=True)
            else:
                action = agent.get_action(action_input, deterministic=True)
            o, r, terminated, truncated, info = env.step(action)
            env.unwrapped.real_render()
            ep_ret += r
            ep_len += 1
        # env.unwrapped.run_simulation()
        # env.unwrapped.save_result()
        print(f"Episode {i}: Success - {info['is_success']}")
        if not info['is_success']:
            failed_cases.append((o, info))
            failed_number += 1
    print("failed number: ", failed_number)
    #     env.unwrapped.save_result()
    # Save all failed cases to a single file
    # with open('datasets/all_failed_cases_rl0.pkl', 'wb') as f:
    #     pickle.dump(failed_cases, f)
    
    
        
        
if __name__ == "__main__":
    main()
    