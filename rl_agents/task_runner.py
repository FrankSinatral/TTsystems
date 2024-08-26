import torch
import random
import numpy as np
from joblib import Parallel, delayed
from utils import planner
import time
import os
import joblib
import pickle
# from multiprocessing import Pool
# def multi_run_wrapper(args):
#     return planner.find_astar_trajectory(*args)
class TaskRunner:
    def __init__(self, 
                 env_fn, 
                 config: dict = None):
        self.config = config
        self.seed = self.config.get("seed", 0)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.env_config = self.config.get("env_config", {})
        self.env = env_fn(self.env_config)
        # with open("datasets/fixed_obstacles_info.pickle", 'rb') as file:
        #     self.task_list = pickle.load(file)
        #     self.env.unwrapped.update_task_list(self.task_list)
        self.astar_mp_steps = self.config.get("astar_mp_steps", 10)
        self.astar_N_steps = self.config.get("astar_N_steps", 10)
        self.astar_max_iter = self.config.get("astar_max_iter", 5)
        self.astar_heuristic_type = self.config.get("astar_heuristic_type", 'traditional')
        self.astar_batch_size = self.config.get("astar_batch_size", 1000)
        self.astar_total_batch = self.config.get("astar_total_batch", 100)
        self.save_model_path = self.config.get("save_model_path", "datasets/data")
        self.observation_type = self.config.get("observation_type", "lidar_detection_one_hot_triple")
        self.whether_test_fixed_datasets = self.config.get("whether_test_fixed_datasets", False)
        self.finish_episode_number = 0
        self.encounter_task_list = []
        self.feasible_seed_number = 0
        self.planner_config = {
            "plot_final_path": False,
            "plot_rs_path": False,
            "plot_expand_tree": False,
            "mp_step": self.astar_mp_steps, # Important
            "N_steps": self.astar_N_steps, # Important
            "range_steer_set": 20,
            "max_iter": self.astar_max_iter,
            "heuristic_type": self.astar_heuristic_type,
            "save_final_plot": False,
            "controlled_vehicle_config": {
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
                "safe_metric": 3.0, # the safe distance from the vehicle to obstacle
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
            "acceptance_error": 0.5,
        }
        self.check_planner_config = {
            "plot_final_path": False,
            "plot_rs_path": False,
            "plot_expand_tree": False,
            "mp_step": self.astar_mp_steps, # Important
            "N_steps": self.astar_N_steps, # Important
            "range_steer_set": 20,
            "max_iter": self.astar_max_iter,
            "heuristic_type": "traditional",
            "save_final_plot": False,
            "controlled_vehicle_config": {
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
                "safe_metric": 3.0, # the safe distance from the vehicle to obstacle
                "xi_max": (np.pi) / 4, # jack-knife constraint  
            },
            "acceptance_error": 0.5,
        }
        
    def run(self):
        if not self.whether_test_fixed_datasets:
            self.finish_episode_number = 0
            self.feasible_seed_number = 0
            self.encounter_task_list = []
            while self.finish_episode_number // self.astar_batch_size < self.astar_total_batch:
                self.feasible_seed_number += 1
                o, info = self.env.reset(seed=self.seed + self.feasible_seed_number)
                while (not planner.check_is_start_feasible(o["achieved_goal"], info["obstacles_info"], info["map_vertices"], self.check_planner_config)) or (not self.env.unwrapped.check_goal_with_using_lidar_detection_one_hot()):
                    self.feasible_seed_number += 1
                    o, info = self.env.reset(seed=self.seed + self.feasible_seed_number)
                self.encounter_task_list.append((o["achieved_goal"], o["desired_goal"], info["obstacles_info"], info["map_vertices"], info["obstacles_properties"], info["map_properties"]))
                self.finish_episode_number += 1
                if len(self.encounter_task_list) >= self.astar_batch_size:
                    print("Start Collecting Buffer from Astar")
                    start_time = time.time()
                    # with Pool(20) as p:
                    #     astar_results = p.map(multi_run_wrapper, list((task[0], task[1], task[2], task[3], self.planner_config, self.observation_type) for task in self.encounter_task_list))
                    astar_results = Parallel(n_jobs=128)(delayed(planner.find_astar_trajectory)(task[0], task[1], task[2], task[3], self.planner_config, self.observation_type) for task in self.encounter_task_list)
                    # astar_results = [planner.find_astar_trajectory(task[0], task[1], task[2], task[3], self.planner_config, self.observation_type) for task in self.encounter_task_list]
                    self.save_results(astar_results)
                    end_time = time.time()
                    print("Astar collecting time:", end_time - start_time)
        else:
            with open("datasets/10task_list_evaluation_5000.pkl", "rb") as f:
                task_list = pickle.load(f)
            self.encounter_task_list = []
            for j in range(len(task_list)):
                now_task_list = [task_list[j]]
                self.env.unwrapped.update_task_list(now_task_list)
                o, info = self.env.reset()
                self.encounter_task_list.append((o["achieved_goal"], o["desired_goal"], info["obstacles_info"], info["map_vertices"], info["obstacles_properties"], info["map_properties"]))
            print("Start Evaluation")
            astar_results = Parallel(n_jobs=128)(delayed(planner.find_astar_trajectory)(task[0], task[1], task[2], task[3], self.planner_config, self.observation_type) for task in self.encounter_task_list)
            self.save_results(astar_results)  
            print("End Evaluation and saved")
    
    def save_results(self, astar_results):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        
        existing_files = [f for f in os.listdir(self.save_model_path) if f.startswith(f"astar_result_{self.observation_type}_") and f.endswith(".pkl")]
        if existing_files:
            existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            next_index = max(existing_indices) + 1
        else:
            next_index = 0

        save_path = os.path.join(self.save_model_path, f'astar_result_{self.observation_type}_{next_index}.pkl')
        
        data_to_save = {
            "tasks": self.encounter_task_list,
            "results": astar_results
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        # joblib.dump(data_to_save, save_path)
        print(f"Results saved to {save_path}")
        
        # 清空 encounter_task_list 和 astar_results
        self.encounter_task_list = []
        astar_results = []
        
    def load_astar_results(self, data_dir):
        # 找到所有符合条件的文件
        pkl_files = [f for f in os.listdir(data_dir) if f.startswith(f"astar_result_{self.observation_type}") and f.endswith(".pkl")]
        
        if not pkl_files:
            print("No task files found.")
            return [], []
        
        # 按照文件名中的索引排序
        pkl_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        results = []
        tasks = []
        for pkl_file in pkl_files:
            file_path = os.path.join(data_dir, pkl_file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            # data = joblib.load(file_path)
            results.extend(data['results'])
            tasks.extend(data['tasks'])
            print(f"Loaded {pkl_file}:")
            print(f"  Number of tasks: {len(data['tasks'])}")
            print(f"  Number of results: {len(data['results'])}")
        
        return tasks, results
    
    def load_specific_astar_results(self, data_dir, index):
        # 构建符合条件的文件名
        file_name = f"astar_result_{self.observation_type}_{index}.pkl"
        file_path = os.path.join(data_dir, file_name)
        
        # 检查文件是否存在
        if not os.path.isfile(file_path):
            print(f"No file found: {file_name}")
            return [], []
        
        # 加载数据
        data = joblib.load(file_path)
        tasks = data['tasks']
        results = data['results']
        
        print(f"Loaded {file_name}:")
        print(f"  Number of tasks: {len(tasks)}")
        print(f"  Number of results: {len(results)}")
        
        return tasks, results