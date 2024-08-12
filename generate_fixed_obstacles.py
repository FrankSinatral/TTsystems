import pickle
task_list = []
obstacles_info = [[(20, -10), (20, -20), (-20, -20), (-20, -10)]]

task_dict = {
    "obstacles_info": obstacles_info
}
task_list.append(task_dict)
with open("datasets/fixed_obstacles_info.pickle", 'wb') as file:
    pickle.dump(task_list, file)