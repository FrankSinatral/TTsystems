import pickle

def main():
    # 加载任务列表
    with open("datasets/task_list_way_point_new1.pkl", "rb") as f:
        task_list = pickle.load(f)
    
    # 验证每个任务的 'waypoint' 键是否包含三个元素
    valid_task_count = 0
    invalid_task_indices = []

    for i, task in enumerate(task_list):
        waypoints = task.get("way_point", [])
        if len(waypoints) == 3:
            valid_task_count += 1
        else:
            invalid_task_indices.append(i)
            print(f"Task {i} is invalid with {len(waypoints)} waypoints.")

    print(f"Number of valid tasks: {valid_task_count}")
    print(f"Number of invalid tasks: {len(invalid_task_indices)}")

if __name__ == "__main__":
    main()
