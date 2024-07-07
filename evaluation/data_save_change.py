import os
import joblib
import pickle

def process_directory(input_dir, output_dir):
    # 创建输出目录如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pkl'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_dir)
                output_file_path = os.path.join(output_dir, relative_path)

                # 确保输出文件的目录存在
                output_file_dir = os.path.dirname(output_file_path)
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)

                # 使用 joblib 加载文件
                try:
                    data = joblib.load(input_file_path)
                    print(f"Data loaded successfully from joblib file: {input_file_path}")

                    # 使用 pickle 保存文件
                    with open(output_file_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Data saved successfully to pickle file: {output_file_path}")

                except Exception as e:
                    print(f"An error occurred while processing {input_file_path}: {e}")

if __name__ == "__main__":
    input_dir = 'datasets/astar_result_obstacle_4'
    output_dir = 'datasets/astar_result_obstacle_4_pickle'
    process_directory(input_dir, output_dir)