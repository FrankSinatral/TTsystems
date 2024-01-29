import numpy as np

def matrix_multiply():
    # 生成两个大矩阵
    A = np.random.rand(2000, 2000)
    B = np.random.rand(2000, 2000)
    # 执行矩阵乘法
    return A @ B

# import time

# # 定义任务次数
# num_tasks = 1000

# # 顺序执行并计时
# start_time = time.time()
# results = [matrix_multiply() for _ in range(num_tasks)]
# end_time = time.time()

# print("顺序处理时间:", end_time - start_time)


from joblib import Parallel, delayed
import time

# 定义任务次数
num_tasks = 100000

# 并行执行并计时
start_time = time.time()
results = Parallel(n_jobs=100)(delayed(matrix_multiply)() for _ in range(num_tasks))
end_time = time.time()

print("并行处理时间:", end_time - start_time)