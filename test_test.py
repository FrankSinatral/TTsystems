import numpy as np

# 定义新的函数 shift_one_hot_representation
def shift_one_hot_representation(x, y, n):
    # 使用arctan2来计算角度，该函数会返回[-π, π]之间的角度
    angle = np.arctan2(y, x)
    
    # 如果角度为负，将其转换为[0, 2π)范围内的角度
    if angle < 0:
        angle += 2 * np.pi
        
    # 将[0, 2π)范围分成n等分，每份的角度范围
    angle_increment = (2 * np.pi) / n
    
    # 计算角度所在的范围
    index = int(angle // angle_increment)
    
    # 创建一个n维的零向量
    one_hot_vector = np.zeros(n)
    
    # 将对应范围的索引设置为1
    one_hot_vector[index] = 1
    
    return one_hot_vector

# 编写一个测试脚本来验证函数的行为
def test_shift_one_hot_representation():
    # 定义测试点集
    test_points = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]
    # 定义分割数
    n = 36
    
    # 对每个测试点应用函数并打印结果
    for x, y in test_points:
        vector = shift_one_hot_representation(x, y, n)
        print(f"Point ({x}, {y}) results in one-hot vector: {vector}")

# 运行测试脚本
test_shift_one_hot_representation()