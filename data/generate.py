import os
import sys
import random
import numpy as np

# 【工程黑科技】解决 Python 的模块导入路径问题
# 这一行确保了即便你在 data 文件夹里运行这个脚本，它也能准确找到外面的 world 文件夹
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.state import RigidBodyState, Force, Vector2
from world.teacher import TeacherWorld

def generate_phase1_data(num_samples: int = 10000, output_dir: str = "data"):
    # 1. 请出我们的物理老师，步长设为 1.0
    teacher = TeacherWorld(dt=1.0)
    
    # 用两个大列表来收集所有的 X 和 Y
    X_data = [] 
    Y_data = [] 
    
    # 2. 定义 Phase 1 的题库规则：力的大小固定为 10.0，只有上下左右四个方向
    force_magnitude = 10.0
    directions = [
        Vector2(force_magnitude, 0.0),   # 向右推
        Vector2(-force_magnitude, 0.0),  # 向左推
        Vector2(0.0, force_magnitude),   # 向上推
        Vector2(0.0, -force_magnitude)   # 向下推
    ]
    
    print(f"⏳ 开始生成 {num_samples} 条 Phase 1 训练数据...")
    
    for _ in range(num_samples):
        # 3. 随机生成一个初始状态 (位置在 -100 到 100 之间，初始速度在 -10 到 10 之间)
        x = np.random.uniform(-1000.0, 100.0)
        y = np.random.uniform(-1000.0, 100.0)
        # 【修改这里】：把速度范围扩大，包容我们 Rollout 测试时可能达到的极限速度
        vx = np.random.uniform(-200.0, 200.0) 
        vy = np.random.uniform(-200.0, 200.0)
        
        state = RigidBodyState(position=Vector2(x, y), velocity=Vector2(vx, vy))
        
        # 4. 从四个方向中随机抽取一个力
        f_vec = random.choice(directions)
        force = Force(vector=Vector2(f_vec.x, f_vec.y))
        
        # 5. 【核心步骤】让老师算出下一帧的绝对正确答案
        next_state = teacher.step(state, force, phase=1)
        
        # 6. 数据序列化：把对象转成一维数组
        state_arr = state.to_array(phase=1)    # [x, y, vx, vy]
        force_arr = force.to_array(phase=1)    # [fx, fy]
        
        # 把状态和力拼接成一个长度为 6 的输入向量 X
        x_feature = np.concatenate([state_arr, force_arr])
        
        # 提取下一帧状态作为标签 Y (长度为 4)
        y_label = next_state.to_array(phase=1) 
        
        X_data.append(x_feature)
        Y_data.append(y_label)
        
    # 7. 将列表转换为 NumPy 的多维矩阵，并指定数据类型为 float32 (PyTorch 最喜欢的数据类型)
    X_matrix = np.array(X_data, dtype=np.float32)
    Y_matrix = np.array(Y_data, dtype=np.float32)
    
    # 8. 确保 data 文件夹存在，然后打包保存为 .npz
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "phase1_dataset.npz")
    np.savez(save_path, X=X_matrix, Y=Y_matrix)
    
    print(f"✅ 数据生成完毕！已安全保存至: {save_path}")
    print(f"📊 X (Input) 矩阵形状: {X_matrix.shape}  -> 意思是 {num_samples} 行，每行 6 列 [x, y, vx, vy, fx, fy]")
    print(f"📊 Y (Label) 矩阵形状: {Y_matrix.shape}  -> 意思是 {num_samples} 行，每行 4 列 [x', y', vx', vy']")
    
    # 为了直观，打印第一条数据给你看看长什么样
    print("\n--- 🔍 验卷：来看一眼第一题和老师给的答案 ---")
    print(f"当前状态和受力 (X) : {X_matrix[0]}")
    print(f"老师算的下一帧 (Y) : {Y_matrix[0]}")

if __name__ == "__main__":
    generate_phase1_data()