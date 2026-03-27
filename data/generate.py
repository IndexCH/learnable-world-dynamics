import os
import sys
import random
import numpy as np

# 确保路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.state import RigidBodyState, Force, Vector2
from world.teacher import TeacherWorld

def generate_phase1_trajectories(num_trajectories: int = 5000, seq_length: int = 15, output_dir: str = "data"):
    """
    终极版 Phase 1 轨迹生成器：引入动态变轨和全向连续受力
    """
    teacher = TeacherWorld(dt=1.0)
    X_list = [] 
    Y_list = [] 
    
    print(f"⏳ 开始生成 {num_trajectories} 条长度为 {seq_length} 的高难度物理轨迹...")
    
    for i in range(num_trajectories):
        # 1. 随机初始状态（覆盖较广的速度范围）
        x, y = np.random.uniform(-500.0, 500.0, 2)
        vx, vy = np.random.uniform(-100.0, 100.0, 2)
        state = RigidBodyState(position=Vector2(x, y), velocity=Vector2(vx, vy))
        
        traj_X = []
        traj_Y = []

        # 初始化一个受力
        fx, fy = np.random.uniform(-15.0, 15.0, 2)

        # 2. 连续演化 seq_length 步
        for step in range(seq_length):
            # 【终极防作弊机制】：每 4 步随机改变一次受力方向和大小！
            # 这强迫模型绝对不能依赖惯性，必须时刻读取新的力学输入去进行转弯
            if step % 4 == 0:
                fx, fy = np.random.uniform(-15.0, 15.0, 2)
            
            force = Force(vector=Vector2(fx, fy))

            # 记录当前帧状态 + 受力
            current_x = np.concatenate([state.to_array(1), force.to_array(1)])
            traj_X.append(current_x)
            
            # 物理老师计算下一帧
            next_state = teacher.step(state, force, phase=1)
            traj_Y.append(next_state.to_array(1))
            
            # 更新状态，进入下一帧
            state = next_state
        
        X_list.append(traj_X)
        Y_list.append(traj_Y)

        if (i + 1) % 1000 == 0:
            print(f"已完成 {i + 1}/{num_trajectories}")

    # 3. 转换为 NumPy 矩阵
    X_final = np.array(X_list, dtype=np.float32)
    Y_final = np.array(Y_list, dtype=np.float32)
    
    # 4. 保存
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "phase1_trajectories.npz")
    np.savez(save_path, X=X_final, Y=Y_final)
    
    print(f"\n✅ 终极版轨迹数据生成完毕！")
    print(f"📊 X 形状: {X_final.shape} -> [N, {seq_length}, 6]")
    print(f"📊 Y 形状: {Y_final.shape} -> [N, {seq_length}, 4]")

if __name__ == "__main__":
    generate_phase1_trajectories()