import numpy as np
import os
from alien_physics.alien_teacher import AlienTeacher

def generate_alien_data(num_trajectories=2000, seq_len=30, dt=1.0):
    teacher = AlienTeacher(dt=dt)
    all_states = []
    all_forces = []
    
    print("🛸 开始生成异星宇宙数据集...")
    
    for _ in range(num_trajectories):
        # 随机初始状态: 坐标 [-1000, 1000], 速度 [-50, 50]
        pos = np.random.uniform(-1000, 1000, 2)
        vel = np.random.uniform(-50, 50, 2)
        state = np.concatenate([pos, vel])
        
        traj_states = [state]
        traj_forces = []
        
        # 为了应对强非线性，力的大小稍微控制一下，防止 F^3 直接数值爆炸
        force = np.random.uniform(-3, 3, 2) 
        
        for t in range(seq_len):
            # 每 4 步变轨一次 (流形扩张策略保持不变)
            if t > 0 and t % 4 == 0:
                force = np.random.uniform(-3, 3, 2)
            
            # 向异星老师请教下一步
            next_state = teacher.get_next_state(state, force)
            
            traj_forces.append(force)
            traj_states.append(next_state)
            state = next_state
            
        all_states.append(traj_states)
        all_forces.append(traj_forces)
        
    # 确保 data 文件夹存在
    os.makedirs("data", exist_ok=True)
    
    # 保存为 alien_dataset.npz
    np.savez("data/alien_dataset.npz", 
             states=np.array(all_states), 
             forces=np.array(all_forces))
    print(f"✅ 成功生成 {num_trajectories} 条异星轨迹！数据保存在 data/alien_dataset.npz")

if __name__ == "__main__":
    # 在命令行运行: python -m data.generate_alien
    generate_alien_data()