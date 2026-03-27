import torch
import torch.nn as nn

class DynamicsMLP(nn.Module):
    """
    Phase 1 终极完美版：嵌入辛欧拉积分器 (Symplectic Euler Integrator)
    网络只负责学习未知的动力学规律 (加速度)，运动学 (位移) 交给硬编码的物理公式。
    """
    def __init__(self, input_dim: int = 6, output_dim: int = 4, hidden_dim: int = 64):
        super(DynamicsMLP, self).__init__()
        
        # ⚠️ 架构大瘦身：
        # 核心网络只接收 [vx, vy, fx, fy] 4个维度
        # 并且只输出 [dvx, dvy] 2个维度
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 终极改动：只预测速度的变化量 (即加速度的影响)
            nn.Linear(hidden_dim, 2) 
        )
        
        # 物理世界的时间步长 (必须和 TeacherWorld 保持一致)
        self.dt = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：神经网络(动力学) + 硬编码积分器(运动学)
        """
        # 1. 拆解输入 x: [batch_size, 6] -> [pos_x, pos_y, vel_x, vel_y, fx, fy]
        pos = x[:, 0:2]
        vel = x[:, 2:4]
        force = x[:, 4:6]
        
        # 2. 剥离绝对坐标，只把速度和受力喂给神经网络
        kinematic_features = torch.cat([vel, force], dim=1)
        
        # 3. 网络只负责一件事：预测速度的变化量 (Delta Velocity)
        delta_vel = self.network(kinematic_features)
        
        # 4. 【核心物理硬约束】：辛欧拉积分
        # 这部分是绝对精确的数学定义，没有任何参数，不需要学习，也没有误差！
        new_vel = vel + delta_vel
        new_pos = pos + new_vel * self.dt 
        
        # 5. 重新拼装成 [x', y', vx', vy'] 吐出去给外面的时序训练循环
        next_state = torch.cat([new_pos, new_vel], dim=1)
        
        return next_state

if __name__ == "__main__":
    model = DynamicsMLP()
    print("✅ 终极可微积分器网络已构建!")
    
    dummy_input = torch.tensor([[1000.0, 500.0, 1.0, 0.0, 10.0, 0.0]], dtype=torch.float32)
    dummy_output = model(dummy_input)
    
    print(f"✅ 输入状态: {dummy_input.detach().numpy()}")
    print(f"✅ 严格符合运动学的输出: {dummy_output.detach().numpy()}")