import torch
import torch.nn as nn

class DynamicsMLP(nn.Module):
    """
    学习物理动力学的核心神经网络。
    使用带残差连接 (Residual Connection) 的多层感知机。
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super(DynamicsMLP, self).__init__()
        
        effective_input_dim = input_dim - 2
        self.network = nn.Sequential(
            nn.Linear(effective_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # 注意：这里的网络输出的不再是绝对状态，而是变化量 (Delta)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：融合了先验常识的残差结构。
        """
        # x 的维度是 [batch_size, 6]，依次是 [x, y, vx, vy, fx, fy]
        
        # 提取出当前的状态特征 [x, y, vx, vy]（取前 4 列）
        current_state = x[:, :4]
        
        # 从索引 2 开始切，只保留 [vx, vy, fx, fy]
        kinematic_features = x[:, 2:]
        # 让黑盒网络去思考物理规律，算出一个纯粹的变化量 (Delta)
        # 让黑盒只根据速度和力去算变化量
        delta = self.network(kinematic_features)
        
        # 3. 【核心魔法】：跳跃连接 (Skip Connection)
        # 将变化量与原状态相加，作为最终输出。
        # 这一步不是外部硬编码，而是计算图的一部分，可以完美反向传播！
        next_state = current_state + delta
        
        return next_state

if __name__ == "__main__":
    model = DynamicsMLP(input_dim=6, output_dim=4)
    print("✅ 残差网络结构已成功构建!")
    
    dummy_input = torch.tensor([[1000.0, 500.0, 1.0, 0.0, 10.0, 0.0]], dtype=torch.float32)
    dummy_output = model(dummy_input)
    
    # 你会发现，即便在模型未经训练（MLP 输出接近 0）的情况下，
    # 它的初始猜测也会非常接近 1000.0 和 500.0，这就极大降低了学习难度！
    print(f"✅ 输入状态: {dummy_input[:, :4].detach().numpy()}")
    print(f"✅ 未训练时的输出猜测: {dummy_output.detach().numpy()}")