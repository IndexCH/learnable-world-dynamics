import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 将项目根目录添加到搜索路径，确保能导入 models 和 world
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import DynamicsMLP

def train_alien_universe():
    # 1. 设备与环境配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🛸 异星物理时序训练启动 | 使用算力: {device}")

    # 2. 加载并动态组装异星轨迹数据
    data_path = "data/alien_dataset.npz"
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件 {data_path}，请先运行 python -m data.generate_alien")
        return

    print("📦 正在解析异星数据并重构时序流形...")
    data = np.load(data_path)
    states = data['states'] # [N, seq_len+1, 4]
    forces = data['forces'] # [N, seq_len, 2]
    
    # 【核心修复】：动态拼装自回归特征 X 和监督标签 Y
    # X: 每一帧包含 [当前状态(4维), 当前受力(2维)] -> 共 6 维
    X_np = np.concatenate([states[:, :-1, :], forces], axis=-1)
    # Y: 对应的监督信号是 [下一步的真实状态(4维)]
    Y_np = states[:, 1:, :]
    
    X_raw = torch.from_numpy(X_np).float() # [N, seq_len, 6]
    Y_raw = torch.from_numpy(Y_np).float() # [N, seq_len, 4]

    dataset = TensorDataset(X_raw, Y_raw)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True) # 稍微调大 batch_size 增加梯度稳定性

    # 3. 初始化模型与优化器
    model = DynamicsMLP(input_dim=6, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002) # 初始学习率稍微给大一点
    
    # 变速箱：如果连续 8 个 epoch Loss 不降，学习率减半
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)    

    criterion = nn.MSELoss()
    num_epochs = 150 # 异星规律难学，增加训练轮次
    seq_length = X_raw.shape[1] 
    print(f"📏 轨迹步长: {seq_length} 步 | 样本总数: {X_raw.shape[0]}")

    # 4. 开启 BPTT (Backpropagation Through Time) 训练
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            
            step_losses = 0
            # 提取序列的第一帧状态作为推演起点
            current_state = batch_X[:, 0, :4] 
            
            for t in range(seq_length):
                # 提取当前时刻的真实受力
                current_force = batch_X[:, t, 4:6]
                
                # 拼接 [状态, 受力] 喂给黑盒
                model_input = torch.cat([current_state, current_force], dim=-1)
                
                # 模型直接输出完整的下一帧状态 (内部已包含辛欧拉积分)
                pred_next_state = model(model_input)
                target_next_state = batch_Y[:, t, :]
                
                # 累计本步的预测误差
                step_losses += criterion(pred_next_state, target_next_state)
                
                # 【自回归自闭环】：用"自己预测的状态"去迎接下一步的受力！
                current_state = pred_next_state 
            
            # 平均化序列 Loss 并反向传播
            total_loss = step_losses / seq_length
            total_loss.backward()
            
            # 物理防爆盾：防止 F^3 导致梯度飞天
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr'] 
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{num_epochs} | Seq Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.6f}")
        
        scheduler.step(avg_epoch_loss)

    # 5. 保存胜利果实
    os.makedirs("models", exist_ok=True)
    save_path = "models/alien_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 异星动力学模型已成功训练并保存至: {save_path}")

if __name__ == "__main__":
    train_alien_universe()