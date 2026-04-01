import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

#把项目的根目录添加到搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import DynamicsMLP

def train_phase1_sequential():
    #设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"时序训练开始，使用设备: {device}")

    #加载轨迹数据 [N, Seq, Dim]
    data_path = "data/phase1_trajectories.npz"
    if not os.path.exists(data_path):
        print("找不到轨迹数据，请先运行 generate.py")
        return

    data = np.load(data_path)
    X_raw = torch.from_numpy(data['X']).float() # [N, 15, 6]
    Y_raw = torch.from_numpy(data['Y']).float() # [N, 15, 4]

    dataset = TensorDataset(X_raw, Y_raw)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    #初始化模型与优化器
    model = DynamicsMLP(input_dim=6, output_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #自适应学习率变速箱
    #如果连续 5 个 epoch Loss 降不下去，就把学习率砍半，进行精细微调
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)    

    criterion = nn.MSELoss()

    num_epochs = 100 #难度变大，适当增加 epoch 数量
    seq_length = X_raw.shape[1] 
    print(f"检测到轨迹步长 (Sequence Length): {seq_length} 步")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            
            #核心：多步自回归推演
            step_losses = 0
            current_input = batch_X[:, 0, :] 
            
            for t in range(seq_length):
                #模型预测下一步
                pred_next_state = model(current_input)
                
                #计算损失累加
                target_next_state = batch_Y[:, t, :]
                step_losses += criterion(pred_next_state, target_next_state)
                
                #构造下一步的输入：使用预测值 + 下一帧真实的动态受力
                if t < seq_length - 1:
                    next_force = batch_X[:, t+1, 4:] 
                    current_input = torch.cat([pred_next_state, next_force], dim=1)
            
            #穿透时间的梯度反向传播
            total_loss = step_losses / seq_length
            total_loss.backward()
            
            #梯度裁剪，防爆盾
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        #获取当前优化器的实时学习率
        current_lr = optimizer.param_groups[0]['lr'] 
        
        #把学习率也打印出来
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Average Sequential Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.6f}")
        #让变速箱根据当前 Epoch 的平均 Loss 决定是否要降低学习率
        scheduler.step(avg_epoch_loss)

    #保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/phase1_model.pth")
    print("终极时序回归模型训练完成并已保存！")

if __name__ == "__main__":
    train_phase1_sequential()