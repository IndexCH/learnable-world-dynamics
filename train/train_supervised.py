import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # 进度条神器

# 确保脚本能找到上一级目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import DynamicsMLP

def train_phase1():
    # ==========================================
    # 1. 超参数设置 (Hyperparameters) - 炼丹的火候
    # ==========================================
    BATCH_SIZE = 64        # 每次看 64 道题就总结一次经验（更新一次权重）
    EPOCHS = 50            # 把这 10000 道题反反复复刷 50 遍
    LEARNING_RATE = 0.001  # 学习率：每次改错时，步伐迈多大

    # 自动检测是否有显卡 (NVIDIA GPU -> 'cuda', 否则用 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 当前使用的计算设备: {device}")

    # ==========================================
    # 2. 加载数据 (Data Loading)
    # ==========================================
    data_path = os.path.join("data", "phase1_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件 {data_path}，请先运行 generate.py")
        
    data = np.load(data_path)
    # 将 Numpy 数组转换为 PyTorch 认识的 Tensor，并搬运到 GPU/CPU 上
    X_tensor = torch.tensor(data['X'], dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(data['Y'], dtype=torch.float32).to(device)
    
    # 使用 TensorDataset 和 DataLoader 将数据打包成批次 (Batches)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ==========================================
    # 3. 初始化模型、损失函数和优化器
    # ==========================================
    # Phase 1: 输入 6 维 [x, y, vx, vy, fx, fy]，输出 4 维 [x', y', vx', vy']
    model = DynamicsMLP(input_dim=6, output_dim=4, hidden_dim=64).to(device)
    
    # 损失函数：均方误差 MSE (Mean Squared Error) —— 专门用来做数值预测
    criterion = nn.MSELoss()
    
    # 优化器：Adam (目前最主流的自适应优化算法，负责帮你调整模型里的参数)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ==========================================
    # 4. 正式开始训练循环 (Training Loop)
    # ==========================================
    print(f"\n🔥 开始训练 Phase 1 模型...")
    
    for epoch in range(EPOCHS):
        model.train() # 将模型设置为训练模式
        epoch_loss = 0.0
        
        # tqdm 包装 dataloader，生成一个漂亮的进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch_X, batch_Y in progress_bar:
            # 步骤 A：清空上一次的梯度（避免前一次的经验干扰这一次）
            optimizer.zero_grad()
            
            # 步骤 B：前向传播 (Forward Pass) -> 让模型做题
            predictions = model(batch_X)
            
            # 步骤 C：计算误差 (Compute Loss) -> 老师用红笔批改
            loss = criterion(predictions, batch_Y)
            
            # 步骤 D：反向传播 (Backward Pass) -> 找出是哪个脑细胞算错了
            loss.backward()
            
            # 步骤 E：更新权重 (Update Weights) -> 纠正脑细胞
            optimizer.step()
            
            epoch_loss += loss.item()
            # 实时更新进度条上的 Loss 显示
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # 每一个 Epoch 结束后，打印一下平均误差
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.6f}")
        
    # ==========================================
    # 5. 保存训练好的模型
    # ==========================================
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "phase1_model.pth")
    # 只保存模型的参数权重（state_dict），这是 PyTorch 的官方推荐做法
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 训练完成！模型权重已保存至: {save_path}")

if __name__ == "__main__":
    train_phase1()