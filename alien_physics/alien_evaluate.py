import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mlp import DynamicsMLP 
from alien_physics.alien_teacher import AlienTeacher

def evaluate_alien_universe(model_path="models/alien_model.pth", steps=50):
    # 1. 加载模型
    model = DynamicsMLP(input_dim=4, output_dim=2) 
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"✅ 成功加载物理模型: {model_path}")
    except FileNotFoundError:
        print(f"❌ 找不到模型文件 {model_path}，请先训练模型！")
        return

    teacher = AlienTeacher()
    
    # 2. 初始状态与受力序列设计
    start_state = np.array([0.0, 0.0, 10.0, 10.0]) # 初始坐标(0,0)，初速度(10,10)
    forces = [np.random.uniform(-2, 2, 2) for _ in range(steps)]
    
    # 3. 轨迹生成容器
    true_states = [start_state]
    pred_states = [start_state]
    kir_errors = []
    
    curr_true_state = start_state.copy()
    curr_pred_state = torch.tensor(start_state, dtype=torch.float32).unsqueeze(0) 
    
    print("🚀 开始异星轨迹推演...")
    for t in range(steps):
        force = forces[t]
        force_tensor = torch.tensor(force, dtype=torch.float32).unsqueeze(0) 
        
        # --- Teacher 真实推演 ---
        curr_true_state = teacher.get_next_state(curr_true_state, force)
        true_states.append(curr_true_state)
        
        # --- AI 预测推演 ---
        with torch.no_grad():
            model_input = torch.cat([curr_pred_state, force_tensor], dim=-1)
            
            # 【终极修复】你的模型直接输出了完整的下一帧状态 (4维)！
            next_state = model(model_input)
            
            # 提取预测出的位置和速度，用于计算 KIR
            pred_pos_next = next_state[:, :2]
            pred_vel_next = next_state[:, 2:4]
            
            # 计算 KIR: 验证模型内部的硬约束是否生效
            kir = torch.norm(pred_pos_next - (curr_pred_state[:, :2] + pred_vel_next * 1.0)).item()
            kir_errors.append(kir)
            
            # 更新状态
            curr_pred_state = next_state
            pred_states.append(curr_pred_state.squeeze().numpy())

    # 4. 数据整理与可视化
    true_states = np.array(true_states)
    pred_states = np.array(pred_states)
    
    print("-" * 30)
    print(f"🌟 最终评估报告 🌟")
    print(f"平均运动学自相矛盾率 (KIR): {np.mean(kir_errors):.10f}")
    if np.mean(kir_errors) < 1e-6:
        print("💡 结论: 完美！你的架构在异星物理下依然保持了绝对的数学严谨性！")
    else:
        print("⚠️ 结论: 存在运动学幻觉，请检查计算图中的积分器约束！")
    print("-" * 30)

    # 画图
    plt.figure(figsize=(10, 8))
    plt.plot(true_states[:, 0], true_states[:, 1], 'b^-', label='Alien Teacher (Ground Truth)', alpha=0.6)
    plt.plot(pred_states[:, 0], pred_states[:, 1], 'ro--', label='V1.5 AI Prediction', alpha=0.6)
    plt.scatter(true_states[0, 0], true_states[0, 1], c='green', s=100, label='Start Point', zorder=5)
    
    plt.title(r"Alien Universe OOD Generalization Test: $a = F^3 + \sin(v)$")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("figures/alien_rollout.png")
    print("📈 轨迹对比图已保存至 figures/alien_rollout.png")
    plt.show()

if __name__ == "__main__":
    evaluate_alien_universe()