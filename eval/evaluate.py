import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 确保脚本能找到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import DynamicsMLP
from world.state import RigidBodyState, Force, Vector2
from world.teacher import TeacherWorld

def run_single_rollout(model, teacher, device, steps=40):
    """运行一次完整的轨迹预测对比"""
    # 随机初始状态 (在训练分布内随机)
    initial_state = RigidBodyState(
        position=Vector2(np.random.uniform(-50, 50), np.random.uniform(-50, 50)),
        velocity=Vector2(np.random.uniform(-20, 20), np.random.uniform(-20, 20))
    )
    
    # 随机生成一个变化的受力序列 [steps, 2]
    # 模拟“乱开”的情况：每 10 步换一个随机推力
    forces = []
    current_f = Vector2(np.random.uniform(-15, 15), np.random.uniform(-15, 15))
    for i in range(steps):
        if i % 10 == 0:
            current_f = Vector2(np.random.uniform(-15, 15), np.random.uniform(-15, 15))
        forces.append(Force(vector=current_f))

    true_traj = [initial_state.position.to_list()]
    pred_traj = [initial_state.position.to_list()]
    
    curr_true = initial_state
    curr_pred = initial_state
    
    model.eval()
    with torch.no_grad():
        for i in range(steps):
            f = forces[i]
            # 老师计算
            curr_true = teacher.step(curr_true, f, phase=1)
            true_traj.append(curr_true.position.to_list())
            
            # AI 计算
            state_arr = curr_pred.to_array(phase=1)
            force_arr = f.to_array(phase=1)
            x_input = torch.tensor([np.concatenate([state_arr, force_arr])], dtype=torch.float32).to(device)
            
            y_pred = model(x_input).cpu().numpy()[0]
            curr_pred = RigidBodyState(
                position=Vector2(y_pred[0], y_pred[1]),
                velocity=Vector2(y_pred[2], y_pred[3])
            )
            pred_traj.append(curr_pred.position.to_list())
            
    return np.array(true_traj), np.array(pred_traj)

def multi_scenario_evaluate(num_tests=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicsMLP(input_dim=6, output_dim=4).to(device)
    model.load_state_dict(torch.load("models/phase1_model.pth", map_location=device))
    teacher = TeacherWorld(dt=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    print(f"🧪 正在进行 {num_tests} 组随机压力测试...")

    for i in range(num_tests):
        true_np, pred_np = run_single_rollout(model, teacher, device)
        
        ax = axes[i]
        ax.plot(true_np[:, 0], true_np[:, 1], 'b-o', label='True (Teacher)', alpha=0.4)
        ax.plot(pred_np[:, 0], pred_np[:, 1], 'r--x', label='AI Prediction')
        
        # 标记起点
        ax.scatter(true_np[0, 0], true_np[0, 1], c='green', s=100, label='Start', zorder=5)
        
        ax.set_title(f"Test Scenario {i+1}")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Phase 1: Multi-Scenario Stress Test", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "eval/phase1_multi_test.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 随机测试完成！结果已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    multi_scenario_evaluate()