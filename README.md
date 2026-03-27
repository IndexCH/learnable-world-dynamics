# 🚀 Learnable World Dynamics System

[cite_start]一个基于数据驱动的可学习物理动力学仿真引擎。本项目旨在通过深度学习方法，使 AI 在不硬编码特定物理公式（如 $F=ma$）的前提下，通过观察物体状态演化自行发现并模拟物理规律 [cite: 237, 238]。

## 🎯 项目愿景 (Project Goal)
* [cite_start]**约束学习**：自主发现不可逾越的边界（如碰撞） [cite: 242]。
* [cite_start]**动力学发现**：自行拟合力与运动之间的复杂非线性关系 [cite: 243]。
* [cite_start]**通用物理仿真**：具备模拟“异星物理”的潜力 [cite: 573]。

## 🚀 核心技术演进 (Evolutionary Path)

| 版本 | 技术定义 | 核心改进 | 解决的科学问题 |
| :--- | :--- | :--- | :--- |
| **V1.0** | 纯端到端映射 | [cite_start]使用 vanilla MLP 学习 $S_t \to S_{t+1}$ [cite: 251, 252]。 | [cite_start]动力学基本建模 [cite: 252]。 |
| **V1.1** | 残差与特征解耦 | [cite_start]引入残差架构学习 $\Delta$，剥离绝对坐标 [cite: 302, 303]。 | [cite_start]坐标依赖与平移不变性 [cite: 303]。 |
| **V1.5** | **可微积分器嵌入** | [cite_start]**显式集成辛欧拉积分器 (Symplectic Euler)** [cite: 486, 489]。 | [cite_start]**运动学自相矛盾 (Kinematic Hallucination)** [cite: 490]。 |

## 🧠 核心架构：灰盒模型 (Gray-Box Design)

[cite_start]在最新的 **V1.5** 版本中，我们实现了动力学与运动学的严格解耦 [cite: 487, 490]：

1. [cite_start]**黑盒动力学**：神经网络仅负责预测加速度影响下的速度变化量 $\Delta v$ [cite: 488, 490]。
2. [cite_start]**白盒运动学**：在计算图末端硬编码数学公理 $x_{t+1} = x_t + v_{t+1} \cdot dt$ [cite: 489, 490]。

## 🧪 实验与验证 (Validation)

### 1. 运动学一致性测试 (KIR)
[cite_start]定义 **运动学自相矛盾率 (KIR)** [cite: 565]：
$$KIR = \frac{1}{N} \sum \left\| \Delta x_{pred} - v_{pred} \cdot \Delta t \right\|^2$$
* [cite_start]**V1.5 (有先验)**：KIR 在数学上严格等于 0，确保了时空逻辑的绝对严谨 [cite: 568]。

### 2. 长期推演稳定性 (Long-term Rollout)
[cite_start]实验证明，在 10,000 步的连续推演中，系统依然能保持完美的匀速直线运动，无能量爆炸风险 [cite: 585]。

## 📊 训练表现
[cite_start]在 Phase 1 的 15 步时序回归训练中，多步平均 Loss 从初始的 **77,178** 最终稳定收敛至约 **12.3** [cite: 417, 505]。