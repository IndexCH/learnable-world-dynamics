<div align="right">
  <a href="#english-version">🇺🇸 English</a> | <a href="#中文版">🇨🇳 中文</a>
</div>

<a name="中文版"></a>
# 🚀 Learnable World Dynamics System / 可学习世界动力学系统

一个基于数据驱动的可学习物理动力学仿真引擎。本项目旨在通过深度学习方法，使 AI 在不硬编码特定物理公式（如 $F=ma$）的前提下，通过观察物体状态演化自行发现并模拟物理规律。

## 🎯 项目愿景

* **约束学习**：自主发现不可逾越的边界（如碰撞）。
* **动力学发现**：自行拟合力与运动之间的复杂非线性关系。
* **通用物理仿真**：具备模拟“异星物理”的潜力。

## 🚀 核心技术演进 (Phase 1)

| 版本 | 技术定义 | 核心改进 | 解决的科学问题 |
| :--- | :--- | :--- | :--- |
| **V1.0** | **纯端到端映射** | 使用 MLP 直接学习绝对状态到下一状态的映射。 | 建立了动力学基本建模，但暴露了缺乏平移不变性和分布外失效的缺陷。 |
| **V1.1** | **残差与特征解耦** | 引入残差架构学习状态变化量，显式剥离绝对坐标特征。 | 解决了坐标依赖问题，显著提升了空间维度上的泛化能力。 |
| **V1.2** | **数据流形扩张** | 扩大训练集在速度向量空间上的支撑集范围。 | 缓解了模型作为插值器在动态范围边缘的外推失效与非线性发散。 |
| **V1.3** | **递归自回归训练** | 引入闭环推演与随时间反向传播 (BPTT)，计算跨多步的累积损失。 | 解决了多步推演中的暴露偏差，极大增强了长线预测稳健性。 |
| **V1.4** | **动态扰动与自适应优化** | 引入非稳态随机过程（高频变轨）与二阶优化策略。 | 强制解耦运动惯性与瞬时加速度，实现了复杂时变受力环境下的动力学一致性。 |
| **V1.5** | **可微积分器嵌入** | 在计算图末端显式嵌入无参数的辛欧拉积分器，剥夺网络预测位移的权限。 | 彻底消除了速度与位移断裂的“运动学自相矛盾”，逼近理论极限。 |

## 🧠 核心架构：灰盒模型

在最新的 **V1.5** 版本中，我们实现了动力学与运动学的严格解耦：

1. **黑盒动力学**：神经网络仅负责预测加速度影响下的速度变化量 $\Delta v$。
2. **白盒运动学**：在计算图末端硬编码数学公理 $x_{t+1} = x_t + v_{t+1} \cdot dt$ 作为辛欧拉积分节点。

## 🧪 实验与核心验证

### 1. 运动学一致性测试 (KIR)
为了量化模型是否符合基本时空逻辑，我们定义了 **运动学自相矛盾率 (KIR)**：

$$KIR = \frac{1}{N} \sum \left\| \Delta x_{pred} - v_{pred} \cdot \Delta t \right\|^2$$ 

* **V1.4 (纯黑盒)**：KIR 绝对不等于 0，网络会产生位移与速度断裂的空间幻觉。
* **V1.5 (灰盒模型)**：KIR 在数学上严格等于 0，保证了物理引擎的底层严谨性。

### 2. “异星物理”分布外 (OOD) 泛化测试
为了证明模型没有死记硬背牛顿定律，我们设计了包含非线性高阶法则的数据集 ($a = F^3/m + \sin(v)$)。实验证明，V1.5 架构对**物质的交互法则（动力学）**保持了绝对开放（Zero Prior），是一个真正的通用物理发现器。

### 3. 极限长线推演稳定性
在连续 1000 步以上的无老师纠正自回归推演中，得益于辛欧拉积分“保体积”的几何特性，系统即使长期运行也维持完美的匀速直线运动，拒绝能量爆炸。

---

<br>

<a name="english-version"></a>
# 🚀 Learnable World Dynamics System

A data-driven learnable physics dynamics simulation engine. This project aims to use deep learning to enable AI to autonomously discover and simulate physical laws by observing the evolution of object states, without hardcoding specific dynamic formulas (like $F=ma$).

## 🎯 Project Goals

* **Constraint Learning**: Autonomously discover impassable boundaries (e.g., collisions).
* **Dynamics Discovery**: Fit complex non-linear relationships between force and motion.
* **Generalizable Physics Simulation**: Possess the potential to simulate non-Newtonian "Alien Physics".

## 🚀 Evolutionary Path (Phase 1)

| Version | Technical Definition | Core Improvement | Solved Scientific Problem |
| :--- | :--- | :--- | :--- |
| **V1.0** | **Vanilla E2E Mapping** | Uses MLP to map absolute state directly to the next state. | Established basic dynamic modeling, but exposed translation invariance and OOD failures. |
| **V1.1** | **Residual & Feature Decoupling** | Introduces residual architecture to learn state changes; strips absolute coordinates. | Solved coordinate dependence, significantly improving spatial generalization. |
| **V1.2** | **Support Set Expansion** | Expands the support set range of the training data in the velocity vector space. | Mitigated extrapolation failure and non-linear divergence at the edges of the dynamic range. |
| **V1.3** | **Autoregressive Training** | Introduces closed-loop rollouts and BPTT, calculating cumulative loss over multiple steps. | Addressed exposure bias in multi-step predictions, enhancing long-term robustness. |
| **V1.4** | **Dynamic Disturbance & Adaptive Optim** | Introduces non-stationary random processes and adaptive learning rate scheduling. | Forced the decoupling of motion inertia and instantaneous acceleration, achieving dynamic consistency. |
| **V1.5** | **Differentiable Integrator** | Embeds a parameter-free Symplectic Euler Integrator at the end of the computational graph. | Completely eliminated "Kinematic Hallucinations" (rupture between velocity and displacement). |

## 🧠 Core Architecture: Gray-Box Design

In the latest **V1.5** version, we achieved strict decoupling of dynamics and kinematics:

1. **Black-Box Dynamics**: The neural network is only responsible for predicting velocity changes ($\Delta v$) caused by acceleration.
2. **White-Box Kinematics**: The mathematical axiom $x_{t+1} = x_t + v_{t+1} \cdot dt$ is hardcoded at the end of the computational graph as a Symplectic Euler node.

## 🧪 Experiments & Core Validation

### 1. Kinematic Inconsistency Rate (KIR)
To quantify adherence to spatiotemporal logic, we define the **KIR metric**:

$$KIR = \frac{1}{N} \sum \left\| \Delta x_{pred} - v_{pred} \cdot \Delta t \right\|^2$$ 

* **V1.4 (Pure Black-box)**: KIR $\neq$ 0. The network generates spatial hallucinations where velocity and displacement contradict each other.
* **V1.5 (Gray-box Model)**: KIR is exactly 0 mathematically, ensuring the fundamental rigor of the physics engine.

### 2. "Alien Universe" OOD Generalization Test
To prove the model didn't merely memorize Newton's laws, we tested it on a dataset with high-order non-linear rules ($a = F^3/m + \sin(v)$). Results prove V1.5 maintains an absolute Zero Prior on dynamics, acting as a true "general physics discoverer".

### 3. Extreme Long-Term Rollout Stability
In continuous autoregressive rollouts exceeding 1,000 steps without teacher correction, the system maintains perfect linear motion and avoids energy explosion, thanks to the volume-preserving geometric nature of the Symplectic Euler integrator.