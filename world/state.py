from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# ==========================================
# 1. 基础数学类
# ==========================================
@dataclass
class Vector2:
    """二维向量，用于表示位置、速度、力等"""
    x: float = 0.0
    y: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)
    
    def to_list(self) -> list:
        return [self.x, self.y]

# ==========================================
# 2. 外部干预 (Intervention)
# ==========================================
@dataclass
class Force:
    """外部施加的力"""
    # Phase 1+ : 力的向量 (fx, fy)
    vector: Vector2 
    
    # Phase 6+ : 力的作用点 (相对方块中心的局部坐标)
    # 默认为 (0, 0)，即作用在质心，不产生旋转
    application_point: Vector2 = field(default_factory=Vector2)

    def to_array(self, phase: int = 1) -> np.ndarray:
        """根据不同阶段，提取力的特征向量"""
        base = self.vector.to_list()
        if phase >= 6:
            base.extend(self.application_point.to_list())
        return np.array(base, dtype=np.float32)

# ==========================================
# 3. 刚体状态 (The Core State)
# ==========================================
@dataclass
class RigidBodyState:
    """
    单个方块的完整物理状态。
    所有属性都有默认值，方便在 Phase 1 时只关注位置和速度。
    """
    # Phase 1-4 (运动学基础)
    position: Vector2 = field(default_factory=Vector2)
    velocity: Vector2 = field(default_factory=Vector2)
    
    # Box 尺寸 (用于 Phase 4+ 碰撞检测，假设是矩形)
    size_x: float = 1.0
    size_y: float = 1.0

    # Phase 3+ (质量)
    mass: float = 1.0

    # Phase 5+ (环境交互)
    friction: float = 0.0

    # Phase 6+ (旋转动力学)
    theta: float = 0.0      # 朝向角度 (弧度)
    omega: float = 0.0      # 角速度 (弧度/秒)

    def to_array(self, phase: int = 1) -> np.ndarray:
        """
        【关键方法】：将状态扁平化为 1D Numpy 数组，供神经网络使用。
        根据训练阶段(phase)动态包含特征。
        """
        features = self.position.to_list() + self.velocity.to_list()
        
        if phase >= 3:
            features.append(self.mass)
        if phase >= 5:
            features.append(self.friction)
        if phase >= 6:
            features.extend([self.theta, self.omega])
            
        return np.array(features, dtype=np.float32)

# ==========================================
# 4. 环境约束 (Constraints)
# ==========================================
@dataclass
class Wall:
    """
    Phase 4+ 的静态环境约束。
    为了简化碰撞检测，初期定义为无限长的坐标轴对齐的墙（Axis-Aligned）。
    """
    is_vertical: bool   # True 表示这是一面竖直墙 (x = position)，False 表示水平墙 (y = position)
    position: float     # 墙所在的坐标轴位置
    normal_dir: int = 1 # 1 或 -1，表示哪一面是“实体”。比如 x=10 且 dir=-1，表示方块只能在 x < 10 的区域活动。

# ==========================================
# 5. 全局世界状态 (World State)
# ==========================================
@dataclass
class WorldState:
    """
    整个仿真世界的快照。
    Phase 7+ 会包含多个 body。
    """
    bodies: List[RigidBodyState] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)