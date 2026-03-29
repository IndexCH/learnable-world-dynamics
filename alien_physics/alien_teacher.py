import numpy as np

class AlienTeacher:
    """
    异星物理法则 (Alien Physics Rules):
    这里没有牛顿。加速度不仅和受力的立方成正比，还受到当前速度的正弦扰动。
    法则: a = F^3 + sin(v)
    """
    def __init__(self, dt=1.0):
        self.dt = dt

    def get_next_state(self, state, force):
        """
        根据当前状态和受力，计算下一个状态。
        state: [x, y, vx, vy]
        force: [fx, fy]
        """
        pos = state[:2]
        vel = state[2:4]
        
        # 异星动力学：非线性受力 + 速度相依的奇葩阻尼
        accel = (force ** 3) + np.sin(vel) 
        
        # 运动学公理 (辛欧拉积分)：这部分必须和你的 v1.5 架构保持绝对一致
        new_vel = vel + accel * self.dt
        new_pos = pos + new_vel * self.dt
        
        return np.concatenate([new_pos, new_vel])