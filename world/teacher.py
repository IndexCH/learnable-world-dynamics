import copy
from .state import RigidBodyState, Force

class TeacherWorld:
    """
    大自然法则 / The Ground-Truth Physics Engine
    负责根据当前状态和外力，严格按照牛顿力学计算下一帧的绝对正确状态。
    """
    def __init__(self, dt: float = 1.0):
        #1 tick = 1 time step
        self.dt = dt

    def step_phase1(self, state: RigidBodyState, force: Force) -> RigidBodyState:
        """
        Phase 1：
        只处理受力平移。忽略旋转、环境约束、摩擦力。质量默认为 1.0。
        """
        next_state = copy.deepcopy(state)
        
        #提取质量，防范除以 0 的情况（Phase 1 默认质量是 1.0）
        mass = next_state.mass if next_state.mass > 0 else 1.0
        
        #计算加速度 a = F / m
        ax = force.vector.x / mass
        ay = force.vector.y / mass
        
        #更新速度 v' = v + a * dt
        next_state.velocity.x += ax * self.dt
        next_state.velocity.y += ay * self.dt
        
        #更新位置 p' = p + v' * dt
        # 这里使用的是“半隐式欧拉积分 (Semi-implicit Euler)”。
        # 即：用【更新后】的新速度来计算新位置。它比用旧速度计算位置更符合真实物理，误差更小。
        next_state.position.x += next_state.velocity.x * self.dt
        next_state.position.y += next_state.velocity.y * self.dt
        
        return next_state

    def step(self, state: RigidBodyState, force: Force, phase: int = 1) -> RigidBodyState:
        """
        通用步进接口。
        随着项目的推进，可以在这里挂载 Phase 2, Phase 3 的复杂物理逻辑。
        """
        if phase == 1:
            return self.step_phase1(state, force)
        elif phase >= 2:
            raise NotImplementedError(f"Phase {phase} physics not implemented yet. Keep building!")
        else:
            raise ValueError("Phase must be >= 1")