"""请求/响应的 Pydantic 数据模型"""

from pydantic import BaseModel, Field


class BanditRunRequest(BaseModel):
    """多臂老虎机实验请求参数"""

    epsilons: list[float] = Field(
        default=[0, 0.01, 0.1],
        min_length=1,
        max_length=10,
        description="要对比的 ε 值列表",
    )
    k: int = Field(default=10, ge=2, le=50, description="老虎机数量")
    steps: int = Field(default=1000, ge=10, le=10000, description="每次实验步数")
    n_runs: int = Field(default=200, ge=1, le=2000, description="重复实验次数")


class BanditSummaryItem(BaseModel):
    """单个 ε 值的摘要统计"""

    epsilon: float
    avg_reward: float
    optimal_pct: float


class BanditRunResponse(BaseModel):
    """多臂老虎机实验响应"""

    epsilons: list[float]
    rewards: dict[str, list[float]]
    optimal_pct: dict[str, list[float]]
    summary: list[BanditSummaryItem]


# ===== GridWorld Q-Learning =====


class GridWorldRunRequest(BaseModel):
    """GridWorld Q-Learning 实验请求参数"""

    rows: int = Field(default=5, ge=3, le=10, description="网格行数")
    cols: int = Field(default=5, ge=3, le=10, description="网格列数")
    traps: list[list[int]] = Field(
        default=[[1, 3], [3, 1]],
        description="陷阱坐标列表 [[row, col], ...]",
    )
    walls: list[list[int]] = Field(
        default=[[1, 1], [2, 3]],
        description="墙壁坐标列表 [[row, col], ...]",
    )
    episodes: int = Field(default=500, ge=10, le=5000, description="训练回合数")
    alpha: float = Field(default=0.1, gt=0, le=1, description="学习率")
    gamma: float = Field(default=0.99, ge=0, le=1, description="折扣因子")
    epsilon: float = Field(default=1.0, ge=0, le=1, description="初始探索率")
    epsilon_decay: float = Field(default=0.995, gt=0, le=1, description="ε 衰减率")
    epsilon_min: float = Field(default=0.01, ge=0, le=1, description="最小 ε")
    n_runs: int = Field(default=1, ge=1, le=20, description="重复训练次数")


class GridCellInfo(BaseModel):
    """单个网格单元的信息"""

    type: str
    action: int
    arrow: str
    q_values: list[float]


class GridWorldSummary(BaseModel):
    """GridWorld 实验摘要"""

    final_avg_reward: float
    final_avg_steps: float
    converged_episode: int


class GridWorldRunResponse(BaseModel):
    """GridWorld Q-Learning 实验响应"""

    rows: int
    cols: int
    episodes: int
    avg_rewards: list[float]
    avg_steps: list[float]
    grid: list[list[GridCellInfo]]
    summary: GridWorldSummary
