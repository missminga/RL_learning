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
