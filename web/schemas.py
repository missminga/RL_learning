"""请求/响应的 Pydantic 数据模型"""

from typing import Any

from pydantic import BaseModel, Field


class AsyncControl(BaseModel):
    seed: int | None = Field(
        default=None, ge=0, le=2_147_483_647, description="随机种子"
    )
    timeout_seconds: int = Field(
        default=600, ge=10, le=3600, description="任务超时秒数"
    )


class BanditRunRequest(AsyncControl):
    epsilons: list[float] = Field(default=[0, 0.01, 0.1], min_length=1, max_length=10)
    k: int = Field(default=10, ge=2, le=50)
    steps: int = Field(default=1000, ge=10, le=10000)
    n_runs: int = Field(default=200, ge=1, le=2000)


class BanditSummaryItem(BaseModel):
    epsilon: float
    avg_reward: float
    optimal_pct: float


class BanditRunResponse(BaseModel):
    epsilons: list[float]
    rewards: dict[str, list[float]]
    optimal_pct: dict[str, list[float]]
    summary: list[BanditSummaryItem]
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class GridWorldRunRequest(AsyncControl):
    rows: int = Field(default=5, ge=3, le=10)
    cols: int = Field(default=5, ge=3, le=10)
    traps: list[list[int]] = Field(default=[[1, 3], [3, 1]])
    walls: list[list[int]] = Field(default=[[1, 1], [2, 3]])
    episodes: int = Field(default=500, ge=10, le=5000)
    alpha: float = Field(default=0.1, gt=0, le=1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    epsilon: float = Field(default=1.0, ge=0, le=1)
    epsilon_decay: float = Field(default=0.995, gt=0, le=1)
    epsilon_min: float = Field(default=0.01, ge=0, le=1)
    n_runs: int = Field(default=1, ge=1, le=20)


class GridCellInfo(BaseModel):
    type: str
    action: int
    arrow: str
    q_values: list[float]


class GridWorldSummary(BaseModel):
    final_avg_reward: float
    final_avg_steps: float
    converged_episode: int


class GridWorldRunResponse(BaseModel):
    rows: int
    cols: int
    episodes: int
    avg_rewards: list[float]
    avg_steps: list[float]
    grid: list[list[GridCellInfo]]
    summary: GridWorldSummary
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class CartPoleRunRequest(AsyncControl):
    episodes: int = Field(default=300, ge=10, le=2000)
    hidden_dim: int = Field(default=128, ge=16, le=512)
    lr: float = Field(default=1e-3, gt=0, le=0.1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    epsilon: float = Field(default=1.0, ge=0, le=1)
    epsilon_decay: float = Field(default=0.995, gt=0, le=1)
    epsilon_min: float = Field(default=0.01, ge=0, le=1)
    buffer_capacity: int = Field(default=10000, ge=100, le=100000)
    batch_size: int = Field(default=64, ge=8, le=512)
    target_update_freq: int = Field(default=10, ge=1, le=100)
    n_runs: int = Field(default=1, ge=1, le=5)


class CartPoleSummary(BaseModel):
    final_avg_reward: float
    final_avg_steps: float
    max_reward: float
    solved_episode: int | None


class CartPoleRunResponse(BaseModel):
    episodes: int
    avg_rewards: list[float]
    avg_steps: list[float]
    avg_losses: list[float]
    summary: CartPoleSummary
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ReinforceRunRequest(AsyncControl):
    episodes: int = Field(default=500, ge=10, le=2000)
    hidden_dim: int = Field(default=128, ge=16, le=512)
    lr: float = Field(default=1e-3, gt=0, le=0.1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    n_runs: int = Field(default=1, ge=1, le=5)


class ReinforceSummary(BaseModel):
    final_avg_reward: float
    final_avg_steps: float
    max_reward: float
    solved_episode: int | None


class ReinforceRunResponse(BaseModel):
    episodes: int
    avg_rewards: list[float]
    avg_steps: list[float]
    avg_losses: list[float]
    summary: ReinforceSummary
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str
    kind: str


class TaskStatusResponse(BaseModel):
    task_id: str
    kind: str
    status: str
    progress: float
    message: str
    result: dict[str, Any] | None = None
    error: str | None = None


class TaskCancelResponse(BaseModel):
    task_id: str
    status: str
