"""GridWorld Q-Learning 实验 API 路由"""

from fastapi import APIRouter

from core.q_learning import run_gridworld_experiment
from web.schemas import GridWorldRunRequest, GridWorldRunResponse

router = APIRouter(prefix="/api/gridworld", tags=["gridworld"])


@router.post("/run", response_model=GridWorldRunResponse)
async def run_gridworld(req: GridWorldRunRequest):
    """运行 GridWorld Q-Learning 实验"""
    # 把 [[r,c], ...] 转成 [(r,c), ...] 元组列表
    traps = [tuple(t) for t in req.traps]
    walls = [tuple(w) for w in req.walls]

    result = run_gridworld_experiment(
        rows=req.rows,
        cols=req.cols,
        traps=traps,
        walls=walls,
        episodes=req.episodes,
        alpha=req.alpha,
        gamma=req.gamma,
        epsilon=req.epsilon,
        epsilon_decay=req.epsilon_decay,
        epsilon_min=req.epsilon_min,
        n_runs=req.n_runs,
    )
    return result
