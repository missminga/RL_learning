"""GridWorld Q-Learning 实验 API 路由"""

from fastapi import APIRouter

from core.q_learning import run_gridworld_experiment
from web.schemas import GridWorldRunRequest, GridWorldRunResponse, TaskSubmitResponse
from web.task_manager import manager

router = APIRouter(prefix="/api/gridworld", tags=["gridworld"])


@router.post("/run-sync", response_model=GridWorldRunResponse)
async def run_gridworld_sync(req: GridWorldRunRequest):
    traps = [tuple(t) for t in req.traps]
    walls = [tuple(w) for w in req.walls]
    return run_gridworld_experiment(
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
        seed=req.seed,
    )


@router.post("/run", response_model=TaskSubmitResponse)
async def run_gridworld(req: GridWorldRunRequest):
    traps = [tuple(t) for t in req.traps]
    walls = [tuple(w) for w in req.walls]
    task = manager.submit(
        "gridworld",
        run_gridworld_experiment,
        timeout_seconds=req.timeout_seconds,
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
        seed=req.seed,
    )
    return {"task_id": task.task_id, "status": task.status, "kind": task.kind}
