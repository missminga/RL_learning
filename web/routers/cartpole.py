"""DQN CartPole 实验 API"""

from fastapi import APIRouter

from core.dqn import run_cartpole_experiment
from web.schemas import CartPoleRunRequest, CartPoleRunResponse, TaskSubmitResponse
from web.task_manager import manager

router = APIRouter(prefix="/api/cartpole", tags=["cartpole"])


@router.post("/run-sync", response_model=CartPoleRunResponse)
async def run_cartpole_sync(req: CartPoleRunRequest):
    return run_cartpole_experiment(
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        epsilon=req.epsilon,
        epsilon_decay=req.epsilon_decay,
        epsilon_min=req.epsilon_min,
        buffer_capacity=req.buffer_capacity,
        batch_size=req.batch_size,
        target_update_freq=req.target_update_freq,
        max_steps=500,
        n_runs=req.n_runs,
        seed=req.seed,
    )


@router.post("/run", response_model=TaskSubmitResponse)
async def run_cartpole(req: CartPoleRunRequest):
    task = manager.submit(
        "cartpole",
        run_cartpole_experiment,
        timeout_seconds=req.timeout_seconds,
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        epsilon=req.epsilon,
        epsilon_decay=req.epsilon_decay,
        epsilon_min=req.epsilon_min,
        buffer_capacity=req.buffer_capacity,
        batch_size=req.batch_size,
        target_update_freq=req.target_update_freq,
        max_steps=500,
        n_runs=req.n_runs,
        seed=req.seed,
    )
    return {"task_id": task.task_id, "status": task.status, "kind": task.kind}
