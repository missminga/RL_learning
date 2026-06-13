"""Actor-Critic (A2C) CartPole 实验 API"""

from fastapi import APIRouter

from core.actor_critic import run_a2c_experiment
from web.schemas import A2CRunRequest, A2CRunResponse, TaskSubmitResponse
from web.task_manager import manager

router = APIRouter(prefix="/api/actor-critic", tags=["actor-critic"])


@router.post("/run-sync", response_model=A2CRunResponse)
async def run_a2c_sync(req: A2CRunRequest):
    return run_a2c_experiment(
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        value_coef=req.value_coef,
        entropy_coef=req.entropy_coef,
        max_steps=500,
        n_runs=req.n_runs,
        seed=req.seed,
    )


@router.post("/run", response_model=TaskSubmitResponse)
async def run_a2c(req: A2CRunRequest):
    task = manager.submit(
        "actor-critic",
        run_a2c_experiment,
        timeout_seconds=req.timeout_seconds,
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        value_coef=req.value_coef,
        entropy_coef=req.entropy_coef,
        max_steps=500,
        n_runs=req.n_runs,
        seed=req.seed,
    )
    return {"task_id": task.task_id, "status": task.status, "kind": task.kind}
