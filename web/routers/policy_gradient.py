"""REINFORCE Policy Gradient CartPole 实验 API"""

from fastapi import APIRouter

from core.policy_gradient import run_reinforce_experiment
from web.schemas import ReinforceRunRequest, ReinforceRunResponse, TaskSubmitResponse
from web.task_manager import manager

router = APIRouter(prefix="/api/policy-gradient", tags=["policy-gradient"])


@router.post("/run-sync", response_model=ReinforceRunResponse)
async def run_reinforce_sync(req: ReinforceRunRequest):
    return run_reinforce_experiment(
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        max_steps=500,
        n_runs=req.n_runs,
        seed=req.seed,
    )


@router.post("/run", response_model=TaskSubmitResponse)
async def run_reinforce(req: ReinforceRunRequest):
    task = manager.submit(
        "policy-gradient",
        run_reinforce_experiment,
        timeout_seconds=req.timeout_seconds,
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        max_steps=500,
        n_runs=req.n_runs,
        seed=req.seed,
    )
    return {"task_id": task.task_id, "status": task.status, "kind": task.kind}
