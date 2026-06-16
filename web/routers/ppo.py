"""PPO (近端策略优化) CartPole 实验 API"""

from fastapi import APIRouter

from core.ppo import run_ppo_experiment
from web.schemas import PPORunRequest, PPORunResponse, TaskSubmitResponse
from web.task_manager import manager

router = APIRouter(prefix="/api/ppo", tags=["ppo"])


@router.post("/run-sync", response_model=PPORunResponse)
async def run_ppo_sync(req: PPORunRequest):
    return run_ppo_experiment(
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        gae_lambda=req.gae_lambda,
        clip_eps=req.clip_eps,
        value_coef=req.value_coef,
        entropy_coef=req.entropy_coef,
        update_epochs=req.update_epochs,
        minibatch_size=req.minibatch_size,
        max_steps=500,
        n_runs=req.n_runs,
        seed=req.seed,
    )


@router.post("/run", response_model=TaskSubmitResponse)
async def run_ppo(req: PPORunRequest):
    task = manager.submit(
        "ppo",
        run_ppo_experiment,
        timeout_seconds=req.timeout_seconds,
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        gae_lambda=req.gae_lambda,
        clip_eps=req.clip_eps,
        value_coef=req.value_coef,
        entropy_coef=req.entropy_coef,
        update_epochs=req.update_epochs,
        minibatch_size=req.minibatch_size,
        max_steps=500,
        n_runs=req.n_runs,
        seed=req.seed,
    )
    return {"task_id": task.task_id, "status": task.status, "kind": task.kind}
