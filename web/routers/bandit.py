"""多臂老虎机实验 API 路由"""

from fastapi import APIRouter

from core.bandits import run_comparison
from web.schemas import BanditRunRequest, BanditRunResponse, TaskSubmitResponse
from web.task_manager import manager

router = APIRouter(prefix="/api/bandit", tags=["bandit"])


@router.post("/run-sync", response_model=BanditRunResponse)
async def run_bandit_experiment_sync(req: BanditRunRequest):
    return run_comparison(
        epsilons=req.epsilons,
        k=req.k,
        steps=req.steps,
        n_runs=req.n_runs,
        seed=req.seed,
    )


@router.post("/run", response_model=TaskSubmitResponse)
async def run_bandit_experiment(req: BanditRunRequest):
    task = manager.submit(
        "bandit",
        run_comparison,
        timeout_seconds=req.timeout_seconds,
        epsilons=req.epsilons,
        k=req.k,
        steps=req.steps,
        n_runs=req.n_runs,
        seed=req.seed,
    )
    return {"task_id": task.task_id, "status": task.status, "kind": task.kind}
