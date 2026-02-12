"""多臂老虎机实验 API 路由"""

from fastapi import APIRouter

from core.bandits import run_comparison
from web.schemas import BanditRunRequest, BanditRunResponse

router = APIRouter(prefix="/api/bandit", tags=["bandit"])


@router.post("/run", response_model=BanditRunResponse)
async def run_bandit_experiment(req: BanditRunRequest):
    """运行多臂老虎机对比实验"""
    result = run_comparison(
        epsilons=req.epsilons,
        k=req.k,
        steps=req.steps,
        n_runs=req.n_runs,
    )
    return result
