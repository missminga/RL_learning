"""REINFORCE Policy Gradient CartPole 实验 API"""

from fastapi import APIRouter

from web.schemas import ReinforceRunRequest, ReinforceRunResponse
from core.policy_gradient import run_reinforce_experiment

router = APIRouter(prefix="/api/policy-gradient", tags=["policy-gradient"])


@router.post("/run", response_model=ReinforceRunResponse)
async def run_reinforce(req: ReinforceRunRequest):
    """运行 REINFORCE CartPole 实验"""
    result = run_reinforce_experiment(
        episodes=req.episodes,
        hidden_dim=req.hidden_dim,
        lr=req.lr,
        gamma=req.gamma,
        max_steps=500,
        n_runs=req.n_runs,
    )
    return result
