"""DQN CartPole 实验 API"""

from fastapi import APIRouter

from web.schemas import CartPoleRunRequest, CartPoleRunResponse
from core.dqn import run_cartpole_experiment

router = APIRouter(prefix="/api/cartpole", tags=["cartpole"])


@router.post("/run", response_model=CartPoleRunResponse)
async def run_cartpole(req: CartPoleRunRequest):
    """运行 DQN CartPole 实验"""
    result = run_cartpole_experiment(
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
    )
    return result
