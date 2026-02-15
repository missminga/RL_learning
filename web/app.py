"""FastAPI 应用入口"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.routers import bandit, gridworld, cartpole

app = FastAPI(title="RL 学习平台")

# 注册 API 路由
app.include_router(bandit.router)
app.include_router(gridworld.router)
app.include_router(cartpole.router)

# 静态文件目录
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    """返回主页（多臂老虎机）"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/gridworld")
async def gridworld_page():
    """GridWorld Q-Learning 页面"""
    return FileResponse(str(static_dir / "gridworld.html"))



@app.get("/cartpole")
async def cartpole_page():
    """DQN CartPole 页面"""
    return FileResponse(str(static_dir / "cartpole.html"))
