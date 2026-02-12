"""FastAPI 应用入口"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.routers import bandit

app = FastAPI(title="RL 学习平台 - 多臂老虎机实验")

# 注册 API 路由
app.include_router(bandit.router)

# 静态文件目录
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    """返回主页"""
    return FileResponse(str(static_dir / "index.html"))
