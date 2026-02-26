"""FastAPI 应用入口"""

import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web.routers import bandit, cartpole, gridworld, policy_gradient, tasks

app = FastAPI(title="RL 学习平台")
logger = logging.getLogger("rl_learning")
logging.basicConfig(level=logging.INFO)


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    cost_ms = int((time.time() - start) * 1000)
    logger.info(
        json.dumps(
            {
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": cost_ms,
            },
            ensure_ascii=False,
        )
    )
    return response


app.include_router(bandit.router)
app.include_router(gridworld.router)
app.include_router(cartpole.router)
app.include_router(policy_gradient.router)
app.include_router(tasks.router)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/gridworld")
async def gridworld_page():
    return FileResponse(str(static_dir / "gridworld.html"))


@app.get("/cartpole")
async def cartpole_page():
    return FileResponse(str(static_dir / "cartpole.html"))


@app.get("/policy-gradient")
async def policy_gradient_page():
    return FileResponse(str(static_dir / "policy_gradient.html"))
