"""项目主入口：用于本地直接启动 FastAPI 服务。"""

import os

import uvicorn


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)
