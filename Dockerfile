FROM python:3.12-slim

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 复制项目代码
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
