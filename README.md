# RL_learning

一个可运行、可测试、可部署的强化学习学习项目：
- **Bandit**（多臂老虎机）
- **GridWorld + Q-Learning**
- **CartPole + DQN**
- **CartPole + REINFORCE**

提供 FastAPI + 前端页面 + 异步训练任务接口。

---

## 1. 主入口与项目结构

### 主入口
- **`main.py`**：项目服务主入口（本地直接运行时使用）
  - 读取 `HOST` / `PORT` 环境变量
  - 启动 `uvicorn`，加载 `web.app:app`
- **`web/app.py`**：Web 应用入口
  - 注册 API 路由
  - 挂载静态页面
  - 暴露 `/healthz` 健康检查
  - 输出结构化请求日志

### 关键目录
- `core/`：算法核心实现
- `web/routers/`：API 路由
- `web/static/`：前端页面
- `tests/`：测试
- `.github/workflows/ci.yml`：CI 流水线
- `docs/HEALTH_BASELINE.md`：项目健康基线

---

## 2. 安装

### 环境要求
- Python >= 3.12
- 推荐使用 [uv](https://docs.astral.sh/uv/)

### 安装依赖
```bash
cd /root/project/RL_learning
uv sync --group dev
```

---

## 3. 启动

### 方式 A：Python 主入口
```bash
uv run python main.py
```

### 方式 B：直接用 uvicorn
```bash
uv run uvicorn web.app:app --host 0.0.0.0 --port 8000
```

启动后访问：
- `http://127.0.0.1:8000/`（Bandit）
- `http://127.0.0.1:8000/gridworld`
- `http://127.0.0.1:8000/cartpole`
- `http://127.0.0.1:8000/policy-gradient`
- `http://127.0.0.1:8000/docs`（OpenAPI）

---

## 4. API 路径说明

### 通用
- `GET /healthz`：健康检查
- `GET /api/tasks/{task_id}`：查询异步任务状态/结果
- `POST /api/tasks/{task_id}/cancel`：取消任务

### Bandit
- `POST /api/bandit/run`：异步提交训练任务
- `POST /api/bandit/run-sync`：同步运行（兼容）

### GridWorld
- `POST /api/gridworld/run`
- `POST /api/gridworld/run-sync`

### DQN CartPole
- `POST /api/cartpole/run`
- `POST /api/cartpole/run-sync`

### REINFORCE
- `POST /api/policy-gradient/run`
- `POST /api/policy-gradient/run-sync`

> 所有训练请求均支持 `seed`（统一随机种子）与 `timeout_seconds`（任务超时保护）。

---

## 5. 页面说明

- `/`：多臂老虎机 ε-greedy 对比
- `/gridworld`：GridWorld Q-Learning 训练与策略可视化
- `/cartpole`：DQN CartPole 训练曲线
- `/policy-gradient`：REINFORCE 训练曲线

---

## 6. 测试分层（本地/CI 命令矩阵）

### Fast（默认开发回归）
```bash
uv run pytest -m "not slow"
```

### Slow（训练集成）
```bash
uv run pytest -m slow
```

### 全量
```bash
uv run pytest
```

### 格式与静态检查
```bash
uv run ruff format --check main.py core web tests
uv run ruff check main.py core web tests
```

### 构建校验
```bash
uv run python -m compileall -q .
```

CI 中默认执行：format + lint + fast tests + build sanity。

---

## 7. 部署建议（Docker / Render）

### Docker
```bash
docker build -t rl-learning .
docker run -p 8000:8000 -e PORT=8000 -e RL_MAX_CONCURRENT_TASKS=2 rl-learning
```

### Render（`render.yaml`）
建议配置环境变量：
- `PORT`：服务端口（Render 注入）
- `HOST=0.0.0.0`
- `RL_MAX_CONCURRENT_TASKS=2`（按实例规格调整）

### 运行保护参数
- 请求参数上限：由 Pydantic schema 强制限制
- 并发限制：`RL_MAX_CONCURRENT_TASKS`
- 超时保护：请求参数 `timeout_seconds`

---

## 8. 常见问题（FAQ）

### Q1: 训练任务很慢怎么办？
- 优先用异步 `/run` 接口提交任务，然后轮询 `/api/tasks/{task_id}`。
- 减少 `episodes` / `n_runs` 以快速验证。

### Q2: 如何保证复现？
- 在请求中传固定 `seed`。
- 项目会统一设置 `random/numpy/torch/gym` 的随机种子策略。

### Q3: 为什么有 `run` 和 `run-sync` 两种接口？
- `run`：生产推荐（异步，支持进度、取消、超时）
- `run-sync`：调试和兼容旧调用方式

### Q4: 前端页面打不开？
- 检查服务是否启动（`/healthz`）
- 检查端口映射与防火墙

### Q5: CI 慢测是否默认执行？
- 默认不执行 slow，仅执行 fast；慢测建议夜间或手动触发。
