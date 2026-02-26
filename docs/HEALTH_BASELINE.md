# Project Health Baseline

## Scope
- API service startup
- Core algorithm unit tests (Bandit / GridWorld / DQN / REINFORCE)
- Async task submit/query/cancel workflow
- Static pages route availability

## Current Baseline (2026-02-26)
- Python: 3.12+
- Dependency manager: uv
- CI checks:
  - `ruff format --check main.py core web tests`
  - `ruff check main.py core web tests`
  - `pytest -m "not slow"`
  - `python -m compileall -q .`

## Test Layers
- Fast: deterministic/unit + API smoke tests
- Slow: training integration (`@pytest.mark.slow`)

## Runtime Protection Defaults
- Max concurrent async tasks: `RL_MAX_CONCURRENT_TASKS=2`
- Per-task timeout: request param `timeout_seconds` (10~3600)
- Request parameter upper bounds: enforced by Pydantic schema

## SLO (suggested)
- `/healthz` p95 < 100ms
- Fast test suite < 2 min
- API error rate < 1% in normal workloads
