# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a reinforcement learning (RL) study repository for beginners. The goal is to learn RL concepts through progressively complex examples, starting from the simplest cases.

Language: Python. Target audience: RL beginners (小白).

## Commands

```bash
# Install dependencies
uv sync

# Run a specific example
uv run python examples/<example_name>.py

# Run tests
uv run pytest tests/
uv run pytest tests/test_<name>.py -v   # single test file

# Add a new dependency
uv add <package>

# Start Web app (development)
uv run uvicorn web.app:app --reload --port 8000
```

## Architecture

Examples are organized by progressive difficulty in `examples/`:
- Each example is a self-contained Python script that can be run independently
- Examples include inline comments in Chinese explaining RL concepts
- Shared utilities (plotting, environment wrappers) go in `utils/`
- Core RL algorithms live in `core/` and are shared by CLI examples and the Web app
- Web application (FastAPI + ECharts frontend) lives in `web/`

## Key Dependencies

- `numpy` - numerical computation
- `gymnasium` (OpenAI Gym successor) - RL environments
- `matplotlib` - visualization of training results
- `fastapi` + `uvicorn` - Web API server

## Conventions

- All code comments and docstrings in Chinese (中文) to match the learner's language
- Each example should print/plot training progress so results are immediately visible
- Keep examples simple and self-contained; avoid unnecessary abstraction
