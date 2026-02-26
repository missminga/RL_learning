"""训练异步任务管理：提交、查询、取消，含并发/超时保护。"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TaskState:
    task_id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    timeout_seconds: int = 600
    result: dict[str, Any] | None = None
    error: str | None = None
    cancelled: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)


class TaskManager:
    def __init__(self):
        max_workers = int(os.getenv("RL_MAX_CONCURRENT_TASKS", "2"))
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskState] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        kind: str,
        fn: Callable[..., dict[str, Any]],
        timeout_seconds: int = 600,
        **kwargs,
    ) -> TaskState:
        task_id = str(uuid.uuid4())
        state = TaskState(task_id=task_id, kind=kind, timeout_seconds=timeout_seconds)
        with self._lock:
            self._tasks[task_id] = state

        def _runner():
            state.status = "running"
            state.started_at = time.time()

            def _should_stop() -> bool:
                if state.cancel_event.is_set():
                    return True
                if time.time() - (state.started_at or time.time()) > timeout_seconds:
                    state.error = f"task timeout after {timeout_seconds}s"
                    return True
                return False

            def _on_progress(
                run_idx: int, ep: int, reward: float, steps: int, metric: float
            ):
                total = max(
                    kwargs.get("n_runs", 1)
                    * kwargs.get("episodes", kwargs.get("steps", 1)),
                    1,
                )
                current = (
                    run_idx * kwargs.get("episodes", kwargs.get("steps", 1)) + ep + 1
                )
                state.progress = min(1.0, current / total)
                state.message = f"run={run_idx + 1}, episode={ep + 1}, reward={round(float(reward), 3)}, steps={steps}"

            try:
                result = fn(
                    **kwargs,
                    should_stop=_should_stop,
                    on_episode_end=_on_progress,
                )
                if state.cancel_event.is_set():
                    state.status = "cancelled"
                    state.cancelled = True
                elif state.error and "timeout" in state.error:
                    state.status = "timeout"
                else:
                    state.status = "completed"
                state.result = result
            except Exception as e:  # noqa: BLE001
                state.status = "failed"
                state.error = str(e)
            finally:
                state.progress = (
                    1.0
                    if state.status in {"completed", "failed", "cancelled", "timeout"}
                    else state.progress
                )
                state.finished_at = time.time()

        fut = self._executor.submit(_runner)
        with self._lock:
            self._futures[task_id] = fut
        return state

    def get(self, task_id: str) -> TaskState | None:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> TaskState | None:
        state = self._tasks.get(task_id)
        if not state:
            return None
        state.cancel_event.set()
        fut = self._futures.get(task_id)
        if fut and not fut.done():
            fut.cancel()
        return state


manager = TaskManager()
