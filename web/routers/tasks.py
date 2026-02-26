"""异步任务状态查询与取消。"""

from fastapi import APIRouter, HTTPException

from web.schemas import TaskStatusResponse, TaskCancelResponse
from web.task_manager import manager

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "status": task.status,
        "progress": task.progress,
        "message": task.message,
        "result": task.result,
        "error": task.error,
    }


@router.post("/{task_id}/cancel", response_model=TaskCancelResponse)
async def cancel_task(task_id: str):
    task = manager.cancel(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task.task_id, "status": "cancel_requested"}
