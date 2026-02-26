from fastapi.testclient import TestClient

from web.app import app


client = TestClient(app)


def test_healthz_ok():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_async_task_submit_and_query():
    payload = {"episodes": 10, "n_runs": 1, "seed": 42, "timeout_seconds": 120}
    submit = client.post("/api/gridworld/run", json=payload)
    assert submit.status_code == 200
    task_id = submit.json()["task_id"]

    status = client.get(f"/api/tasks/{task_id}")
    assert status.status_code == 200
    assert status.json()["status"] in {"queued", "running", "completed"}


def test_sync_api_has_seed_and_params():
    payload = {"episodes": 10, "n_runs": 1, "seed": 123}
    resp = client.post("/api/policy-gradient/run-sync", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["seed"] == 123
    assert "params" in data
