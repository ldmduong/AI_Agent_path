from uuid import UUID, uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import AnalysisRequest
from app.services import memory, state_store
from app.services.agent import Specialist


def test_analysis_cache_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    result = {"table": {"rows": [{"total": 42}]}}

    state_store.cache_analysis("dataset", "request", result)
    assert state_store.get_cached_analysis("dataset", "request") == result

    monkeypatch.setattr(state_store, "CACHE_TTL_SECONDS", -1)
    state_store.cache_analysis("dataset", "expired", result)
    assert state_store.get_cached_analysis("dataset", "expired") is None


def test_conversation_rejects_a_second_active_job(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    state_store.create_analysis_job(
        str(uuid4()), "dataset", None, "first question", "conversation"
    )

    with pytest.raises(ValueError, match="đang có tác vụ"):
        state_store.create_analysis_job(
            str(uuid4()), "dataset", None, "second question", "conversation"
        )


def test_analyze_endpoint_returns_cached_result(tmp_path, monkeypatch):
    import app.main as api

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        api,
        "run_analysis",
        lambda dataframe, request: calls.append(request) or {"table": {"rows": []}},
    )
    client = TestClient(app)
    uploaded = client.post(
        "/files",
        files={"file": ("sales.csv", b"revenue\n10\n", "text/csv")},
    )
    dataset_id = uploaded.json()["dataset_id"]
    request = {
        "operation": "aggregate",
        "metrics": [{"column": "revenue", "function": "sum"}],
    }

    first = client.post(f"/datasets/{dataset_id}/analyze", json=request)
    second = client.post(f"/datasets/{dataset_id}/analyze", json=request)

    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert len(calls) == 1


def test_conversation_checkpoint_persists_limited_plan_history():
    dataset_id = uuid4()
    conversation_id = uuid4()
    plan = {
        "operation": "aggregate",
        "group_by": ["region"],
        "metrics": [{"column": "revenue", "function": "sum"}],
        "filters": [{"column": "customer", "value": "private"}],
    }

    for index in range(12):
        memory.record_conversation_turn(
            dataset_id,
            conversation_id,
            f"Question {index}",
            plan,
        )

    history = memory.get_conversation_history(dataset_id, conversation_id)
    assert len(history) == 10
    assert history[0]["question"] == "Question 2"
    assert history[-1]["metrics"] == plan["metrics"]
    assert "filters" not in history[-1]

    memory.delete_conversation(dataset_id, conversation_id)
    assert memory.get_conversation_history(dataset_id, conversation_id) == []


def test_async_job_uses_and_can_clear_conversation_memory(tmp_path, monkeypatch):
    import app.main as api

    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)

    plan = AnalysisRequest(
        operation="aggregate",
        group_by=["region"],
        metrics=[{"column": "revenue", "function": "sum"}],
    )
    calls = []

    def analyze(question, dataframe, history=None):
        calls.append(history or [])
        return Specialist.analytics, plan, {
            "table": {
                "columns": ["region", "sum_revenue"],
                "rows": [{"region": "North", "sum_revenue": 10}],
                "total_rows": 1,
                "truncated": False,
            },
            "chart": None,
        }

    monkeypatch.setattr(api, "run_agent_analysis", analyze)
    client = TestClient(app)
    uploaded = client.post(
        "/files",
        files={"file": ("sales.csv", b"region,revenue\nNorth,10\n", "text/csv")},
    )
    dataset_id = uploaded.json()["dataset_id"]
    conversation_id = str(uuid4())
    submitted = client.post(
        f"/datasets/{dataset_id}/ask/jobs",
        json={
            "question": "Tổng doanh thu theo vùng",
            "conversation_id": conversation_id,
        },
    )

    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    job = client.get(f"/analysis-jobs/{job_id}")
    assert job.json()["status"] == "succeeded"
    assert job.json()["result"]["conversation_id"] == conversation_id

    followup = client.post(
        f"/datasets/{dataset_id}/ask",
        json={
            "question": "Tiếp tục phân tích",
            "conversation_id": conversation_id,
        },
    )
    assert followup.status_code == 200
    assert len(calls[1]) == 1

    deleted = client.delete(
        f"/datasets/{dataset_id}/conversations/{conversation_id}"
    )
    assert deleted.status_code == 204
    assert memory.get_conversation_history(
        UUID(dataset_id),
        UUID(conversation_id),
    ) == []
