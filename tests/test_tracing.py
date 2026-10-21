import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.graph.workflow import run_analysis
from app.schemas import AnalysisRequest
from app.services import state_store, tracing


def test_graph_tracing_records_only_safe_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "traces.sqlite3")
    monkeypatch.setattr(tracing, "TRACE_ENABLED", True)

    run_analysis(
        dataframe=pd.DataFrame(
            {"customer": ["sensitive-customer-name"], "revenue": [12]}
        ),
        plan=AnalysisRequest(
            operation="aggregate",
            metrics=[{"column": "revenue", "function": "sum"}],
        ),
    )

    traces = state_store.list_trace_events()
    assert traces
    assert any(event["graph_name"] == "data_analysis" for event in traces)
    assert any(event["event"] == "tool_start" for event in traces)
    assert not any(
        "sensitive-customer-name" in str(event)
        for event in traces
    )
    assert all("inputs" not in event and "outputs" not in event for event in traces)


def test_trace_can_be_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "TRACE_ENABLED", False)

    assert tracing.traced_config("data_analysis") == {}


def test_trace_endpoint_returns_recent_trace_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "traces.sqlite3")
    state_store.record_trace_event(
        "trace-id",
        "run-id",
        None,
        "data_analysis",
        "tool_end",
        "aggregate_dataset",
        12,
    )

    response = TestClient(app).get("/traces?limit=1")

    assert response.status_code == 200
    assert response.json()["traces"][0]["component"] == "aggregate_dataset"
    assert "inputs" not in response.json()["traces"][0]
