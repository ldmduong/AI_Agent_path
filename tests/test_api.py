from io import BytesIO
from uuid import UUID

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def test_upload_profile_preview_analyze_and_delete(tmp_path, monkeypatch):
    import app.main as api

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    client = TestClient(app)
    response = client.post(
        "/files",
        files={"file": ("sales.csv", b"region,revenue\nNorth,10\nSouth,5\n", "text/csv")},
    )
    assert response.status_code == 201
    dataset_id = response.json()["dataset_id"]
    UUID(dataset_id)

    profile = client.get(f"/datasets/{dataset_id}/profile")
    assert profile.json()["row_count"] == 2

    preview = client.get(f"/datasets/{dataset_id}/preview")
    assert len(preview.json()["rows"]) == 2

    analysis = client.post(
        f"/datasets/{dataset_id}/analyze",
        json={
            "operation": "aggregate",
            "group_by": ["region"],
            "metrics": [{"column": "revenue", "function": "sum"}],
        },
    )
    assert analysis.status_code == 200
    assert analysis.json()["table"]["total_rows"] == 2

    deleted = client.delete(f"/datasets/{dataset_id}")
    assert deleted.status_code == 204
    assert client.get(f"/datasets/{dataset_id}/profile").status_code == 404


def test_analyze_trend_endpoint_returns_series_and_chart(tmp_path, monkeypatch):
    import app.main as api

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    client = TestClient(app)
    uploaded = client.post(
        "/files",
        files={
            "file": (
                "monthly.csv",
                b"date,revenue\n2026-02-01,20\n2026-01-01,10\n",
                "text/csv",
            )
        },
    )
    dataset_id = uploaded.json()["dataset_id"]
    response = client.post(
        f"/datasets/{dataset_id}/analyze",
        json={
            "operation": "trend",
            "time_column": "date",
            "interval": "month",
            "metrics": [{"column": "revenue", "function": "sum"}],
            "chart": {"type": "line", "x": "date", "y": "sum_revenue"},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["trend"][0]["direction"] == "increasing"
    assert response.json()["chart"]["type"] == "line"
    assert [row["sum_revenue"] for row in response.json()["table"]["rows"]] == [10, 20]


def test_upload_rejects_unsupported_file_type(tmp_path, monkeypatch):
    import app.main as api

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    client = TestClient(app)

    response = client.post(
        "/files",
        files={"file": ("payload.txt", BytesIO(b"no"), "text/plain")},
    )

    assert response.status_code == 400


def test_document_upload_ask_and_delete(monkeypatch):
    import app.main as api

    stored_documents = set()
    monkeypatch.setattr(
        api,
        "ingest_document",
        lambda document_id, filename, content: (
            stored_documents.add(str(document_id)) or 2
        ),
    )
    monkeypatch.setattr(
        api,
        "existing_document_ids",
        lambda document_ids: {
            str(document_id) for document_id in document_ids
            if str(document_id) in stored_documents
        },
    )
    monkeypatch.setattr(
        api,
        "delete_document",
        lambda document_id: stored_documents.discard(str(document_id)),
    )
    monkeypatch.setattr(
        api,
        "list_documents",
        lambda: [
            {
                "document_id": document_id,
                "filename": "policy.txt",
                "chunk_count": 2,
            }
            for document_id in stored_documents
        ],
    )
    monkeypatch.setattr(
        api,
        "run_document_question",
        lambda question, document_ids: {
            "answer": "Nhân viên có 15 ngày phép [1].",
            "sources": [
                {
                    "id": 1,
                    "filename": "policy.txt",
                    "page": 1,
                    "chunk": 1,
                    "excerpt": "Nhân viên có 15 ngày phép.",
                }
            ],
        },
    )
    client = TestClient(app)
    uploaded = client.post(
        "/documents",
        files={"file": ("policy.txt", "Nghỉ phép năm".encode("utf-8"), "text/plain")},
    )
    assert uploaded.status_code == 201
    document_id = uploaded.json()["document_id"]
    assert uploaded.json()["chunk_count"] == 2
    assert client.get("/documents").json()["documents"][0]["document_id"] == document_id

    answer = client.post(
        "/documents/ask",
        json={
            "question": "Chính sách phép năm thế nào?",
            "document_ids": [document_id],
        },
    )
    assert answer.status_code == 200
    assert answer.json()["sources"][0]["filename"] == "policy.txt"
    assert answer.json()["model_provider"] == api.LLM_PROVIDER

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 204
    missing = client.post(
        "/documents/ask",
        json={
            "question": "Chính sách phép năm thế nào?",
            "document_ids": [document_id],
        },
    )
    assert missing.status_code == 404


def test_document_upload_rejects_unsupported_format():
    response = TestClient(app).post(
        "/documents",
        files={"file": ("report.docx", b"unsupported", "application/octet-stream")},
    )

    assert response.status_code == 400


def test_profile_xlsx_lists_sheets(tmp_path, monkeypatch):
    import app.main as api

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"sales": [2, 3]}).to_excel(writer, sheet_name="Sales", index=False)
        pd.DataFrame({"cost": [1]}).to_excel(writer, sheet_name="Costs", index=False)

    response = TestClient(app).post(
        "/files",
        files={"file": ("report.xlsx", workbook.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )

    assert response.status_code == 201
    assert response.json()["profile"]["sheets"] == ["Sales", "Costs"]
    dataset_id = response.json()["dataset_id"]
    analysis = TestClient(app).post(
        f"/datasets/{dataset_id}/analyze",
        params={"sheet": "Costs"},
        json={
            "operation": "aggregate",
            "metrics": [{"column": "cost", "function": "sum"}],
        },
    )
    assert analysis.status_code == 200
    assert analysis.json()["table"]["rows"][0]["sum_cost"] == 1


def test_ask_uses_local_model_plan_and_runs_analysis(tmp_path, monkeypatch):
    import app.main as api
    from app.schemas import AnalysisRequest
    from app.services.agent import Specialist

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    plan = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["region"],
            "metrics": [{"column": "revenue", "function": "sum"}],
            "filters": [],
            "columns": [],
            "limit": 500,
            "chart": None,
        }
    )
    monkeypatch.setattr(
        api,
        "run_agent_analysis",
        lambda question, dataframe: (
            Specialist.analytics,
            plan,
            {
                "answer": "Đã tổng hợp dữ liệu thành 2 nhóm.",
                "table": {
                    "columns": ["region", "sum_revenue"],
                    "rows": [{"region": "North", "sum_revenue": 10}, {"region": "South", "sum_revenue": 5}],
                    "total_rows": 2,
                    "truncated": False,
                },
                "chart": None,
            },
        ),
    )
    client = TestClient(app)
    uploaded = client.post(
        "/files",
        files={"file": ("sales.csv", b"region,revenue\nNorth,10\nSouth,5\n", "text/csv")},
    )
    dataset_id = uploaded.json()["dataset_id"]

    response = client.post(
        f"/datasets/{dataset_id}/ask",
        json={"question": "Tính tổng doanh thu theo vùng"},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "qwen2.5:1.5b"
    assert response.json()["specialist"] == "analytics"
    assert response.json()["table"]["total_rows"] == 2
