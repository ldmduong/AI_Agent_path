from uuid import uuid4

from langchain_core.messages import AIMessage

from app.graph import document_workflow


def test_document_graph_abstains_without_relevant_sources(monkeypatch):
    def fail_if_model_is_called():
        raise AssertionError("The model must not run without relevant passages.")

    monkeypatch.setattr(
        document_workflow,
        "retrieve_document_context",
        lambda question, document_ids: {"context": "", "sources": []},
    )
    monkeypatch.setattr(document_workflow, "get_chat_model", fail_if_model_is_called)

    result = document_workflow.run_document_question("Unknown topic?", [uuid4()])

    assert result["answer"] == "Không tìm thấy thông tin liên quan trong các tài liệu đã chọn."
    assert result["sources"] == []


def test_document_graph_answers_from_retrieved_context(monkeypatch):
    class FakeChatModel:
        def invoke(self, messages):
            assert "[1] policy.txt" in messages[1].content
            return AIMessage(content="Nhân viên có 15 ngày phép [1].")

    source = {
        "id": 1,
        "filename": "policy.txt",
        "page": 1,
        "chunk": 1,
        "excerpt": "Nhân viên có 15 ngày phép.",
    }
    monkeypatch.setattr(
        document_workflow,
        "retrieve_document_context",
        lambda question, document_ids: {
            "context": "[1] policy.txt (trang 1, đoạn 1):\nNhân viên có 15 ngày phép.",
            "sources": [source],
        },
    )
    monkeypatch.setattr(
        document_workflow,
        "get_chat_model",
        lambda: FakeChatModel(),
    )

    result = document_workflow.run_document_question("Chính sách phép năm?", [uuid4()])

    assert result["answer"] == (
        "Nhân viên có 15 ngày phép [1].\n\nNguồn: [1] policy.txt, trang 1, đoạn 1"
    )
    assert result["sources"] == [source]
