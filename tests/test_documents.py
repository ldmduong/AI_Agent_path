from uuid import uuid4

import pytest
from langchain_core.documents import Document

from app.services import documents


def test_text_document_is_split_and_scoped_to_its_id(monkeypatch):
    class FakeStore:
        added_documents = []
        added_ids = []

        def add_documents(self, documents, ids):
            self.added_documents = documents
            self.added_ids = ids

    store = FakeStore()
    monkeypatch.setattr(documents, "get_vector_store", lambda: store)
    document_id = uuid4()

    chunk_count = documents.ingest_document(
        document_id,
        "policy.txt",
        "Nghỉ phép năm: 15 ngày.".encode("utf-8"),
    )

    assert chunk_count == 1
    assert store.added_ids == [f"{document_id}:0"]
    assert store.added_documents[0].metadata == {
        "source": "policy.txt",
        "page": 1,
        "document_id": str(document_id),
        "chunk": 1,
        "chunk_count": 1,
    }


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("policy.docx", b"content", "Chỉ hỗ trợ"),
        ("policy.txt", b"\xff", "UTF-8"),
        ("empty.txt", b" \n", "không có văn bản"),
    ],
)
def test_invalid_documents_are_rejected(filename, content, message):
    with pytest.raises(ValueError, match=message):
        documents._load_document(filename, content)


def test_retrieval_filters_by_distance_and_returns_citation_metadata(monkeypatch):
    document_id = uuid4()

    class FakeStore:
        def similarity_search_with_score(self, question, k, filter):
            assert question == "What is the leave policy?"
            assert k == documents.TOP_K
            assert filter == {"document_id": {"$in": [str(document_id)]}}
            return [
                (
                    Document(
                        page_content="Employees receive 15 days of leave.",
                        metadata={
                            "source": "policy.pdf",
                            "page": 2,
                            "chunk": 1,
                        },
                    ),
                    0.2,
                ),
                (
                    Document(
                        page_content="An unrelated paragraph.",
                        metadata={
                            "source": "policy.pdf",
                            "page": 3,
                            "chunk": 2,
                        },
                    ),
                    documents.MAX_DISTANCE + 0.1,
                ),
            ]

    monkeypatch.setattr(documents, "get_vector_store", lambda: FakeStore())
    result = documents.retrieve_document_context(
        "What is the leave policy?",
        [document_id],
    )

    assert result["sources"] == [
        {
            "id": 1,
            "filename": "policy.pdf",
            "page": 2,
            "chunk": 1,
            "excerpt": "Employees receive 15 days of leave.",
        }
    ]
    assert "[1] policy.pdf (trang 2, đoạn 1)" in result["context"]
