import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from uuid import UUID

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import ResponseError
from pypdf import PdfReader

from app.services.agent import ModelUnavailableError


VECTOR_STORE_DIR = Path(os.getenv("AI_AGENT_VECTOR_DIR", "data/vector_store")).resolve()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MAX_DISTANCE = float(os.getenv("DOCUMENT_MAX_DISTANCE", "0.45"))
TOP_K = 5


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name="uploaded_documents",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _load_document(filename: str, content: bytes) -> list[Document]:
    extension = Path(filename).suffix.lower()
    if extension == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("File TXT phải dùng mã hóa UTF-8.") from exc
        pages = [(1, text)]
    elif extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
        except Exception as exc:
            raise ValueError("File PDF không hợp lệ hoặc không đọc được.") from exc
        if reader.is_encrypted:
            raise ValueError("Không hỗ trợ PDF được mã hóa hoặc yêu cầu mật khẩu.")
        try:
            pages = [
                (index, page.extract_text() or "")
                for index, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise ValueError("File PDF không hợp lệ hoặc không đọc được.") from exc
    else:
        raise ValueError("Chỉ hỗ trợ tài liệu TXT và PDF.")

    documents = [
        Document(
            page_content=text,
            metadata={"source": Path(filename).name, "page": page_number},
        )
        for page_number, text in pages
        if text.strip()
    ]
    if not documents:
        raise ValueError("Tài liệu không có văn bản để lập chỉ mục.")
    return documents


def ingest_document(document_id: UUID, filename: str, content: bytes) -> int:
    raw_documents = _load_document(filename, content)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(raw_documents)
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "document_id": str(document_id),
                "chunk": chunk_index + 1,
                "chunk_count": len(chunks),
            }
        )

    store = get_vector_store()
    try:
        store.add_documents(
            chunks,
            ids=[f"{document_id}:{index}" for index in range(len(chunks))],
        )
    except (httpx.RequestError, ResponseError) as exc:
        store.delete(where={"document_id": str(document_id)})
        raise ModelUnavailableError(
            f"Không thể tạo embedding với Ollama/{EMBEDDING_MODEL}."
        ) from exc
    except Exception:
        store.delete(where={"document_id": str(document_id)})
        raise
    return len(chunks)


def delete_document(document_id: UUID) -> None:
    get_vector_store().delete(where={"document_id": str(document_id)})


def existing_document_ids(document_ids: list[UUID]) -> set[str]:
    result = get_vector_store().get(
        where={"document_id": {"$in": [str(document_id) for document_id in document_ids]}},
        include=["metadatas"],
    )
    return {metadata["document_id"] for metadata in result["metadatas"]}


def list_documents() -> list[dict]:
    result = get_vector_store().get(
        where={"chunk": 1},
        include=["metadatas"],
    )
    documents = [
        {
            "document_id": metadata["document_id"],
            "filename": metadata["source"],
            "chunk_count": metadata["chunk_count"],
        }
        for metadata in result["metadatas"]
    ]
    return sorted(documents, key=lambda document: document["filename"].casefold())


def retrieve_document_context(question: str, document_ids: list[UUID]) -> dict:
    try:
        matches = get_vector_store().similarity_search_with_score(
            question,
            k=TOP_K,
            filter={"document_id": {"$in": [str(document_id) for document_id in document_ids]}},
        )
    except (httpx.RequestError, ResponseError) as exc:
        raise ModelUnavailableError(
            f"Không thể tạo embedding câu hỏi với Ollama/{EMBEDDING_MODEL}."
        ) from exc
    sources = []
    context = []
    for document, distance in matches:
        if distance > MAX_DISTANCE:
            continue
        source_id = len(sources) + 1
        source = {
            "id": source_id,
            "filename": document.metadata["source"],
            "page": document.metadata["page"],
            "chunk": document.metadata["chunk"],
            "excerpt": document.page_content,
        }
        sources.append(source)
        context.append(
            f"[{source_id}] {source['filename']} (trang {source['page']}, "
            f"đoạn {source['chunk']}):\n{document.page_content}"
        )
    return {"sources": sources, "context": "\n\n".join(context)}
