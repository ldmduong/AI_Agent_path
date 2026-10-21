import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from starlette.concurrency import run_in_threadpool

from app.graph.document_workflow import run_document_question
from app.graph.workflow import run_agent_analysis, run_analysis, tool_name_for_plan
from app.schemas import AnalysisRequest, AskRequest, DocumentAskRequest
from app.services.memory import (
    cleanup_expired_conversations,
    conversation_execution_lock,
    delete_conversation,
    delete_dataset_conversations,
    get_conversation_history,
    record_conversation_turn,
)
from app.services.agent import (
    LLM_MODEL,
    LLM_PROVIDER,
    ModelResponseError,
    ModelUnavailableError,
)
from app.services.datasets import preview_dataset, profile_dataset, read_dataset
from app.services.state_store import (
    STATE_LOCK,
    cache_analysis,
    conversation_exists,
    create_analysis_job,
    delete_dataset_records,
    get_analysis_job,
    get_cached_analysis,
    list_trace_events,
    mark_interrupted_jobs,
    touch_conversation,
    update_analysis_job,
)
from app.services.documents import (
    delete_document,
    existing_document_ids,
    ingest_document,
    list_documents,
)


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
UPLOAD_DIR = Path(os.getenv("AI_AGENT_UPLOAD_DIR", "data/uploads")).resolve()
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
ALLOWED_DOCUMENT_EXTENSIONS = {".txt", ".pdf"}
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    mark_interrupted_jobs()
    cleanup_expired_conversations()
    yield


app = FastAPI(title="AI Data Agent API", version="0.1.0", lifespan=lifespan)


def _dataset_path(dataset_id: str) -> Path:
    try:
        normalized_id = str(UUID(dataset_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Không tìm thấy dataset.") from exc

    for extension in ALLOWED_EXTENSIONS:
        path = UPLOAD_DIR / f"{normalized_id}{extension}"
        if path.is_file():
            return path
    raise HTTPException(status_code=404, detail="Không tìm thấy dataset.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/traces")
def get_traces(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"traces": list_trace_events(limit)}


@app.post("/files", status_code=201)
async def upload_file(file: UploadFile = File(...)) -> dict:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file CSV và XLSX.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid4())
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{dataset_id}.",
            suffix=f".tmp{extension}",
            dir=UPLOAD_DIR,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File vượt giới hạn 25 MB.")
                temporary_file.write(chunk)

        profile = await run_in_threadpool(profile_dataset, temporary_path)
        stored_path = UPLOAD_DIR / f"{dataset_id}{extension}"
        temporary_path.replace(stored_path)
        temporary_path = None
        return {
            "dataset_id": dataset_id,
            "filename": filename,
            "profile": profile,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        await file.close()


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)) -> dict:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if Path(filename).suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tài liệu TXT và PDF.")
    try:
        content = await file.read(MAX_DOCUMENT_BYTES + 1)
        if len(content) > MAX_DOCUMENT_BYTES:
            raise HTTPException(status_code=413, detail="Tài liệu vượt giới hạn 25 MB.")
        document_id = uuid4()
        chunk_count = await run_in_threadpool(ingest_document, document_id, filename, content)
        return {
            "document_id": str(document_id),
            "filename": filename,
            "chunk_count": chunk_count,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()


@app.get("/documents")
def get_documents() -> dict:
    return {"documents": list_documents()}


@app.post("/documents/ask")
def ask_documents(request: DocumentAskRequest) -> dict:
    document_ids = [str(document_id) for document_id in request.document_ids]
    existing_ids = existing_document_ids(request.document_ids)
    if missing_ids := set(document_ids) - existing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy tài liệu: {', '.join(sorted(missing_ids))}.",
        )
    try:
        result = run_document_question(request.question, request.document_ids)
    except ModelResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "model": LLM_MODEL,
        "model_provider": LLM_PROVIDER,
    }


@app.delete("/documents/{document_id}", status_code=204)
def remove_document(document_id: UUID) -> None:
    if str(document_id) not in existing_document_ids([document_id]):
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    delete_document(document_id)


@app.get("/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str, sheet: str | None = None) -> dict:
    path = _dataset_path(dataset_id)
    try:
        return profile_dataset(path, sheet)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/datasets/{dataset_id}/preview")
def get_preview(
    dataset_id: str,
    sheet: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    path = _dataset_path(dataset_id)
    try:
        return preview_dataset(path, sheet, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/datasets/{dataset_id}/analyze")
def analyze(
    dataset_id: str,
    request: AnalysisRequest,
    response: Response,
    sheet: str | None = None,
) -> dict:
    with STATE_LOCK:
        path = _dataset_path(dataset_id)
        dataset_id = str(UUID(dataset_id))
        cache_key = json.dumps(
            {"sheet": sheet, "request": request.model_dump(mode="json")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cached = get_cached_analysis(dataset_id, cache_key)
    if cached is not None:
        response.headers["X-Cache"] = "HIT"
        return cached
    try:
        dataframe, _, _ = read_dataset(path, sheet)
        result = run_analysis(dataframe, request)
        with STATE_LOCK:
            _dataset_path(dataset_id)
            cache_analysis(dataset_id, cache_key, result)
        response.headers["X-Cache"] = "MISS"
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _execute_question(
    dataset_id: str,
    question: str,
    sheet: str | None,
    conversation_id: UUID,
) -> dict:
    dataset_uuid = UUID(dataset_id)
    with conversation_execution_lock(dataset_uuid, conversation_id):
        return _execute_question_locked(
            dataset_id,
            question,
            sheet,
            conversation_id,
            dataset_uuid,
        )


def _execute_question_locked(
    dataset_id: str,
    question: str,
    sheet: str | None,
    conversation_id: UUID,
    dataset_uuid: UUID,
) -> dict:
    path = _dataset_path(dataset_id)
    # Keep a marker so deleting this conversation during analysis prevents it
    # from being recreated when the in-flight request finishes.
    touch_conversation(f"{dataset_uuid}:{conversation_id}")
    dataframe, _, _ = read_dataset(path, sheet)
    history = get_conversation_history(dataset_uuid, conversation_id)
    if history:
        specialist, plan, result = run_agent_analysis(question, dataframe, history)
    else:
        specialist, plan, result = run_agent_analysis(question, dataframe)
    serialized_plan = plan.model_dump(mode="json")
    with STATE_LOCK:
        _dataset_path(dataset_id)
        if conversation_exists(f"{dataset_uuid}:{conversation_id}"):
            record_conversation_turn(
                dataset_uuid,
                conversation_id,
                question,
                serialized_plan,
            )
    return {
        **result,
        "question": question,
        "specialist": specialist.value,
        "plan": serialized_plan,
        "tool": tool_name_for_plan(plan),
        "model": LLM_MODEL,
        "model_provider": LLM_PROVIDER,
        "conversation_id": str(conversation_id),
    }


@app.post("/datasets/{dataset_id}/ask")
def ask(dataset_id: str, request: AskRequest, sheet: str | None = None) -> dict:
    _dataset_path(dataset_id)
    conversation_id = request.conversation_id or uuid4()
    try:
        return _execute_question(
            dataset_id,
            request.question,
            sheet,
            conversation_id,
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _run_analysis_job(
    job_id: str,
    dataset_id: str,
    question: str,
    sheet: str | None,
    conversation_id: UUID,
) -> None:
    update_analysis_job(job_id, "running")
    try:
        result = _execute_question(
            dataset_id,
            question,
            sheet,
            conversation_id,
        )
        update_analysis_job(job_id, "succeeded", result=result)
    except (ModelUnavailableError, ModelResponseError, ValueError, HTTPException) as exc:
        error = exc.detail if isinstance(exc, HTTPException) else str(exc)
        update_analysis_job(job_id, "failed", error=error)
    except Exception:
        logger.exception("Asynchronous analysis job %s failed", job_id)
        update_analysis_job(
            job_id,
            "failed",
            error="Tác vụ thất bại do lỗi nội bộ; hãy kiểm tra log API.",
        )


@app.post("/datasets/{dataset_id}/ask/jobs", status_code=202)
def submit_analysis_job(
    dataset_id: str,
    request: AskRequest,
    background_tasks: BackgroundTasks,
    sheet: str | None = None,
) -> dict:
    with STATE_LOCK:
        _dataset_path(dataset_id)
        conversation_id = request.conversation_id or uuid4()
        job_id = uuid4()
        try:
            create_analysis_job(
                str(job_id),
                str(UUID(dataset_id)),
                sheet,
                request.question,
                str(conversation_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(
        _run_analysis_job,
        str(job_id),
        str(UUID(dataset_id)),
        request.question,
        sheet,
        conversation_id,
    )
    return {
        "job_id": str(job_id),
        "status": "queued",
        "conversation_id": str(conversation_id),
    }


@app.get("/analysis-jobs/{job_id}")
def get_analysis_job_status(job_id: UUID) -> dict:
    job = get_analysis_job(str(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ.")
    return job


@app.delete("/datasets/{dataset_id}/conversations/{conversation_id}", status_code=204)
def remove_conversation(dataset_id: str, conversation_id: UUID) -> None:
    with STATE_LOCK:
        _dataset_path(dataset_id)
        delete_conversation(UUID(dataset_id), conversation_id)


@app.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str) -> None:
    with STATE_LOCK:
        path = _dataset_path(dataset_id)
        dataset_uuid = UUID(dataset_id)
        delete_dataset_conversations(dataset_uuid)
        delete_dataset_records(str(dataset_uuid))
        path.unlink()
