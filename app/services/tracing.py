import logging
import os
import re
import sqlite3
import time
from threading import Lock
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler

from app.services.state_store import record_trace_event


logger = logging.getLogger(__name__)
TRACE_ENABLED = os.getenv("AI_AGENT_TRACE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SAFE_COMPONENT = re.compile(r"[^a-zA-Z0-9_.:-]")


class SafeTraceCallbackHandler(BaseCallbackHandler):
    def __init__(self, trace_id: UUID, graph_name: str):
        self.trace_id = str(trace_id)
        self.graph_name = graph_name
        self._started_at: dict[UUID, tuple[float, str]] = {}
        self._lock = Lock()

    def _emit(self, event: str, run_id: UUID, parent_run_id: UUID | None, component: str):
        with self._lock:
            if event in {"chain_start", "tool_start", "model_start"}:
                self._started_at[run_id] = (time.perf_counter(), component)
                duration_ms = None
            else:
                started = self._started_at.pop(run_id, None)
                started_at = started[0] if started else None
                if started and component in {"chain", "tool", "model", "chat_model"}:
                    component = started[1]
                duration_ms = (
                    round((time.perf_counter() - started_at) * 1000)
                    if started_at is not None
                    else None
                )
        try:
            record_trace_event(
                self.trace_id,
                str(run_id),
                str(parent_run_id) if parent_run_id else None,
                self.graph_name,
                event,
                _SAFE_COMPONENT.sub("_", component)[:100] or "unknown",
                duration_ms,
            )
        except (OSError, sqlite3.Error):
            logger.exception("Could not persist a sanitized trace event.")

    @staticmethod
    def _component(serialized: dict | None, fallback: str) -> str:
        return str((serialized or {}).get("name") or fallback)

    def on_chain_start(
        self,
        serialized: dict | None,
        inputs: dict,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        metadata = kwargs.get("metadata") or {}
        component = (
            metadata.get("langgraph_node")
            or kwargs.get("name")
            or self._component(serialized, "chain")
        )
        self._emit(
            "chain_start",
            run_id,
            parent_run_id,
            str(component),
        )

    def on_chain_end(
        self,
        outputs: dict,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit("chain_end", run_id, parent_run_id, "chain")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit("chain_error", run_id, parent_run_id, "chain")

    def on_tool_start(
        self,
        serialized: dict | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit(
            "tool_start",
            run_id,
            parent_run_id,
            self._component(serialized, "tool"),
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit("tool_end", run_id, parent_run_id, "tool")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit("tool_error", run_id, parent_run_id, "tool")

    def on_chat_model_start(
        self,
        serialized: dict | None,
        messages: list,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit(
            "model_start",
            run_id,
            parent_run_id,
            self._component(serialized, "chat_model"),
        )

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit(
            "model_start",
            run_id,
            parent_run_id,
            self._component(serialized, "llm"),
        )

    def on_llm_end(
        self,
        response,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit("model_end", run_id, parent_run_id, "model")

    def on_chat_model_end(
        self,
        response,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit("model_end", run_id, parent_run_id, "chat_model")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit("model_error", run_id, parent_run_id, "model")

    def on_chat_model_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self._emit("model_error", run_id, parent_run_id, "model")


def traced_config(graph_name: str) -> dict:
    if not TRACE_ENABLED:
        return {}
    trace_id = uuid4()
    return {
        "callbacks": [SafeTraceCallbackHandler(trace_id, graph_name)],
        "metadata": {"trace_id": str(trace_id), "graph": graph_name},
    }
