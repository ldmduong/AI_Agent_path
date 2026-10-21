from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, TypedDict
from uuid import UUID

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.services.state_store import (
    STATE_DB_PATH,
    STATE_LOCK,
    delete_conversation_record,
    get_dataset_conversations,
    get_expired_conversations,
    touch_conversation,
)


class ConversationState(TypedDict, total=False):
    turn: dict
    turns: list[dict]


def _record_turn(state: ConversationState) -> dict:
    return {"turns": (state.get("turns", []) + [state["turn"]])[-10:]}


_builder = StateGraph(ConversationState)
_builder.add_node("record_turn", _record_turn)
_builder.add_edge(START, "record_turn")
_builder.add_edge("record_turn", END)

@lru_cache(maxsize=4)
def _get_checkpoint_graph(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return _builder.compile(checkpointer=checkpointer), checkpointer


def _thread_id(dataset_id: UUID, conversation_id: UUID) -> str:
    return f"{dataset_id}:{conversation_id}"


_CONVERSATION_LOCKS: dict[str, tuple[Any, int]] = {}
_CONVERSATION_LOCKS_GUARD = threading.Lock()


@contextmanager
def conversation_execution_lock(
    dataset_id: UUID, conversation_id: UUID
) -> Iterator[None]:
    thread_id = _thread_id(dataset_id, conversation_id)
    with _CONVERSATION_LOCKS_GUARD:
        lock, users = _CONVERSATION_LOCKS.get(
            thread_id, (threading.Lock(), 0)
        )
        _CONVERSATION_LOCKS[thread_id] = (lock, users + 1)
    try:
        with lock:
            yield
    finally:
        with _CONVERSATION_LOCKS_GUARD:
            lock, users = _CONVERSATION_LOCKS[thread_id]
            if users == 1:
                del _CONVERSATION_LOCKS[thread_id]
            else:
                _CONVERSATION_LOCKS[thread_id] = (lock, users - 1)


def get_conversation_history(dataset_id: UUID, conversation_id: UUID) -> list[dict]:
    thread_id = _thread_id(dataset_id, conversation_id)
    cleanup_expired_conversations()
    with STATE_LOCK:
        graph, _ = _get_checkpoint_graph(str(STATE_DB_PATH))
        snapshot = graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if not snapshot.values:
            return []
        return snapshot.values.get("turns", [])


def record_conversation_turn(
    dataset_id: UUID,
    conversation_id: UUID,
    question: str,
    plan: dict,
) -> None:
    thread_id = _thread_id(dataset_id, conversation_id)
    turn = {
        "question": question,
        "operation": plan["operation"],
        "group_by": plan.get("group_by", []),
        "time_column": plan.get("time_column"),
        "interval": plan.get("interval"),
        "metrics": plan.get("metrics", []),
    }
    with STATE_LOCK:
        graph, _ = _get_checkpoint_graph(str(STATE_DB_PATH))
        graph.invoke(
            {"turn": turn},
            {"configurable": {"thread_id": thread_id}},
        )
        touch_conversation(thread_id)


def delete_conversation(dataset_id: UUID, conversation_id: UUID) -> None:
    thread_id = _thread_id(dataset_id, conversation_id)
    with STATE_LOCK:
        _, checkpointer = _get_checkpoint_graph(str(STATE_DB_PATH))
        checkpointer.delete_thread(thread_id)
        delete_conversation_record(thread_id)


def cleanup_expired_conversations() -> None:
    with STATE_LOCK:
        _, checkpointer = _get_checkpoint_graph(str(STATE_DB_PATH))
        for thread_id in get_expired_conversations():
            checkpointer.delete_thread(thread_id)
            delete_conversation_record(thread_id)


def delete_dataset_conversations(dataset_id: UUID) -> None:
    with STATE_LOCK:
        _, checkpointer = _get_checkpoint_graph(str(STATE_DB_PATH))
        for thread_id in get_dataset_conversations(str(dataset_id)):
            checkpointer.delete_thread(thread_id)
            delete_conversation_record(thread_id)
