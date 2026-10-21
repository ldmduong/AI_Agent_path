import json
from typing import Any, NotRequired, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.schemas import AnalysisOperation, AnalysisRequest
from app.services.agent import Specialist, create_analysis_plan, route_question
from app.services.tracing import traced_config
from app.tools.data_tools import CHART_DATA_TOOL, DATA_ANALYSIS_TOOLS, TOOLS_BY_OPERATION


class AgentState(TypedDict):
    question: NotRequired[str]
    dataframe: pd.DataFrame
    history: NotRequired[list[dict]]
    specialist: NotRequired[str]
    plan: NotRequired[AnalysisRequest]
    messages: NotRequired[list[Any]]
    result: NotRequired[dict]


def supervisor_node(state: AgentState) -> dict:
    if "plan" in state:
        plan = state["plan"]
        if plan.charts:
            specialist = Specialist.visualization
        elif plan.operation == AnalysisOperation.filter:
            specialist = Specialist.filter
        elif plan.operation == AnalysisOperation.aggregate:
            specialist = Specialist.analytics
        else:
            specialist = Specialist.dataset
    else:
        history = state.get("history")
        specialist = (
            route_question(state["question"], state["dataframe"], history)
            if history
            else route_question(state["question"], state["dataframe"])
        )
    return {"specialist": specialist.value}


def specialist_node(specialist: Specialist):
    def plan(state: AgentState) -> dict:
        if "plan" in state:
            return {}
        history = state.get("history")
        if history:
            plan = create_analysis_plan(
                state["question"],
                state["dataframe"],
                specialist,
                history,
            )
        else:
            plan = create_analysis_plan(
                state["question"],
                state["dataframe"],
                specialist,
            )
        return {
            "plan": plan
        }

    return plan


def dispatch_node(state: AgentState) -> dict:
    plan = state["plan"]
    tool = CHART_DATA_TOOL if plan.charts else TOOLS_BY_OPERATION[plan.operation]
    if tool == CHART_DATA_TOOL:
        arguments = {"query_spec": plan.model_dump(mode="json")}
    elif plan.operation == AnalysisOperation.filter:
        arguments = plan.model_dump(mode="json", include={"filters", "columns", "limit"})
    elif plan.operation == AnalysisOperation.aggregate:
        arguments = plan.model_dump(
            mode="json",
            include={"group_by", "metrics", "filters", "limit"},
        )
    elif plan.operation == AnalysisOperation.trend:
        arguments = plan.model_dump(
            mode="json",
            include={"time_column", "interval", "metrics", "filters", "limit"},
        )
    elif plan.operation == AnalysisOperation.describe:
        arguments = plan.model_dump(mode="json", include={"columns", "limit"})
    else:
        arguments = {}
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool.name,
                        "args": arguments,
                        "id": "analysis_tool_call",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def result_node(state: AgentState) -> dict:
    tool_message = next(
        message for message in reversed(state["messages"])
        if isinstance(message, ToolMessage)
    )
    return {"result": json.loads(tool_message.content)}


def tool_name_for_plan(plan: AnalysisRequest) -> str:
    return (CHART_DATA_TOOL if plan.charts else TOOLS_BY_OPERATION[plan.operation]).name


builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
for specialist in Specialist:
    node_name = f"{specialist.value}_agent"
    builder.add_node(node_name, specialist_node(specialist))
    builder.add_edge(node_name, "dispatch")
builder.add_node("dispatch", dispatch_node)
builder.add_node("tools", ToolNode(DATA_ANALYSIS_TOOLS, handle_tool_errors=False))
builder.add_node("result", result_node)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["specialist"],
    {specialist.value: f"{specialist.value}_agent" for specialist in Specialist},
)
builder.add_edge("dispatch", "tools")
builder.add_edge("tools", "result")
builder.add_edge("result", END)
analysis_graph = builder.compile()


def run_agent_analysis(
    question: str,
    dataframe: pd.DataFrame,
    history: list[dict] | None = None,
) -> tuple[Specialist, AnalysisRequest, dict]:
    initial_state = {"question": question, "dataframe": dataframe}
    if history:
        initial_state["history"] = history
    state = analysis_graph.invoke(
        initial_state,
        config=traced_config("data_analysis"),
    )
    return Specialist(state["specialist"]), state["plan"], state["result"]


def run_analysis(dataframe: pd.DataFrame, plan: AnalysisRequest) -> dict:
    state = analysis_graph.invoke(
        {"plan": plan, "dataframe": dataframe},
        config=traced_config("data_analysis"),
    )
    return state["result"]
