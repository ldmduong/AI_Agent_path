import pandas as pd
import pytest

from app.schemas import AnalysisRequest
from app.graph import workflow
from app.services.agent import Specialist
from app.tools.data_tools import DATA_ANALYSIS_TOOLS


def test_graph_plans_then_executes_analysis(monkeypatch):
    dataframe = pd.DataFrame({"month": ["Jan", "Feb", "Jan"], "revenue": [10, 5, 7]})
    plan = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["month"],
            "metrics": [{"column": "revenue", "function": "sum"}],
        }
    )
    monkeypatch.setattr(
        workflow,
        "route_question",
        lambda question, frame: Specialist.analytics,
    )
    monkeypatch.setattr(
        workflow,
        "create_analysis_plan",
        lambda question, frame, specialist: plan,
    )

    specialist, returned_plan, result = workflow.run_agent_analysis(
        "total revenue by month",
        dataframe,
    )

    assert specialist == Specialist.analytics
    assert returned_plan == plan
    assert {row["month"]: row["sum_revenue"] for row in result["table"]["rows"]} == {
        "Jan": 17,
        "Feb": 5,
    }


def test_graph_passes_conversation_history_to_specialists(monkeypatch):
    dataframe = pd.DataFrame({"revenue": [10]})
    history = [{"question": "Tính doanh thu", "operation": "aggregate"}]
    plan = AnalysisRequest(
        operation="aggregate",
        metrics=[{"column": "revenue", "function": "sum"}],
    )
    observed = {}

    def route_question(question, frame, previous_turns):
        observed["route_history"] = previous_turns
        return Specialist.analytics

    def create_analysis_plan(question, frame, specialist, previous_turns):
        observed["plan_history"] = previous_turns
        return plan

    monkeypatch.setattr(workflow, "route_question", route_question)
    monkeypatch.setattr(workflow, "create_analysis_plan", create_analysis_plan)

    workflow.run_agent_analysis("Tiếp tục", dataframe, history)

    assert observed == {"route_history": history, "plan_history": history}


def test_supervisor_routes_visualization_to_chart_specialist(monkeypatch):
    dataframe = pd.DataFrame({"month": ["Jan"], "revenue": [10]})
    plan = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["month"],
            "metrics": [{"column": "revenue", "function": "sum"}],
            "chart": {"type": "bar", "x": "month", "y": "sum_revenue"},
        }
    )
    monkeypatch.setattr(
        workflow,
        "route_question",
        lambda question, frame: Specialist.visualization,
    )
    monkeypatch.setattr(
        workflow,
        "create_analysis_plan",
        lambda question, frame, specialist: (
            plan if specialist == Specialist.visualization else None
        ),
    )

    specialist, returned_plan, result = workflow.run_agent_analysis(
        "Vẽ biểu đồ doanh thu theo tháng",
        dataframe,
    )

    assert specialist == Specialist.visualization
    assert returned_plan == plan
    assert result["chart"] == {"type": "bar", "x": "month", "y": "sum_revenue"}


def test_custom_tools_are_registered_with_expected_names():
    assert {tool.name for tool in DATA_ANALYSIS_TOOLS} == {
        "profile_dataset",
        "list_columns",
        "filter_dataset",
        "aggregate_dataset",
        "analyze_trend",
        "calculate_statistics",
        "get_chart_data",
    }


def test_profile_and_list_columns_run_as_tools():
    dataframe = pd.DataFrame({"revenue": [10, None], "region": ["North", "South"]})

    profile = workflow.run_analysis(
        dataframe,
        AnalysisRequest(operation="profile"),
    )
    columns = workflow.run_analysis(
        dataframe,
        AnalysisRequest(operation="list_columns"),
    )

    assert profile["profile"]["row_count"] == 2
    assert profile["profile"]["columns"][0]["missing_count"] == 1
    assert [row["name"] for row in columns["table"]["rows"]] == ["revenue", "region"]


def test_statistics_can_target_selected_numeric_columns():
    dataframe = pd.DataFrame({"revenue": [10, 20], "units": [1, 2]})
    plan = AnalysisRequest(operation="describe", columns=["units"])

    result = workflow.run_analysis(dataframe, plan)

    assert result["table"]["total_rows"] == 1
    assert result["table"]["rows"][0]["column"] == "units"


def test_chart_data_tool_routes_analysis_and_chart_spec():
    dataframe = pd.DataFrame({"month": ["Jan", "Feb"], "revenue": [10, 20]})
    plan = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["month"],
            "metrics": [{"column": "revenue", "function": "sum"}],
            "chart": {"type": "bar", "x": "month", "y": "sum_revenue"},
        }
    )

    result = workflow.run_analysis(dataframe, plan)

    assert result["chart"] == {"type": "bar", "x": "month", "y": "sum_revenue"}
    assert result["table"]["total_rows"] == 2


def test_tool_errors_are_not_swallowed():
    dataframe = pd.DataFrame({"revenue": [10]})
    plan = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["missing"],
            "metrics": [{"column": "revenue", "function": "sum"}],
        }
    )

    with pytest.raises(ValueError, match="Không tìm thấy cột"):
        workflow.run_analysis(dataframe, plan)


def test_graph_reports_chart_tool_for_chart_plan():
    plan = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["month"],
            "metrics": [{"column": "revenue", "function": "sum"}],
            "chart": {"type": "bar", "x": "month", "y": "sum_revenue"},
        }
    )

    assert workflow.tool_name_for_plan(plan) == "get_chart_data"
