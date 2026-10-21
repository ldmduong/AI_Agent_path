import pytest
import pandas as pd

from app.services import agent
from app.services.agent import ModelResponseError, Specialist, SpecialistRoute, _validate_plan


def test_plan_removes_unrequested_filters_and_chart():
    payload = {
        "operation": "aggregate",
        "group_by": ["month"],
        "metrics": [{"column": "revenue", "function": "sum"}],
        "filters": [{"column": "revenue", "operator": "gt", "value": 10}],
        "columns": ["month", "revenue"],
        "limit": 10,
        "chart": {"type": "bar", "x": "month", "y": "sum_revenue"},
    }

    plan = _validate_plan(payload, "Tính tổng doanh thu theo tháng", Specialist.analytics)

    assert plan.filters == []
    assert plan.columns == []
    assert plan.chart is None
    assert plan.limit == 500


def test_filter_plan_keeps_explicit_filter_but_drops_aggregate_fields():
    plan = _validate_plan(
        {
            "operation": "filter",
            "group_by": ["month"],
            "metrics": [{"column": "revenue", "function": "sum"}],
            "filters": [{"column": "revenue", "operator": "gt", "value": 10}],
            "columns": ["month", "revenue"],
            "limit": 10,
            "chart": None,
        },
        "Lọc doanh thu trên 10",
        Specialist.filter,
    )

    assert plan.group_by == []
    assert plan.metrics == []
    assert plan.filters[0].value == 10
    assert plan.columns == ["month", "revenue"]


def test_filter_plan_requires_an_explicit_condition():
    with pytest.raises(ModelResponseError, match="điều kiện lọc"):
        _validate_plan(
            {
                "operation": "filter",
                "group_by": [],
                "metrics": [],
                "filters": [],
                "columns": [],
                "limit": 500,
                "chart": None,
            },
            "Tìm dữ liệu",
            Specialist.filter,
        )


def test_supervisor_selects_specialist_with_configured_chat_model(monkeypatch):
    class StructuredModel:
        def with_structured_output(self, schema):
            assert schema is SpecialistRoute
            return self

        def invoke(self, messages):
            return SpecialistRoute(specialist=Specialist.visualization)

    monkeypatch.setattr(agent, "get_chat_model", lambda: StructuredModel())

    specialist = agent.route_question(
        "Vẽ biểu đồ doanh thu theo tháng",
        pd.DataFrame({"month": ["Jan"], "revenue": [10]}),
    )

    assert specialist == Specialist.visualization


def test_supervisor_routes_plain_trend_request_to_analytics(monkeypatch):
    class StructuredModel:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return SpecialistRoute(specialist=Specialist.visualization)

    monkeypatch.setattr(agent, "get_chat_model", lambda: StructuredModel())

    specialist = agent.route_question(
        "Xu hướng doanh thu theo tháng",
        pd.DataFrame({"date": ["2026-01-01"], "revenue": [10]}),
    )

    assert specialist == Specialist.analytics


def test_supervisor_routes_explicit_filter_intent_to_filter_specialist(monkeypatch):
    class StructuredModel:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return SpecialistRoute(specialist=Specialist.analytics)

    monkeypatch.setattr(agent, "get_chat_model", lambda: StructuredModel())

    specialist = agent.route_question(
        "Lọc các đơn hàng có doanh thu lớn hơn 10",
        pd.DataFrame({"revenue": [8, 15]}),
    )

    assert specialist == Specialist.filter


def test_visualization_plan_uses_aggregated_metric_as_y_axis():
    plan = _validate_plan(
        {
            "operation": "aggregate",
            "group_by": ["month"],
            "metrics": [{"column": "revenue", "function": "sum"}],
            "filters": [],
            "columns": [],
            "limit": 500,
            "chart": {"type": "bar", "x": "month", "y": "revenue"},
        },
        "Create a bar chart of total revenue by month",
        Specialist.visualization,
    )

    assert plan.chart.y == "sum_revenue"


def test_visualization_expert_accepts_trend_axis():
    plan = _validate_plan(
        {
            "operation": "trend",
            "time_column": "date",
            "interval": "month",
            "metrics": [{"column": "revenue", "function": "sum"}],
            "filters": [],
            "columns": [],
            "limit": 500,
            "chart": {"type": "line", "x": "date", "y": "sum_revenue"},
        },
        "Vẽ biểu đồ xu hướng doanh thu theo tháng",
        Specialist.visualization,
    )

    assert plan.operation.value == "trend"
    assert plan.chart.x == "date"


def test_explicit_trend_question_normalizes_aggregate_plan():
    plan = _validate_plan(
        {
            "operation": "aggregate",
            "group_by": ["date"],
            "metrics": [{"column": "revenue", "function": "sum"}],
            "filters": [],
            "columns": [],
            "limit": 500,
            "chart": None,
        },
        "Xu hướng doanh thu theo tháng",
        Specialist.analytics,
        pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-02-01"],
                "revenue": [10, 20],
            }
        ),
    )

    assert plan.operation.value == "trend"
    assert plan.time_column == "date"
    assert plan.interval.value == "month"


def test_model_factory_uses_provider_and_model_configuration(monkeypatch):
    calls = {}

    def fake_init_chat_model(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(agent, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(agent, "LLM_MODEL", "model-for-test")
    agent.get_chat_model.cache_clear()
    try:
        agent.get_chat_model()
    finally:
        agent.get_chat_model.cache_clear()

    assert calls == {
        "model": "model-for-test",
        "model_provider": "openai",
        "temperature": 0,
    }
