import pandas as pd
import pytest

from app.schemas import AnalysisRequest
from app.services.analysis import analyze_dataframe


def test_aggregate_returns_expected_groups_and_chart():
    dataframe = pd.DataFrame(
        {"region": ["North", "South", "North"], "revenue": [10, 5, 7]}
    )
    request = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["region"],
            "metrics": [{"column": "revenue", "function": "sum"}],
            "chart": {"type": "bar", "x": "region", "y": "sum_revenue"},
        }
    )

    result = analyze_dataframe(dataframe, request)

    assert result["table"]["total_rows"] == 2
    assert {row["region"]: row["sum_revenue"] for row in result["table"]["rows"]} == {
        "North": 17,
        "South": 5,
    }
    assert result["chart"]["type"] == "bar"


def test_filter_rejects_missing_column():
    dataframe = pd.DataFrame({"revenue": [10, 5]})
    request = AnalysisRequest.model_validate(
        {
            "operation": "filter",
            "filters": [{"column": "unknown", "operator": "eq", "value": 1}],
        }
    )

    with pytest.raises(ValueError, match="Không tìm thấy cột"):
        analyze_dataframe(dataframe, request)


def test_rejects_arbitrary_sql_fields():
    with pytest.raises(ValueError):
        AnalysisRequest.model_validate(
            {
                "operation": "filter",
                "filters": [{"column": "revenue", "operator": "eq", "value": 1}],
                "sql": "DROP TABLE data",
            }
        )


def test_filter_numeric_values_and_returns_only_matches():
    dataframe = pd.DataFrame({"revenue": [10, 5, 20]})
    request = AnalysisRequest.model_validate(
        {
            "operation": "filter",
            "filters": [{"column": "revenue", "operator": "gte", "value": "10"}],
        }
    )

    result = analyze_dataframe(dataframe, request)

    assert [row["revenue"] for row in result["table"]["rows"]] == [10, 20]


def test_monthly_trend_sorts_periods_and_reports_endpoint_change():
    dataframe = pd.DataFrame(
        {
            "date": ["2026-03-10", "2026-01-05", "2026-02-14", "not-a-date"],
            "revenue": [30, 10, 20, 999],
        }
    )
    request = AnalysisRequest.model_validate(
        {
            "operation": "trend",
            "time_column": "date",
            "interval": "month",
            "metrics": [{"column": "revenue", "function": "sum"}],
            "chart": {"type": "line", "x": "date", "y": "sum_revenue"},
        }
    )

    result = analyze_dataframe(dataframe, request)

    assert result["table"]["rows"] == [
        {"date": "2026-01-01T00:00:00.000", "sum_revenue": 10},
        {"date": "2026-02-01T00:00:00.000", "sum_revenue": 20},
        {"date": "2026-03-01T00:00:00.000", "sum_revenue": 30},
    ]
    assert result["trend"][0]["direction"] == "increasing"
    assert result["trend"][0]["change"] == 20
    assert result["trend"][0]["change_percent"] == 200


def test_trend_rejects_unparseable_time_column():
    request = AnalysisRequest.model_validate(
        {
            "operation": "trend",
            "time_column": "date",
            "interval": "month",
            "metrics": [{"column": "revenue", "function": "mean"}],
        }
    )

    with pytest.raises(ValueError, match="không có giá trị thời gian hợp lệ"):
        analyze_dataframe(
            pd.DataFrame({"date": ["invalid", "also invalid"], "revenue": [2, 4]}),
            request,
        )


@pytest.mark.parametrize(
    "chart_type",
    ["bar", "line", "scatter", "histogram", "area", "box", "violin", "pie"],
)
def test_supported_chart_types_are_validated(chart_type):
    dataframe = pd.DataFrame(
        {"region": ["A", "B"], "score": [1, 2], "sales": [10, 20]}
    )
    group_column = "score" if chart_type == "scatter" else "region"
    chart_x = "sum_sales" if chart_type == "histogram" else group_column
    chart = {
        "type": chart_type,
        "x": chart_x,
        "y": None if chart_type == "histogram" else "sum_sales",
    }
    request = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": [group_column],
            "metrics": [{"column": "sales", "function": "sum"}],
            "chart": chart,
        }
    )

    result = analyze_dataframe(dataframe, request)

    assert result["chart"]["type"] == chart_type


def test_scatter_requires_numeric_axes():
    dataframe = pd.DataFrame({"region": ["A", "B"], "sales": [10, 20]})
    request = AnalysisRequest.model_validate(
        {
            "operation": "aggregate",
            "group_by": ["region"],
            "metrics": [{"column": "sales", "function": "sum"}],
            "chart": {"type": "scatter", "x": "region", "y": "sum_sales"},
        }
    )

    with pytest.raises(ValueError, match="scatter cần hai trục kiểu số"):
        analyze_dataframe(dataframe, request)
