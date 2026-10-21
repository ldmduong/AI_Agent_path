import json
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from app.schemas import (
    AnalysisOperation,
    AnalysisRequest,
    FilterCondition,
    MetricSpec,
    TrendInterval,
)
from app.services.analysis import analyze_dataframe


@tool
def profile_dataset(dataframe: Annotated[pd.DataFrame, InjectedState("dataframe")]) -> str:
    """Return dataset dimensions, column types, and missing-value counts."""
    return json.dumps(
        analyze_dataframe(dataframe, AnalysisRequest(operation=AnalysisOperation.profile)),
        ensure_ascii=False,
    )


@tool
def list_columns(dataframe: Annotated[pd.DataFrame, InjectedState("dataframe")]) -> str:
    """List dataset column names and detected data types."""
    return json.dumps(
        analyze_dataframe(dataframe, AnalysisRequest(operation=AnalysisOperation.list_columns)),
        ensure_ascii=False,
    )


@tool
def filter_dataset(
    dataframe: Annotated[pd.DataFrame, InjectedState("dataframe")],
    filters: list[FilterCondition],
    columns: list[str] = Field(default_factory=list, max_length=100),
    limit: int = Field(default=500, ge=1, le=500),
) -> str:
    """Filter rows using validated column names and comparison operators."""
    request = AnalysisRequest(
        operation=AnalysisOperation.filter,
        filters=filters,
        columns=columns,
        limit=limit,
    )
    return json.dumps(analyze_dataframe(dataframe, request), ensure_ascii=False)


@tool
def aggregate_dataset(
    dataframe: Annotated[pd.DataFrame, InjectedState("dataframe")],
    metrics: list[MetricSpec] = Field(min_length=1, max_length=10),
    group_by: list[str] = Field(default_factory=list, max_length=5),
    filters: list[FilterCondition] = Field(default_factory=list, max_length=10),
    limit: int = Field(default=500, ge=1, le=500),
) -> str:
    """Group rows and calculate allowed numeric aggregates."""
    request = AnalysisRequest(
        operation=AnalysisOperation.aggregate,
        group_by=group_by,
        metrics=metrics,
        filters=filters,
        limit=limit,
    )
    return json.dumps(analyze_dataframe(dataframe, request), ensure_ascii=False)


@tool
def analyze_trend(
    dataframe: Annotated[pd.DataFrame, InjectedState("dataframe")],
    time_column: str,
    interval: TrendInterval,
    metrics: list[MetricSpec] = Field(min_length=1, max_length=10),
    filters: list[FilterCondition] = Field(default_factory=list, max_length=10),
    limit: int = Field(default=500, ge=1, le=500),
) -> str:
    """Aggregate numeric metrics across validated day, week, month, quarter, or year periods."""
    request = AnalysisRequest(
        operation=AnalysisOperation.trend,
        time_column=time_column,
        interval=interval,
        metrics=metrics,
        filters=filters,
        limit=limit,
    )
    return json.dumps(analyze_dataframe(dataframe, request), ensure_ascii=False)


@tool
def calculate_statistics(
    dataframe: Annotated[pd.DataFrame, InjectedState("dataframe")],
    columns: list[str] = Field(default_factory=list, max_length=100),
    limit: int = Field(default=500, ge=1, le=500),
) -> str:
    """Calculate descriptive statistics for all or selected numeric columns."""
    request = AnalysisRequest(
        operation=AnalysisOperation.describe,
        columns=columns,
        limit=limit,
    )
    return json.dumps(analyze_dataframe(dataframe, request), ensure_ascii=False)


@tool
def get_chart_data(
    dataframe: Annotated[pd.DataFrame, InjectedState("dataframe")],
    query_spec: AnalysisRequest,
) -> str:
    """Run a validated analysis and return its table together with chart configurations."""
    if not query_spec.charts:
        raise ValueError("query_spec cần có ít nhất một cấu hình biểu đồ.")
    return json.dumps(analyze_dataframe(dataframe, query_spec), ensure_ascii=False)


DATA_ANALYSIS_TOOLS = [
    profile_dataset,
    list_columns,
    filter_dataset,
    aggregate_dataset,
    analyze_trend,
    calculate_statistics,
    get_chart_data,
]

CHART_DATA_TOOL = get_chart_data

TOOLS_BY_OPERATION = {
    AnalysisOperation.profile: profile_dataset,
    AnalysisOperation.list_columns: list_columns,
    AnalysisOperation.filter: filter_dataset,
    AnalysisOperation.aggregate: aggregate_dataset,
    AnalysisOperation.trend: analyze_trend,
    AnalysisOperation.describe: calculate_statistics,
}
