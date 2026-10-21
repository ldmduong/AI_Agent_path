from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalysisOperation(StrEnum):
    aggregate = "aggregate"
    trend = "trend"
    filter = "filter"
    describe = "describe"
    profile = "profile"
    list_columns = "list_columns"


class FilterOperator(StrEnum):
    eq = "eq"
    ne = "ne"
    gt = "gt"
    gte = "gte"
    lt = "lt"
    lte = "lte"
    contains = "contains"
    is_null = "is_null"
    not_null = "not_null"


class ChartType(StrEnum):
    bar = "bar"
    line = "line"
    scatter = "scatter"
    histogram = "histogram"
    area = "area"
    box = "box"
    violin = "violin"
    pie = "pie"


class TrendInterval(StrEnum):
    day = "day"
    week = "week"
    month = "month"
    quarter = "quarter"
    year = "year"


class FilterCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1, max_length=256)
    operator: FilterOperator
    value: Any = None

    @model_validator(mode="after")
    def require_value_when_needed(self):
        if self.operator not in {FilterOperator.is_null, FilterOperator.not_null} and self.value is None:
            raise ValueError("Điều kiện lọc này cần có value.")
        return self


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1, max_length=256)
    function: str = Field(pattern="^(sum|mean|median|min|max|count)$")


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ChartType
    x: str = Field(min_length=1, max_length=256)
    y: str | None = Field(default=None, max_length=256)
    color: str | None = Field(default=None, max_length=256)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: AnalysisOperation
    time_column: str | None = Field(default=None, min_length=1, max_length=256)
    interval: TrendInterval | None = None
    group_by: list[str] = Field(default_factory=list, max_length=5)
    metrics: list[MetricSpec] = Field(default_factory=list, max_length=10)
    filters: list[FilterCondition] = Field(default_factory=list, max_length=10)
    columns: list[str] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=500, ge=1, le=500)
    chart: ChartSpec | None = None
    charts: list[ChartSpec] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def normalize_charts(self):
        if self.charts:
            self.chart = self.charts[0]
        elif self.chart is not None:
            self.charts = [self.chart]
        chart_keys = [
            (chart.type, chart.x, chart.y, chart.color)
            for chart in self.charts
        ]
        if len(chart_keys) != len(set(chart_keys)):
            raise ValueError("Không được lặp cấu hình biểu đồ.")
        return self

    @model_validator(mode="after")
    def validate_operation_inputs(self):
        if self.operation == AnalysisOperation.aggregate and not self.metrics:
            raise ValueError("Phân tích tổng hợp cần ít nhất một metric.")
        if self.operation == AnalysisOperation.trend and (
            not self.time_column or not self.metrics or not self.interval
        ):
            raise ValueError("Phân tích xu hướng cần time_column, interval và ít nhất một metric.")
        if self.operation != AnalysisOperation.trend and (self.time_column or self.interval):
            raise ValueError("time_column và interval chỉ dùng cho phân tích xu hướng.")
        if self.operation == AnalysisOperation.filter and not self.filters:
            raise ValueError("Phân tích lọc cần ít nhất một điều kiện.")
        if self.operation == AnalysisOperation.filter and (self.metrics or self.group_by):
            raise ValueError("Phép lọc không nhận metrics hoặc group_by.")
        if self.operation == AnalysisOperation.describe and (self.metrics or self.group_by or self.filters):
            raise ValueError("Thống kê mô tả không nhận metrics, group_by hoặc filters.")
        if self.operation in {AnalysisOperation.profile, AnalysisOperation.list_columns} and (
            self.metrics or self.group_by or self.filters or self.columns or self.chart
        ):
            raise ValueError("Profile và danh sách cột không nhận tham số phân tích.")
        if self.operation not in {AnalysisOperation.filter, AnalysisOperation.describe} and self.columns:
            raise ValueError("Trường columns chỉ dùng cho phép lọc hoặc thống kê mô tả.")
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError("Không được lặp cột trong group_by.")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Không được lặp cột trong columns.")
        metric_keys = [(metric.column, metric.function) for metric in self.metrics]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("Không được lặp metric.")
        return self


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2_000)
    conversation_id: UUID | None = None


class DocumentAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2_000)
    document_ids: list[UUID] = Field(min_length=1, max_length=10)

    @field_validator("document_ids")
    @classmethod
    def require_unique_documents(cls, document_ids: list[UUID]) -> list[UUID]:
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Không được chọn lặp tài liệu.")
        return document_ids
