import json
import os
import re
from enum import StrEnum
from functools import lru_cache

import httpx
import pandas as pd
from google.genai import errors as google_genai_errors
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from ollama import ResponseError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIError as OpenAIAPIError
from pydantic import BaseModel, ConfigDict, ValidationError

from app.schemas import AnalysisOperation, AnalysisRequest, ChartType, TrendInterval


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


class Specialist(StrEnum):
    dataset = "dataset"
    filter = "filter"
    analytics = "analytics"
    visualization = "visualization"


class SpecialistRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specialist: Specialist


class ModelUnavailableError(Exception):
    pass


class ModelResponseError(Exception):
    pass


OllamaUnavailableError = ModelUnavailableError
OllamaResponseError = ModelResponseError


@lru_cache(maxsize=1)
def get_chat_model():
    options = {"base_url": OLLAMA_BASE_URL} if LLM_PROVIDER == "ollama" else {}
    return init_chat_model(
        model=LLM_MODEL,
        model_provider=LLM_PROVIDER,
        temperature=0,
        **options,
    )


SPECIALIST_OPERATIONS = {
    Specialist.dataset: {
        AnalysisOperation.profile,
        AnalysisOperation.list_columns,
        AnalysisOperation.describe,
    },
    Specialist.filter: {AnalysisOperation.filter},
    Specialist.analytics: {AnalysisOperation.aggregate, AnalysisOperation.trend},
    Specialist.visualization: {AnalysisOperation.aggregate, AnalysisOperation.trend},
}

FILTER_INTENT_SIGNALS = (
    "filter",
    "lọc",
    "chỉ lấy",
    "chỉ những",
)
_FILTER_VALUE_COMPARISON = re.compile(
    r"(?:lớn hơn|nhỏ hơn|cao hơn|thấp hơn|ít nhất|tối thiểu|tối đa|trên|dưới|"
    r"greater than|less than|higher than|lower than|more than|at least|at most|"
    r"above|below|over|under|>=|<=|>|<|=)\s*[-+]?\d+(?:[.,]\d+)?",
    re.IGNORECASE,
)
CHART_INTENT_SIGNALS = (
    "biểu đồ", "vẽ", "chart", "graph", "plot", "visualize", "visualise",
    "trực quan", "dashboard", "nhiều biểu đồ", "multiple charts", "multi chart", "histogram",
)


def _columns(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": str(column), "dtype": str(dataframe[column].dtype)}
        for column in dataframe.columns
    ]


def _has_filter_intent(question: str) -> bool:
    question_lower = question.casefold()
    return (
        any(signal in question_lower for signal in FILTER_INTENT_SIGNALS)
        or bool(_FILTER_VALUE_COMPARISON.search(question_lower))
    )


def route_question(
    question: str,
    dataframe: pd.DataFrame,
    history: list[dict] | None = None,
) -> Specialist:
    prompt = {
        "question": question,
        "columns": _columns(dataframe),
        "recent_conversation": (history or [])[-5:],
        "specialists": {
            "dataset": "Cấu trúc dataset, liệt kê cột, thống kê mô tả.",
            "filter": "Lọc các dòng theo điều kiện người dùng nêu rõ.",
            "analytics": "Tính tổng hợp, nhóm, xu hướng theo thời gian, đếm, trung bình hoặc so sánh số liệu.",
            "visualization": "Tạo dashboard một hoặc nhiều biểu đồ, gồm biểu đồ xu hướng khi phù hợp.",
        },
    }
    try:
        route = get_chat_model().with_structured_output(SpecialistRoute).invoke(
            [
                SystemMessage(
                    content="Định tuyến câu hỏi đến đúng một chuyên gia phù hợp nhất. "
                    "Chỉ dùng tên chuyên gia trong schema."
                ),
                HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
            ]
        )
        question_lower = question.casefold()
        if any(signal in question_lower for signal in CHART_INTENT_SIGNALS):
            return Specialist.visualization
        if _has_filter_intent(question_lower):
            return Specialist.filter
        if (
            route.specialist == Specialist.visualization
            and not any(signal in question_lower for signal in CHART_INTENT_SIGNALS)
        ):
            return Specialist.analytics
        return route.specialist
    except (OutputParserException, ValidationError, ValueError, TypeError) as exc:
        raise ModelResponseError("Model không chọn được chuyên gia hợp lệ.") from exc
    except (httpx.RequestError, ResponseError, OpenAIAPIConnectionError) as exc:
        raise ModelUnavailableError(
            f"Model {LLM_PROVIDER}/{LLM_MODEL} không khả dụng hoặc từ chối request."
        ) from exc
    except (OpenAIAPIError, google_genai_errors.APIError) as exc:
        raise ModelResponseError(
            f"Provider {LLM_PROVIDER} từ chối hoặc không xử lý được request model."
        ) from exc


def _validate_plan(
    payload: dict,
    question: str,
    specialist: Specialist,
    dataframe: pd.DataFrame | None = None,
) -> AnalysisRequest:
    question_lower = question.casefold()
    if dataframe is not None and specialist in {Specialist.analytics, Specialist.visualization} and any(
        signal in question_lower for signal in ("xu hướng", "xu the", "trend", "over time")
    ):
        time_candidate = payload.get("time_column")
        candidate_columns = [time_candidate] if time_candidate else payload.get("group_by", [])
        group_by = [
            column for column in candidate_columns
            if column in dataframe.columns
            and pd.to_datetime(dataframe[column], errors="coerce", format="mixed").notna().any()
        ]
        if not group_by:
            group_by = [
                str(column) for column in dataframe.columns
                if not pd.api.types.is_numeric_dtype(dataframe[column])
                and pd.to_datetime(dataframe[column], errors="coerce", format="mixed").notna().any()
            ]
        if len(group_by) != 1:
            raise ModelResponseError("Không xác định được một cột thời gian duy nhất để phân tích xu hướng.")
        interval_signals = {
            TrendInterval.day: ("theo ngày", "mỗi ngày", "daily"),
            TrendInterval.week: ("theo tuần", "mỗi tuần", "weekly"),
            TrendInterval.quarter: ("theo quý", "mỗi quý", "quarterly"),
            TrendInterval.year: ("theo năm", "mỗi năm", "yearly", "annually"),
            TrendInterval.month: ("theo tháng", "mỗi tháng", "monthly"),
        }
        interval = next(
            (
                value
                for value, signals in interval_signals.items()
                if any(signal in question_lower for signal in signals)
            ),
            TrendInterval.month,
        )
        payload.update(
            {
                "operation": AnalysisOperation.trend,
                "time_column": group_by[0],
                "interval": interval.value,
                "group_by": [],
            }
        )
    operation = AnalysisOperation(payload.get("operation"))
    if operation not in SPECIALIST_OPERATIONS[specialist]:
        raise ModelResponseError(
            f"Chuyên gia {specialist.value} trả về phép toán ngoài phạm vi."
        )
    if operation != AnalysisOperation.filter:
        payload["columns"] = []
    else:
        payload["group_by"] = []
        payload["metrics"] = []
    if operation == AnalysisOperation.describe:
        payload["group_by"] = []
    if operation in {AnalysisOperation.profile, AnalysisOperation.list_columns}:
        payload["group_by"] = []
        payload["metrics"] = []
        payload["filters"] = []
        payload["columns"] = []
        payload["chart"] = None
        payload["charts"] = []
    if (
        specialist != Specialist.filter
        and not _has_filter_intent(question_lower)
    ):
        payload["filters"] = []
    if not any(signal in question_lower for signal in CHART_INTENT_SIGNALS):
        payload["chart"] = None
        payload["charts"] = []
    payload["limit"] = 500
    if operation == AnalysisOperation.filter and not payload.get("filters"):
        raise ModelResponseError("Chưa nhận diện được điều kiện lọc. Hãy nêu rõ cột và điều kiện.")
    if specialist == Specialist.visualization and not payload.get("chart") and not payload.get("charts"):
        raise ModelResponseError("Chuyên gia biểu đồ chưa tạo được cấu hình biểu đồ.")
    plan = AnalysisRequest.model_validate(payload)
    if specialist == Specialist.visualization and plan.charts:
        metric_columns = {
            f"{metric.function}_{metric.column}" for metric in plan.metrics
        }
        normalized_charts = []
        for chart in plan.charts:
            if len(plan.metrics) == 1 and chart.y == plan.metrics[0].column:
                metric = plan.metrics[0]
                chart = chart.model_copy(update={"y": f"{metric.function}_{metric.column}"})
            if chart.type == ChartType.histogram:
                valid_axes = chart.x in metric_columns and chart.y is None
            elif chart.type == ChartType.scatter:
                valid_axes = chart.x in metric_columns and chart.y in metric_columns
            else:
                valid_x = (
                    chart.x == plan.time_column
                    if plan.operation == AnalysisOperation.trend
                    else chart.x in plan.group_by
                )
                valid_axes = valid_x and chart.y in metric_columns
            if not valid_axes:
                raise ModelResponseError(
                    "Chuyên gia biểu đồ phải dùng cột thời gian/nhóm và metric tổng hợp "
                    "đúng với loại biểu đồ."
                )
            normalized_charts.append(chart)
        plan = plan.model_copy(
            update={"charts": normalized_charts, "chart": normalized_charts[0]}
        )
    return plan


def create_analysis_plan(
    question: str,
    dataframe: pd.DataFrame,
    specialist: Specialist,
    history: list[dict] | None = None,
) -> AnalysisRequest:
    allowed_operations = sorted(
        operation.value for operation in SPECIALIST_OPERATIONS[specialist]
    )
    examples = []
    if specialist in {Specialist.analytics, Specialist.visualization}:
        examples.append({
            "question": "Tính tổng doanh thu theo tháng",
            "columns": ["month", "revenue"],
            "plan": {
                "operation": "aggregate",
                "group_by": ["month"],
                "metrics": [{"column": "revenue", "function": "sum"}],
                "filters": [],
                "columns": [],
                "limit": 500,
                "chart": None,
                "charts": [],
            },
        })
    if specialist == Specialist.analytics:
        examples.append({
            "question": "Xu hướng doanh thu theo tháng trong năm",
            "columns": ["date", "revenue"],
            "plan": {
                "operation": "trend",
                "time_column": "date",
                "interval": "month",
                "metrics": [{"column": "revenue", "function": "sum"}],
                "filters": [],
                "columns": [],
                "limit": 500,
                "chart": None,
                "charts": [],
            },
        })
    if specialist == Specialist.visualization:
        examples.append({
            "question": "Vẽ biểu đồ xu hướng doanh thu theo tháng",
            "columns": ["date", "revenue"],
            "plan": {
                "operation": "trend",
                "time_column": "date",
                "interval": "month",
                "metrics": [{"column": "revenue", "function": "sum"}],
                "filters": [],
                "columns": [],
                "limit": 500,
                "chart": None,
                "charts": [{"type": "line", "x": "date", "y": "sum_revenue"}],
            },
        })
    if specialist == Specialist.visualization:
        examples.append({
            "question": "Tạo dashboard xu hướng doanh thu và số đơn theo tháng",
            "columns": ["date", "revenue", "order_id"],
            "plan": {
                "operation": "trend",
                "time_column": "date",
                "interval": "month",
                "metrics": [
                    {"column": "revenue", "function": "sum"},
                    {"column": "order_id", "function": "count"},
                ],
                "filters": [],
                "columns": [],
                "limit": 500,
                "chart": None,
                "charts": [
                    {"type": "line", "x": "date", "y": "sum_revenue"},
                    {"type": "bar", "x": "date", "y": "count_order_id"},
                ],
            },
        })
        examples.append({
            "question": "Create a bar chart of total revenue by month",
            "columns": ["month", "revenue"],
            "plan": {
                "operation": "aggregate",
                "group_by": ["month"],
                "metrics": [{"column": "revenue", "function": "sum"}],
                "filters": [],
                "columns": [],
                "limit": 500,
                "chart": None,
                "charts": [{"type": "bar", "x": "month", "y": "sum_revenue"}],
            },
        })
    prompt = {
        "question": question,
        "columns": _columns(dataframe),
        "recent_conversation": (history or [])[-5:],
        "allowed_operations": allowed_operations,
        "rules": [
            "Chỉ dùng chính xác tên cột được liệt kê; không tự tạo cột hoặc dữ liệu.",
            "Dùng cấu trúc AnalysisRequest; điền operation và tham số cần thiết.",
            "Khi câu hỏi nói 'theo/by' một chiều và có cột tương ứng, bắt buộc điền cột đó vào group_by.",
            "Với aggregate, metrics dùng sum, mean, median, min, max hoặc count.",
            "Với trend, chọn cột thời gian, interval day/week/month/quarter/year và metric cần tính.",
            "Với filter, filters dùng eq, ne, gt, gte, lt, lte, contains, is_null hoặc not_null.",
            "Chỉ đặt filters hoặc chart khi câu hỏi yêu cầu rõ.",
            "Với visualization, dùng charts (tối đa 6 cấu hình); dashboard/nhiều biểu đồ nên có các biểu đồ bổ trợ trên cùng bảng kết quả.",
            "Chart phải dùng cột nhóm/thời gian và metric tổng hợp; histogram dùng metric số ở trục x, scatter dùng hai metric số.",
            "Không viết SQL hoặc Python.",
        ],
        "examples": examples,
    }
    system_prompt = (
        f"Bạn là chuyên gia {specialist.value} cho ứng dụng phân tích dữ liệu. "
        f"Chỉ được chọn một trong các phép toán: {', '.join(allowed_operations)}. "
        "Không đoán cột hoặc điều kiện người dùng chưa nêu."
    )
    try:
        response = get_chat_model().with_structured_output(AnalysisRequest).invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
            ]
        )
        payload = response.model_dump(mode="python")
        return _validate_plan(payload, question, specialist, dataframe)
    except (OutputParserException, ValidationError, ValueError, TypeError) as exc:
        raise ModelResponseError(
            f"Model trả về kế hoạch không hợp lệ cho chuyên gia {specialist.value}."
        ) from exc
    except (httpx.RequestError, ResponseError, OpenAIAPIConnectionError) as exc:
        raise ModelUnavailableError(
            f"Model {LLM_PROVIDER}/{LLM_MODEL} không khả dụng hoặc từ chối request."
        ) from exc
    except (OpenAIAPIError, google_genai_errors.APIError) as exc:
        raise ModelResponseError(
            f"Provider {LLM_PROVIDER} từ chối hoặc không xử lý được request model."
        ) from exc
