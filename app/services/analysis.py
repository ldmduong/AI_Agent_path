import json

import pandas as pd

from app.schemas import AnalysisOperation, AnalysisRequest, FilterOperator


TREND_FREQUENCIES = {
    "day": "D",
    "week": "W",
    "month": "MS",
    "quarter": "QS",
    "year": "YS",
}


def _trend_analysis(dataframe: pd.DataFrame, request: AnalysisRequest) -> tuple[pd.DataFrame, list[dict]]:
    if request.time_column not in dataframe.columns:
        raise ValueError(f"Không tìm thấy cột thời gian '{request.time_column}'.")
    if pd.api.types.is_numeric_dtype(dataframe[request.time_column]):
        raise ValueError("Cột thời gian phải chứa ngày/giờ dạng văn bản hoặc datetime, không phải số.")
    missing = [metric.column for metric in request.metrics if metric.column not in dataframe.columns]
    if missing:
        raise ValueError(f"Không tìm thấy cột: {', '.join(dict.fromkeys(missing))}.")

    source = _filter_dataframe(dataframe, request) if request.filters else dataframe
    time_values = pd.to_datetime(
        source[request.time_column],
        errors="coerce",
        format="mixed",
    )
    valid = source.loc[time_values.notna()].copy()
    if valid.empty:
        raise ValueError(f"Cột '{request.time_column}' không có giá trị thời gian hợp lệ.")
    valid["__trend_time"] = time_values.loc[time_values.notna()]

    aggregations = {}
    for index, metric in enumerate(request.metrics):
        if metric.function != "count" and not pd.api.types.is_numeric_dtype(valid[metric.column]):
            raise ValueError(f"Phép tính {metric.function} yêu cầu cột '{metric.column}' là kiểu số.")
        aggregate_function = metric.function
        if metric.function == "sum":
            aggregate_function = lambda series: series.sum(min_count=1)
        aggregations[f"metric_{index}"] = (metric.column, aggregate_function)

    output_columns = {
        "__trend_time": request.time_column,
        **{
            f"metric_{index}": f"{metric.function}_{metric.column}"
            for index, metric in enumerate(request.metrics)
        },
    }
    trend = (
        valid.set_index("__trend_time")
        .resample(TREND_FREQUENCIES[request.interval])
        .agg(**aggregations)
        .reset_index()
        .rename(columns=output_columns)
    )
    trend = trend.dropna(subset=[f"{metric.function}_{metric.column}" for metric in request.metrics], how="all")
    insights = []
    for metric in request.metrics:
        column = f"{metric.function}_{metric.column}"
        values = trend[column].dropna()
        if values.empty:
            raise ValueError(f"Không có giá trị hợp lệ cho chỉ số '{metric.column}'.")
        first, last = values.iloc[0].item(), values.iloc[-1].item()
        change = last - first
        insights.append(
            {
                "metric": column,
                "direction": "increasing" if change > 0 else "decreasing" if change < 0 else "stable",
                "first_value": first,
                "last_value": last,
                "change": change,
                "change_percent": (change / first * 100) if first != 0 else None,
                "periods": len(values),
            }
        )
    return trend, insights


def _filter_dataframe(dataframe: pd.DataFrame, request: AnalysisRequest) -> pd.DataFrame:
    result = dataframe
    for condition in request.filters:
        if condition.column not in result.columns:
            raise ValueError(f"Không tìm thấy cột '{condition.column}'.")

        series = result[condition.column]
        operator = condition.operator
        if operator == FilterOperator.is_null:
            mask = series.isna()
        elif operator == FilterOperator.not_null:
            mask = series.notna()
        elif operator == FilterOperator.contains:
            mask = series.astype("string").str.contains(str(condition.value), case=False, regex=False, na=False)
        elif pd.api.types.is_numeric_dtype(series):
            try:
                value = float(condition.value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Giá trị lọc cho cột '{condition.column}' phải là số.") from exc
            mask = _compare(series, operator, value)
        elif operator in {FilterOperator.eq, FilterOperator.ne}:
            values = series.astype("string")
            mask = values.eq(str(condition.value)) if operator == FilterOperator.eq else values.ne(str(condition.value))
            mask = mask.fillna(False)
        else:
            raise ValueError(f"Toán tử '{operator.value}' chỉ hỗ trợ cột số hoặc contains.")
        result = result.loc[mask]
    return result


def _compare(series: pd.Series, operator: FilterOperator, value: float) -> pd.Series:
    comparisons = {
        FilterOperator.eq: series.eq,
        FilterOperator.ne: series.ne,
        FilterOperator.gt: series.gt,
        FilterOperator.gte: series.ge,
        FilterOperator.lt: series.lt,
        FilterOperator.lte: series.le,
    }
    compare = comparisons.get(operator)
    if compare is None:
        raise ValueError(f"Toán tử '{operator.value}' không hợp lệ cho so sánh số.")
    return compare(value).fillna(False)


def _records(dataframe: pd.DataFrame, limit: int) -> dict:
    visible = dataframe.head(limit)
    return {
        "columns": [str(column) for column in visible.columns],
        "rows": json.loads(visible.to_json(orient="records", date_format="iso", force_ascii=False)),
        "total_rows": int(len(dataframe)),
        "truncated": len(dataframe) > limit,
    }


def _validate_chart(result: pd.DataFrame, chart: dict) -> None:
    result_columns = [str(column) for column in result.columns]
    if (
        chart["x"] not in result_columns
        or (chart.get("y") and chart["y"] not in result_columns)
        or (chart.get("color") and chart["color"] not in result_columns)
    ):
        raise ValueError("Trục biểu đồ phải là cột có trong bảng kết quả.")
    if chart["type"] != "histogram" and not chart.get("y"):
        raise ValueError("Biểu đồ này cần chọn cả trục X và trục Y.")
    if chart["type"] == "scatter" and (
        not pd.api.types.is_numeric_dtype(result[chart["x"]])
        or not pd.api.types.is_numeric_dtype(result[chart["y"]])
    ):
        raise ValueError("Biểu đồ scatter cần hai trục kiểu số.")
    if chart["type"] in {"pie", "box", "violin"} and not pd.api.types.is_numeric_dtype(result[chart["y"]]):
        raise ValueError(f"Biểu đồ {chart['type']} cần trục Y kiểu số.")
    if chart["type"] == "histogram" and not pd.api.types.is_numeric_dtype(result[chart["x"]]):
        raise ValueError("Biểu đồ histogram cần trục X kiểu số.")


def analyze_dataframe(dataframe: pd.DataFrame, request: AnalysisRequest) -> dict:
    if request.operation == AnalysisOperation.filter:
        filtered = _filter_dataframe(dataframe, request)
        if request.columns:
            missing = [column for column in request.columns if column not in filtered.columns]
            if missing:
                raise ValueError(f"Không tìm thấy cột: {', '.join(missing)}.")
            filtered = filtered[request.columns]
        result = filtered
        answer = f"Tìm thấy {len(result):,} dòng phù hợp."
    elif request.operation == AnalysisOperation.aggregate:
        source = _filter_dataframe(dataframe, request) if request.filters else dataframe
        missing_columns = [
            column for column in request.group_by + [metric.column for metric in request.metrics]
            if column not in source.columns
        ]
        if missing_columns:
            raise ValueError(f"Không tìm thấy cột: {', '.join(dict.fromkeys(missing_columns))}.")
        output_columns = request.group_by + [
            f"{metric.function}_{metric.column}" for metric in request.metrics
        ]
        if len(output_columns) != len(set(output_columns)):
            raise ValueError("Tên cột nhóm bị trùng với tên cột kết quả tổng hợp.")
        aggregations = {}
        for index, metric in enumerate(request.metrics):
            if metric.function != "count" and not pd.api.types.is_numeric_dtype(source[metric.column]):
                raise ValueError(f"Phép tính {metric.function} yêu cầu cột '{metric.column}' là kiểu số.")
            aggregations[f"{metric.function}_{index}"] = (metric.column, metric.function)

        if request.group_by:
            result = (
                source.groupby(request.group_by, dropna=False, observed=True)
                .agg(**aggregations)
                .reset_index()
            )
        else:
            result = pd.DataFrame(
                [{
                    alias: source[column].agg(function)
                    for alias, (column, function) in aggregations.items()
                }]
            )
        result = result.rename(
            columns={
                f"{metric.function}_{index}": f"{metric.function}_{metric.column}"
                for index, metric in enumerate(request.metrics)
            }
        )
        answer = f"Đã tổng hợp dữ liệu thành {len(result):,} nhóm."
    elif request.operation == AnalysisOperation.trend:
        result, insights = _trend_analysis(dataframe, request)
        directions = {
            "increasing": "tăng",
            "decreasing": "giảm",
            "stable": "ổn định",
        }
        answer = "Xu hướng: " + "; ".join(
            f"{insight['metric']} {directions[insight['direction']]}"
            for insight in insights
        ) + "."
    elif request.operation == AnalysisOperation.describe:
        if request.columns:
            missing = [column for column in request.columns if column not in dataframe.columns]
            if missing:
                raise ValueError(f"Không tìm thấy cột: {', '.join(missing)}.")
            selected = dataframe[request.columns]
        else:
            selected = dataframe.select_dtypes(include="number")
        non_numeric = [
            column for column in selected.columns
            if not pd.api.types.is_numeric_dtype(selected[column])
        ]
        if non_numeric:
            raise ValueError(f"Thống kê mô tả chỉ hỗ trợ cột số: {', '.join(non_numeric)}.")
        numeric_columns = selected.select_dtypes(include="number").columns
        if len(numeric_columns) == 0:
            raise ValueError("Dataset không có cột số để thống kê.")
        result = selected[numeric_columns].describe().T.reset_index(names="column")
        answer = f"Thống kê mô tả cho {len(numeric_columns)} cột số."
    elif request.operation == AnalysisOperation.profile:
        rows = [
            {
                "name": str(column),
                "dtype": str(dataframe[column].dtype),
                "missing_count": int(dataframe[column].isna().sum()),
            }
            for column in dataframe.columns
        ]
        return {
            "answer": f"Dataset có {len(dataframe):,} dòng và {len(dataframe.columns)} cột.",
            "profile": {
                "row_count": int(len(dataframe)),
                "column_count": int(len(dataframe.columns)),
                "columns": rows,
            },
            "table": {
                "columns": ["name", "dtype", "missing_count"],
                "rows": rows,
                "total_rows": len(rows),
                "truncated": False,
            },
            "chart": None,
            "charts": [],
        }
    elif request.operation == AnalysisOperation.list_columns:
        rows = [
            {"name": str(column), "dtype": str(dataframe[column].dtype)}
            for column in dataframe.columns
        ]
        return {
            "answer": f"Dataset có {len(rows)} cột.",
            "table": {
                "columns": ["name", "dtype"],
                "rows": rows,
                "total_rows": len(rows),
                "truncated": False,
            },
            "chart": None,
            "charts": [],
        }
    else:
        raise ValueError(f"Phép phân tích '{request.operation}' chưa được hỗ trợ.")

    charts = [chart.model_dump(mode="json", exclude_none=True) for chart in request.charts]
    for chart in charts:
        _validate_chart(result, chart)

    response = {
        "answer": answer,
        "table": _records(result, request.limit),
        "chart": charts[0] if charts else None,
        "charts": charts,
    }
    if request.operation == AnalysisOperation.trend:
        response["trend"] = insights
    return response
