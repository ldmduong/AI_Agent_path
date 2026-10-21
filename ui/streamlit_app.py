import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.getenv("AI_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="AI Data Agent", page_icon="📊", layout="wide")
st.title("AI Data Agent")
st.caption("LangGraph định tuyến đến chuyên gia phù hợp; các phép tính được kiểm tra và chạy bằng pandas.")


def api_request(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=180, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        st.error(f"API trả lỗi {exc.response.status_code}: {detail}")
    except httpx.RequestError as exc:
        st.error(f"Không kết nối được API tại {API_URL}: {exc}")
    return None


def refresh_dataset_view(dataset_id: str, sheet: str | None) -> bool:
    previous_sheet = st.session_state.get("sheet")
    params = {"sheet": sheet} if sheet else {}
    profile = api_request("GET", f"/datasets/{dataset_id}/profile", params=params)
    preview = api_request(
        "GET",
        f"/datasets/{dataset_id}/preview",
        params={**params, "limit": 100},
    )
    if profile is None or preview is None:
        return False
    if (
        st.session_state.get("dataset_id") == dataset_id
        and previous_sheet != sheet
    ):
        st.session_state.conversation_id = str(uuid4())
        st.session_state.analysis_job_id = None
    st.session_state.profile = profile
    st.session_state.preview = preview
    st.session_state.sheet = sheet
    st.session_state.analysis = None
    st.session_state.analysis_request = None
    return True


if api_request("GET", "/health") is None:
    st.info("Hãy khởi động FastAPI trước; xem README.md để biết lệnh chạy.")
    st.stop()

st.subheader("Hỏi đáp tài liệu")
if "documents" not in st.session_state:
    stored_documents = api_request("GET", "/documents")
    if stored_documents is not None:
        st.session_state.documents = stored_documents["documents"]
document_files = st.file_uploader(
    "Tải tài liệu TXT hoặc PDF",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key="document_files",
)
if document_files and st.button("Nạp tài liệu", key="upload_documents"):
    documents = st.session_state.get("documents", [])
    for document_file in document_files:
        uploaded_document = api_request(
            "POST",
            "/documents",
            files={
                "file": (
                    document_file.name,
                    document_file.getvalue(),
                    document_file.type,
                )
            },
        )
        if uploaded_document:
            documents.append(uploaded_document)
    st.session_state.documents = documents

documents = st.session_state.get("documents", [])
for document in documents:
    left, right = st.columns([5, 1])
    left.caption(
        f"{document['filename']} · {document['chunk_count']} đoạn · "
        f"ID {document['document_id'][:8]}"
    )
    if right.button("Xóa", key=f"delete_document_{document['document_id']}"):
        deleted = api_request("DELETE", f"/documents/{document['document_id']}")
        if deleted is not None:
            st.session_state.documents = [
                item for item in documents
                if item["document_id"] != document["document_id"]
            ]
            st.session_state.pop("document_answer", None)
            st.rerun()

if documents:
    with st.form("document_question_form"):
        selected_document_ids = st.multiselect(
            "Tài liệu cần tra cứu",
            [document["document_id"] for document in documents],
            format_func=lambda document_id: next(
                item["filename"] for item in documents
                if item["document_id"] == document_id
            ),
        )
        document_question = st.text_input(
            "Câu hỏi về tài liệu",
            placeholder="Ví dụ: Chính sách nghỉ phép quy định thế nào?",
            max_chars=2_000,
        )
        ask_documents = st.form_submit_button("Hỏi tài liệu", type="primary")
    if ask_documents:
        st.session_state.pop("document_answer", None)
        if len(document_question.strip()) < 3:
            st.error("Nhập câu hỏi dài ít nhất 3 ký tự.")
        elif not selected_document_ids:
            st.error("Chọn ít nhất một tài liệu.")
        else:
            answer = api_request(
                "POST",
                "/documents/ask",
                json={
                    "question": document_question,
                    "document_ids": selected_document_ids,
                },
            )
            if answer:
                st.session_state.document_answer = answer

document_answer = st.session_state.get("document_answer")
if document_answer:
    st.success(document_answer["answer"])
    st.caption(
        f"Model: {document_answer['model_provider']}/{document_answer['model']}"
    )
    for source in document_answer["sources"]:
        with st.expander(
            f"[{source['id']}] {source['filename']} · trang {source['page']} · "
            f"đoạn {source['chunk']}"
        ):
            st.write(source["excerpt"])

st.divider()

uploaded_file = st.file_uploader(
    "Tải CSV hoặc Excel (.xlsx)",
    type=["csv", "xlsx"],
    max_upload_size=25,
)
if uploaded_file and st.button("Tải file lên", type="primary"):
    uploaded = api_request(
        "POST",
        "/files",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
    )
    if uploaded:
        st.session_state.dataset_id = uploaded["dataset_id"]
        st.session_state.filename = uploaded["filename"]
        st.session_state.conversation_id = str(uuid4())
        st.session_state.analysis_job_id = None
        st.session_state.profile = uploaded["profile"]
        st.session_state.sheet = uploaded["profile"]["active_sheet"]
        st.session_state.analysis = None
        st.session_state.analysis_request = None
        if not refresh_dataset_view(uploaded["dataset_id"], uploaded["profile"]["active_sheet"]):
            st.stop()

dataset_id = st.session_state.get("dataset_id")
if not dataset_id:
    st.info("Tải một file lên để bắt đầu.")
    st.stop()
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())

st.subheader(f"Dataset: {st.session_state.get('filename', dataset_id)}")
profile = st.session_state.profile
sheets = profile.get("sheets", [])
if len(sheets) > 1:
    selected_sheet = st.selectbox(
        "Sheet",
        sheets,
        index=sheets.index(st.session_state.sheet) if st.session_state.sheet in sheets else 0,
    )
    if selected_sheet != st.session_state.sheet:
        if refresh_dataset_view(dataset_id, selected_sheet):
            st.rerun()

profile = st.session_state.profile
metric1, metric2 = st.columns(2)
metric1.metric("Số dòng", f"{profile['row_count']:,}")
metric2.metric("Số cột", profile["column_count"])
st.dataframe(pd.DataFrame(profile["columns"]), hide_index=True, width="stretch")

preview = st.session_state.get("preview")
if preview:
    with st.expander("Xem trước tối đa 100 dòng", expanded=True):
        st.dataframe(pd.DataFrame(preview["rows"]), hide_index=True, width="stretch")

st.subheader("Hỏi dữ liệu bằng AI")
if st.button("Xóa lịch sử hội thoại"):
    deleted = api_request(
        "DELETE",
        f"/datasets/{dataset_id}/conversations/{st.session_state.conversation_id}",
    )
    if deleted is not None:
        st.session_state.conversation_id = str(uuid4())
        st.session_state.analysis = None
        st.session_state.analysis_request = None
        st.session_state.analysis_job_id = None
        st.rerun()

with st.form("ask_form"):
    question = st.text_input(
        "Bạn muốn phân tích gì?",
        placeholder="Ví dụ: Tính tổng doanh thu theo tháng",
        max_chars=2_000,
    )
    ask_submit = st.form_submit_button("Hỏi AI", type="primary")

if ask_submit:
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_request", None)
    if len(question.strip()) < 3:
        st.error("Nhập câu hỏi dài ít nhất 3 ký tự.")
    else:
        params = {"sheet": st.session_state.sheet} if st.session_state.sheet else {}
        job = api_request(
            "POST",
            f"/datasets/{dataset_id}/ask/jobs",
            params=params,
            json={
                "question": question,
                "conversation_id": st.session_state.conversation_id,
            },
        )
        if job:
            st.session_state.analysis_job_id = job["job_id"]
            st.session_state.conversation_id = job["conversation_id"]


@st.fragment(run_every="2s")
def poll_analysis_job():
    job_id = st.session_state.get("analysis_job_id")
    if not job_id:
        return
    job = api_request("GET", f"/analysis-jobs/{job_id}")
    if job is None:
        st.session_state.analysis_job_id = None
    elif job["status"] in {"queued", "running"}:
        st.info("Đang xử lý câu hỏi...")
    elif job["status"] == "succeeded":
        st.session_state.analysis = job["result"]
        st.session_state.analysis_request = job["result"]["plan"]
        st.session_state.conversation_id = job["conversation_id"]
        st.session_state.analysis_job_id = None
        st.rerun()
    else:
        st.error(f"Không thể hoàn thành tác vụ: {job['error']}")
        st.session_state.analysis_job_id = None


poll_analysis_job()

ai_result = st.session_state.get("analysis")
if ai_result and ai_result.get("question"):
    st.success(ai_result.get("answer") or "AI đã xử lý xong nhưng không trả về nội dung câu trả lời.")

st.subheader("Phân tích")
all_columns = [column["name"] for column in profile["columns"]]
numeric_columns = [
    column["name"] for column in profile["columns"]
    if pd.api.types.is_numeric_dtype(column["dtype"])
]

with st.form("analysis_form"):
    operation = st.selectbox(
        "Phép phân tích",
        ["aggregate", "trend", "filter", "describe"],
        format_func={
            "aggregate": "Tổng hợp theo nhóm",
            "trend": "Xu hướng theo thời gian",
            "filter": "Lọc dòng",
            "describe": "Thống kê mô tả cột số",
        }.get,
    )
    group_by = []
    metrics = []
    filters = []
    selected_columns = []

    if operation == "aggregate":
        group_by = st.multiselect("Nhóm theo cột", all_columns)
        function = st.selectbox("Phép tính", ["sum", "mean", "median", "min", "max", "count"])
        metric_options = all_columns if function == "count" else numeric_columns
        metric_columns = st.multiselect("Cột cần tính", metric_options)
        metrics = [{"column": column, "function": function} for column in metric_columns]
    elif operation == "trend":
        time_column = st.selectbox("Cột thời gian", all_columns)
        interval = st.selectbox(
            "Gom theo kỳ",
            ["day", "week", "month", "quarter", "year"],
            index=2,
            format_func={
                "day": "Ngày",
                "week": "Tuần",
                "month": "Tháng",
                "quarter": "Quý",
                "year": "Năm",
            }.get,
        )
        function = st.selectbox("Phép tính mỗi kỳ", ["sum", "mean", "median", "min", "max", "count"])
        metric_options = all_columns if function == "count" else numeric_columns
        metric_columns = st.multiselect("Chỉ số theo dõi", metric_options)
        metrics = [{"column": column, "function": function} for column in metric_columns]
    elif operation == "filter":
        filter_column = st.selectbox("Cột lọc", all_columns)
        filter_is_numeric = filter_column in numeric_columns
        operators = ["eq", "ne", "gt", "gte", "lt", "lte", "contains", "is_null", "not_null"]
        if not filter_is_numeric:
            operators = ["eq", "ne", "contains", "is_null", "not_null"]
        filter_operator = st.selectbox("Điều kiện", operators)
        filter_value = None
        if filter_operator not in {"is_null", "not_null"}:
            filter_value = st.text_input("Giá trị cần so sánh")
        selected_columns = st.multiselect("Cột hiển thị (để trống = tất cả)", all_columns)
        filters = [{"column": filter_column, "operator": filter_operator}]
        if filter_value is not None:
            filters[0]["value"] = filter_value
    submit = st.form_submit_button("Chạy phân tích", type="primary")

if submit:
    if st.session_state.get("analysis_job_id"):
        st.session_state.analysis_job_id = None
        st.session_state.conversation_id = str(uuid4())
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_request", None)
    payload = {
        "operation": operation,
        "group_by": group_by,
        "metrics": metrics,
        "filters": filters,
        "columns": selected_columns,
        "limit": 500,
    }
    if operation == "trend":
        payload["time_column"] = time_column
        payload["interval"] = interval
    if operation in {"aggregate", "trend"} and not metrics:
        st.error("Chọn ít nhất một cột cần tính/theo dõi.")
    else:
        params = {"sheet": st.session_state.sheet} if st.session_state.sheet else {}
        result = api_request(
            "POST",
            f"/datasets/{dataset_id}/analyze",
            params=params,
            json=payload,
        )
        if result:
            st.session_state.analysis = result
            st.session_state.analysis_request = payload

result = st.session_state.get("analysis")
if result:
    if not result.get("question"):
        st.success(result["answer"])
    if result.get("question"):
        st.caption(
            f"Câu hỏi: {result['question']} · Chuyên gia: {result.get('specialist', '—')} · "
            f"Tool: {result['tool']} · Model: "
            f"{result.get('model_provider', '')}/{result['model']}"
        )
        with st.expander("Kế hoạch chuyên gia chuyển thành tool"):
            st.json(result["plan"])
    table = result["table"]
    for insight in result.get("trend", []):
        change = insight["change_percent"]
        change_label = "không xác định (giá trị đầu bằng 0)" if change is None else f"{change:+.1f}%"
        st.metric(
            f"Xu hướng · {insight['metric']}",
            insight["direction"],
            delta=change_label,
        )
    result_frame = pd.DataFrame(table["rows"], columns=table["columns"])
    truncation_note = " — đã giới hạn số dòng hiển thị." if table["truncated"] else ""
    st.caption(f"{table['total_rows']:,} dòng kết quả{truncation_note}")
    st.dataframe(result_frame, hide_index=True, width="stretch")

    if not result_frame.empty:
        existing_charts = result.get("charts") or ([result["chart"]] if result.get("chart") else [])
        chart_types = ["bar", "line", "scatter", "histogram", "area", "box", "violin", "pie"]
        chart_labels = {
            "bar": "Cột",
            "line": "Đường",
            "scatter": "Phân tán",
            "histogram": "Histogram",
            "area": "Miền",
            "box": "Hộp (box)",
            "violin": "Violin",
            "pie": "Tròn",
        }
        numeric_result_columns = [
            column for column in result_frame.columns
            if pd.api.types.is_numeric_dtype(result_frame[column])
        ]
        chart_context_data = json.dumps(
            {
                "dataset_id": dataset_id,
                "sheet": st.session_state.sheet,
                "question": result.get("question"),
                "request": st.session_state.analysis_request,
                "columns": table["columns"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        chart_context = str(abs(hash(chart_context_data)))
        default_chart_count = min(max(len(existing_charts), 1), 6)
        chart_count = st.slider(
            "Số biểu đồ muốn hiển thị",
            min_value=1,
            max_value=6,
            value=default_chart_count,
            key=f"chart_count_{chart_context}",
        )
        chart_requests = []
        chart_config_valid = True
        for chart_index in range(chart_count):
            existing_chart = existing_charts[chart_index] if chart_index < len(existing_charts) else {}
            st.markdown(f"**Cấu hình biểu đồ {chart_index + 1}**")
            chart_type = st.selectbox(
                "Loại biểu đồ",
                chart_types,
                index=chart_types.index(existing_chart["type"])
                if existing_chart.get("type") in chart_types
                else (chart_types.index("line") if st.session_state.analysis_request.get("operation") == "trend" else 0),
                format_func=chart_labels.get,
                key=f"chart_type_{chart_context}_{chart_index}",
            )
            x_options = numeric_result_columns if chart_type in {"histogram", "scatter"} else table["columns"]
            if not x_options:
                st.info(f"{chart_labels[chart_type]} cần ít nhất một cột số.")
                chart_config_valid = False
                continue
            default_x = existing_chart.get("x")
            if default_x not in x_options:
                trend_column = st.session_state.analysis_request.get("time_column")
                default_x = trend_column if trend_column in x_options else x_options[0]
            x_column, y_column, color_column = st.columns(3)
            with x_column:
                chart_x = st.selectbox(
                    "Trục X / nhóm",
                    x_options,
                    index=x_options.index(default_x),
                    key=f"chart_x_{chart_context}_{chart_index}_{chart_type}",
                )
            chart_y = None
            if chart_type != "histogram":
                y_options = [column for column in numeric_result_columns if column != chart_x]
                if not y_options:
                    st.info(f"{chart_labels[chart_type]} cần một cột số khác làm trục Y.")
                    chart_config_valid = False
                    continue
                default_y = existing_chart.get("y")
                if default_y not in y_options:
                    default_y = y_options[0]
                with y_column:
                    chart_y = st.selectbox(
                        "Trục Y",
                        y_options,
                        index=y_options.index(default_y),
                        key=f"chart_y_{chart_context}_{chart_index}_{chart_type}_{chart_x}",
                    )
            color_options = ["Không dùng", *[
                column for column in table["columns"]
                if column not in {chart_x, chart_y}
            ]]
            default_color = existing_chart.get("color") or "Không dùng"
            if default_color not in color_options:
                default_color = "Không dùng"
            with color_column:
                chart_color = st.selectbox(
                    "Tô màu theo cột",
                    color_options,
                    index=color_options.index(default_color),
                    key=f"chart_color_{chart_context}_{chart_index}_{chart_type}_{chart_x}_{chart_y}",
                )
            chart_requests.append({
                "type": chart_type,
                "x": chart_x,
                "y": chart_y,
                "color": None if chart_color == "Không dùng" else chart_color,
            })

        if st.button("Vẽ các biểu đồ", key=f"draw_charts_{chart_context}"):
            if chart_config_valid and len(chart_requests) == chart_count:
                chart_signatures = [
                    (chart["type"], chart["x"], chart["y"], chart["color"])
                    for chart in chart_requests
                ]
                if len(chart_signatures) != len(set(chart_signatures)):
                    st.error("Có biểu đồ bị trùng cấu hình; hãy đổi loại, trục hoặc cột màu.")
                else:
                    chart_request = dict(st.session_state.analysis_request)
                    chart_request["chart"] = None
                    chart_request["charts"] = chart_requests
                    params = {"sheet": st.session_state.sheet} if st.session_state.sheet else {}
                    chart_result = api_request(
                        "POST",
                        f"/datasets/{dataset_id}/analyze",
                        params=params,
                        json=chart_request,
                    )
                    if chart_result:
                        st.session_state.analysis = chart_result
                        result = chart_result
            else:
                st.error("Hoàn tất cấu hình hợp lệ cho từng biểu đồ trước khi vẽ.")

        chart_specs = result.get("charts") or ([result["chart"]] if result.get("chart") else [])
        if chart_specs:
            st.subheader("Biểu đồ")
            if table["truncated"]:
                st.warning("Biểu đồ đang dùng tối đa 500 dòng kết quả trả về; nhóm phía sau không được thể hiện.")
            chart_functions = {
                "bar": px.bar,
                "line": px.line,
                "scatter": px.scatter,
                "histogram": px.histogram,
                "area": px.area,
                "box": px.box,
                "violin": px.violin,
                "pie": px.pie,
            }
            chart_columns = st.columns(2)
            for chart_index, chart in enumerate(chart_specs):
                chart_args = {"title": f"{chart_labels[chart['type']]} · {chart.get('y') or chart['x']}"}
                if chart["type"] == "pie":
                    chart_args.update({"names": chart["x"], "values": chart["y"]})
                else:
                    chart_args["x"] = chart["x"]
                    if chart.get("y"):
                        chart_args["y"] = chart["y"]
                if chart.get("color"):
                    chart_args["color"] = chart["color"]
                with chart_columns[chart_index % 2]:
                    st.plotly_chart(
                        chart_functions[chart["type"]](result_frame, **chart_args),
                        width="stretch",
                        key=f"render_chart_{chart_context}_{chart_index}",
                    )

if st.button("Xóa dataset"):
    deleted = api_request("DELETE", f"/datasets/{dataset_id}")
    if deleted is not None:
        for key in (
            "dataset_id",
            "filename",
            "profile",
            "preview",
            "sheet",
            "analysis",
            "analysis_request",
        ):
            st.session_state.pop(key, None)
        st.rerun()
