# AI Data Agent — thử nghiệm UI và API

MVP này nhận CSV/XLSX, hiển thị profile và preview dữ liệu, rồi chạy các phép lọc, tổng hợp và thống kê mô tả bằng thao tác có cấu trúc. Supervisor trong LangGraph định tuyến câu hỏi đến chuyên gia khám phá dữ liệu, lọc, phân tích hoặc biểu đồ; chuyên gia lập kế hoạch tool có schema, còn phép tính luôn chạy xác định bằng pandas. UI là Streamlit; backend là FastAPI.

> Model chỉ nhận câu hỏi và tên/kiểu cột để định tuyến/lập kế hoạch; không nhận dữ liệu từng dòng và không tự chạy SQL/Python. LangGraph điều phối chuyên gia và tool; pandas thực hiện phép tính.

## Cài đặt

Yêu cầu Python 3.11 trở lên. Tại thư mục gốc repository, chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn việc activate môi trường, có thể gọi trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy ứng dụng

Mở hai terminal tại thư mục gốc.

Terminal 1 — API:

```powershell
python -m uvicorn app.main:app --reload
```

Terminal 2 — giao diện:

```powershell
python -m streamlit run ui/streamlit_app.py
```

Mở URL Streamlit được in trong terminal (thường là `http://localhost:8501`). API docs tương tác ở `http://127.0.0.1:8000/docs`.

Mặc định ứng dụng dùng Ollama với `qwen2.5:1.5b` (`ollama pull qwen2.5:1.5b`). Cấu hình model bằng biến môi trường:

| Biến | Mặc định | Mục đích |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Provider LangChain, ví dụ `ollama`, `openai`, `google_genai` |
| `LLM_MODEL` | giá trị `OLLAMA_MODEL` hoặc `qwen2.5:1.5b` | Tên model; đổi biến này để nâng model mà không sửa graph |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | URL server khi provider là Ollama |
| `AI_AGENT_API_URL` | `http://127.0.0.1:8000` | URL FastAPI mà Streamlit gọi |

Khi dùng OpenAI hoặc Google, đặt thêm `OPENAI_API_KEY` hoặc `GOOGLE_API_KEY`. Các provider tích hợp được khai báo trong `requirements.txt`; khởi động lại API sau khi đổi cấu hình. `OLLAMA_MODEL` vẫn được hỗ trợ để tương thích với cấu hình cũ.

Backend và evaluation tự đọc `.env` ở thư mục gốc repository; biến môi trường của hệ thống được ưu tiên hơn giá trị trong file này.

Hỏi đáp tài liệu dùng embedding Ollama riêng, mặc định `nomic-embed-text` (`ollama pull nomic-embed-text`). Có thể đổi model embedding bằng `EMBEDDING_MODEL`; URL Ollama lấy từ `OLLAMA_BASE_URL`. Chroma lưu bền trong `data/vector_store/` hoặc thư mục `AI_AGENT_VECTOR_DIR` tùy chỉnh. Ngưỡng khoảng cách cosine mặc định là `DOCUMENT_MAX_DISTANCE=0.45`; nếu đổi model embedding, hãy dùng vector directory mới và nạp lại tài liệu.

Có thể tải thử file mẫu `examples/sales.csv`. Mặc định file tải lên được lưu trong `data/uploads/`, giới hạn 25 MB/file, tối đa 500.000 dòng và 200 cột. Đặt biến môi trường `AI_AGENT_UPLOAD_DIR` để đổi thư mục lưu; khi cần giữ file qua lần chạy/container, hãy dùng volume.

## API chính

- `GET /health`
- `POST /files` — upload CSV hoặc XLSX.
- `GET /datasets/{dataset_id}/profile?sheet=...`
- `GET /datasets/{dataset_id}/preview?sheet=...&limit=100`
- `POST /datasets/{dataset_id}/analyze` — phép tính có schema; không nhận SQL hoặc Python tùy ý.
- `POST /datasets/{dataset_id}/ask` — định tuyến đến chuyên gia, lập kế hoạch bằng model đã cấu hình và thực thi bằng các phép phân tích đã cho phép.
- `POST /datasets/{dataset_id}/ask/jobs` — gửi câu hỏi chạy nền; trả `job_id` và `conversation_id`.
- `GET /analysis-jobs/{job_id}` — poll trạng thái `queued`, `running`, `succeeded` hoặc `failed`; kết quả có ở `result` khi hoàn tất.
- `DELETE /datasets/{dataset_id}/conversations/{conversation_id}` — xóa memory/checkpoint của hội thoại.
- `POST /documents` — nạp TXT/PDF, cắt đoạn và lập chỉ mục vector.
- `GET /documents` — liệt kê tài liệu đã lập chỉ mục (để mở lại sau khi khởi động lại UI/API).
- `POST /documents/ask` — hỏi tối đa 10 tài liệu đã nạp; trả nguồn theo tên file, trang và đoạn.
- `DELETE /documents/{document_id}` — xóa các vector của tài liệu.
- `DELETE /datasets/{dataset_id}`

### Memory, checkpoint, cache và tác vụ nền

Trạng thái ứng dụng mặc định được lưu trong `data/agent_state.sqlite3`; đổi vị trí bằng `AI_AGENT_STATE_DB`. LangGraph SQLite checkpointer giữ tối đa 10 lượt hỏi và kế hoạch phân tích gần nhất cho mỗi cặp dataset/conversation, trong 30 ngày. Checkpoint không chứa DataFrame, hàng dữ liệu hoặc kết quả trả về, nhưng có lưu nội dung câu hỏi và tên cột/metrics; xóa lịch sử bằng endpoint conversation hoặc xóa dataset để xóa checkpoint. API đồng bộ `/ask` vẫn được giữ để tương thích; khi gửi `conversation_id`, câu hỏi tiếp theo có thể tham chiếu lịch sử trước đó. UI dùng endpoint job và giữ cùng conversation ID cho đến khi xóa lịch sử hoặc tải dataset khác.

`POST /analyze` cache kết quả xác định theo dataset, sheet và request trong 10 phút (`X-Cache: HIT|MISS`). Kết quả phân tích được lưu cục bộ trong SQLite trong thời hạn cache. Job và kết quả của job được giữ tối đa 24 giờ. Xóa dataset sẽ xóa cache, job và checkpoint liên quan; xóa hội thoại sẽ xóa toàn bộ checkpoint của hội thoại đó.

Job hiện chạy trong tiến trình FastAPI. Khởi động lại API sẽ đánh dấu job đang chờ/chạy là thất bại thay vì tự tiếp tục; chạy nhiều instance/worker không được hỗ trợ ở MVP. SQLite có thể chứa câu hỏi, kế hoạch, kết quả cache và kết quả job; chỉ triển khai trên máy tin cậy, bảo vệ quyền truy cập file DB và không dùng cấu hình này cho dịch vụ đa người dùng chưa có xác thực/phân quyền.

### Evaluation và tracing

Chạy bộ đánh giá hồi quy với model/provider đang cấu hình trong `.env`:

```powershell
python -m scripts.run_evaluations
```

Evaluation dùng các dataset tổng hợp trong `evals/analysis_cases.json`, không đọc file upload. Báo cáo riêng độ chính xác định tuyến specialist, operation, tool execution và đầu ra số/chart. Dùng `--strict` để lệnh trả mã lỗi khác 0 nếu bất kỳ ca nào sai:

```powershell
python -m scripts.run_evaluations --strict
```

`pytest` luôn kiểm tra đáp án phân tích xác định, graph, chart và RAG mà không cần gọi LLM; lệnh evaluation ở trên gọi model thật nên có thể chậm, tốn tài nguyên hoặc chi phí provider.

Tracing LangGraph được bật mặc định và lưu vào SQLite cùng `AI_AGENT_STATE_DB`. `GET /traces?limit=100` trả tên graph/component, trạng thái event, thời lượng và quan hệ parent/child; trace được giữ tối đa 14 ngày. Trace không ghi prompt, câu hỏi, nội dung tài liệu, hàng dữ liệu, đầu vào/đầu ra tool, response model hoặc nội dung lỗi. Có thể tắt bằng `AI_AGENT_TRACE_ENABLED=false`. Nếu bật tracing trong môi trường nhiều người dùng, endpoint traces cần được bảo vệ cùng API.

Có thể bật LangSmith tracing bằng `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` và tùy chọn `LANGSMITH_PROJECT=ai-data-agent` trong `.env`. Mặc định LangSmith không nhận trace; `AI_AGENT_TRACE_ENABLED` chỉ điều khiển tracing local SQLite. Khi bật LangSmith, tích hợp LangGraph gửi run inputs, outputs và metadata: với phân tích dữ liệu, graph state có thể gồm câu hỏi, toàn bộ DataFrame và kết quả; với hỏi đáp tài liệu, trace có thể gồm nội dung các đoạn được truy xuất. Chỉ bật với dữ liệu tổng hợp/đã ẩn danh hoặc khi chính sách dữ liệu cho phép. Các lần chạy `scripts.run_evaluations` cũng tạo trace từng ca; báo cáo tổng hợp vẫn chỉ in ở local.

Sao chép `.env.example` thành `.env`, rồi tự thêm khóa LangSmith nếu muốn bật. Không commit khóa API.

Ví dụ hỏi tài liệu:

```json
{
  "question": "Chính sách nghỉ phép quy định thế nào?",
  "document_ids": ["<document_id trả về khi upload>"]
}
```

UI cho phép nạp nhiều tài liệu và chọn tài liệu để tra cứu. Document Researcher là một LangGraph riêng: tìm các đoạn gần câu hỏi trong Chroma, bỏ qua đoạn vượt ngưỡng tương đồng, sau đó mới gọi model trả lời. Nếu không tìm được đoạn phù hợp, graph trả lời rõ ràng mà không gọi model. PDF scan dạng ảnh chưa được OCR; hiện chỉ lập chỉ mục văn bản trích xuất được.

Ví dụ body tổng hợp:

```json
{
  "operation": "aggregate",
  "group_by": ["region"],
  "metrics": [{"column": "revenue", "function": "sum"}],
  "chart": {"type": "bar", "x": "region", "y": "sum_revenue"}
}
```

Ví dụ phân tích xu hướng tháng (cột thời gian phải chứa ngày/giờ hoặc chuỗi ngày có thể parse):

```json
{
  "operation": "trend",
  "time_column": "date",
  "interval": "month",
  "metrics": [
    {"column": "revenue", "function": "sum"},
    {"column": "order_id", "function": "count"}
  ],
  "charts": [
    {"type": "line", "x": "date", "y": "sum_revenue"},
    {"type": "bar", "x": "date", "y": "count_order_id"}
  ]
}
```

`interval` hỗ trợ `day`, `week`, `month`, `quarter`, `year`. Kết quả trả chuỗi dữ liệu đã sắp xếp cùng `trend` (hướng, giá trị đầu/cuối, thay đổi tuyệt đối và phần trăm). UI hỗ trợ bar, line, scatter, histogram, area, box, violin và pie; có thể cấu hình và hiển thị tối đa 6 biểu đồ cùng lúc. Scatter cần hai trục số; pie/box/violin cần trục Y số. API nhận danh sách `charts`; trường `chart` đơn lẻ vẫn được hỗ trợ để tương thích client cũ.

Supervisor và các chuyên gia là các node LangGraph riêng: `dataset` (profile/cột/thống kê), `filter`, `analytics` và `visualization`. Mỗi chuyên gia chỉ lập kế hoạch trong phạm vi operation của mình; graph sau đó gọi một trong các custom tool `profile_dataset`, `list_columns`, `filter_dataset`, `aggregate_dataset`, `calculate_statistics`, `get_chart_data`. Tool nhận DataFrame qua state nội bộ, không đưa các hàng dữ liệu vào prompt. API trả chuyên gia, model và tool đã dùng để có thể kiểm tra; xem schema API tại `/docs`.

Phép tổng hợp hỗ trợ `sum`, `mean`, `median`, `min`, `max`, `count`; lọc hỗ trợ so sánh số, so khớp văn bản và kiểm tra giá trị null. Thống kê mô tả hỗ trợ chọn các cột số.

## Kiểm thử

```powershell
python -m pytest
```

Các file người dùng upload và Chroma database không được thêm vào Git (`data/uploads/` và `data/vector_store/` đã nằm trong `.gitignore`). MVP hiện chạy cục bộ, chưa có xác thực/phân quyền; chưa phù hợp cho triển khai đa người dùng.
