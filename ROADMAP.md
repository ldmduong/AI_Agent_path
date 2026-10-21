# Lộ trình xây dựng AI Agent phân tích dữ liệu

Tài liệu này mô tả lộ trình xây dựng ứng dụng AI phân tích CSV/Excel với LangGraph, custom tools, vector store, sub-agent, memory, API, giao diện biểu đồ, Docker, tối ưu hiệu suất và evaluation.

## 1. Kiến trúc mục tiêu

```text
Streamlit UI
    │
    ▼
FastAPI
    ├── Upload / dataset API
    ├── Analysis API
    └── Job status API
           │
           ▼
      LangGraph
      ├── Router / Orchestrator
      ├── Data Analyst sub-agent ── custom tools ── CSV / Excel
      ├── Document Researcher ──── vector store
      └── Result formatter ─────── bảng / biểu đồ / trích nguồn
           │
           ├── PostgreSQL: metadata, lịch sử, checkpoint
           ├── Redis: cache và hàng đợi job
           └── File volume: file người dùng tải lên
```

Phân biệt rõ vai trò:

- **Dữ liệu dạng bảng:** dùng công cụ tính toán có cấu trúc; không dùng vector search để thay phép tính.
- **Tài liệu PDF/TXT/DOCX:** đưa qua bước tách đoạn, embedding và vector search.
- **Memory:** lưu trạng thái hội thoại và thông tin nhỏ cần nhớ, không lưu cả file CSV.
- **Biểu đồ:** agent tạo cấu hình biểu đồ; giao diện dùng cấu hình đó để hiển thị.

## 2. Stack đề xuất

| Thành phần | Khởi đầu | Khi cần mở rộng |
|---|---|---|
| Agent/workflow | LangGraph | Thêm graph/subgraph theo nhu cầu |
| API | FastAPI | Chạy nhiều instance khi cần |
| CSV/Excel | pandas, openpyxl | Thử DuckDB khi file lớn hoặc truy vấn chậm |
| Vector store | Chroma lưu bền | pgvector nếu muốn gộp vào PostgreSQL |
| Memory/checkpoint | PostgreSQL | PostgreSQL dùng chung giữa nhiều instance |
| Cache/queue | Redis | Bổ sung monitoring và chính sách eviction |
| UI | Streamlit | Frontend riêng nếu cần UX/phân quyền phức tạp |
| Evaluation | pytest + bộ dữ liệu chuẩn | LangSmith nếu cần giao diện trace/eval |
| Triển khai | Docker Compose | Orchestrator khi cần scale nhiều máy |

Không cần cài và triển khai tất cả ở ngày đầu. Hãy thiết kế ranh giới giữa API, graph, tools và storage để có thể bổ sung dần.

## 3. Lộ trình triển khai

### Giai đoạn 0 — Chốt phạm vi và quy tắc dữ liệu

**Việc cần làm**

- Chốt định dạng đầu vào: CSV và Excel, có hỗ trợ nhiều sheet hay không.
- Chốt giới hạn ban đầu cho dung lượng, số dòng và số sheet.
- Xác định dữ liệu được lưu bao lâu, ai được xem, có cần xóa file sau khi phân tích không.
- Chốt model/LLM và cách cung cấp credentials qua biến môi trường.
- Xác định bộ câu hỏi thường gặp: lọc, nhóm, tổng hợp, xu hướng, tìm tài liệu, tạo biểu đồ.

**Đầu ra**

- Danh sách yêu cầu MVP và giới hạn file.
- Tập câu hỏi mẫu có kết quả đúng để dùng làm evaluation.
- Quy định cơ bản về lưu trữ và dữ liệu nhạy cảm.

**Điều kiện hoàn thành**

- Biết rõ hành vi cho file sai định dạng, cột thiếu, sheet không tồn tại và file quá lớn.

### Giai đoạn 1 — Khảo sát và thiết lập cấu trúc repo

Repo hiện có demo LangGraph trong `rag_langgraph.py` và dependency manifest trong `requirements.txt`. Demo tập trung vào RAG với dữ liệu mẫu, chưa phải ứng dụng phân tích bảng.

**Việc cần làm**

- Giữ demo làm tài liệu/thử nghiệm hoặc tách phần có thể dùng lại; không biến demo thành nơi chứa toàn bộ ứng dụng.
- Kiểm tra các package thực sự được import và đồng bộ với dependency manifest.
- Kiểm tra trạng thái Git của `.env` và dữ liệu Chroma; bảo đảm bí mật và dữ liệu người dùng không bị đưa vào repository.
- Tạo cấu trúc ứng dụng, ví dụ:

```text
app/
  main.py
  api/
    routes_files.py
    routes_analysis.py
  services/
    file_loader.py
    datasets.py
  tools/
    data_tools.py
    document_tools.py
  graph/
    workflow.py
    state.py
  schemas.py
ui/
  streamlit_app.py
tests/
  fixtures/
  test_file_loader.py
  test_tools.py
  test_graph.py
Dockerfile
docker-compose.yml
```

**Điều kiện hoàn thành**

- Chạy được API riêng, không cần chạy script demo RAG.
- Dependency manifest cài được trong môi trường mới.
- Bí mật và dữ liệu phát sinh không bị theo dõi trong Git.

### Giai đoạn 2 — Nạp file và quản lý dataset

**Việc cần làm**

- Tạo bộ đọc CSV/XLSX có kiểm tra định dạng, kích thước, sheet và lỗi mã hóa.
- Sinh `dataset_id` cho file đã nạp; lưu metadata như tên hiển thị, hash nội dung, sheet, ngày tải lên và chủ sở hữu nếu có xác thực.
- Tạo `profile_dataset`: số dòng/cột, tên và kiểu cột, số giá trị thiếu, một mẫu nhỏ.
- Lưu file trong thư mục/volume riêng, không lưu trong thư mục ứng dụng hoặc Docker image.
- Không đọc toàn bộ file vào prompt LLM.

**API nên có**

- `POST /files`: upload, trả `dataset_id`.
- `GET /datasets/{dataset_id}/profile`: profile của dataset.
- `DELETE /datasets/{dataset_id}`: xóa dataset và dữ liệu liên quan.

**Điều kiện hoàn thành**

- Có test cho CSV thường, CSV mã hóa khác, Excel nhiều sheet, file rỗng, file lỗi và file quá giới hạn.
- Lỗi được trả rõ ràng, không biến thành kết quả thành công giả.

### Giai đoạn 3 — Custom tools phân tích dữ liệu

**Bộ tool MVP**

- `profile_dataset(dataset_id)`
- `list_columns(dataset_id)`
- `filter_dataset(dataset_id, filters, columns)`
- `aggregate_dataset(dataset_id, group_by, metrics)`
- `calculate_statistics(dataset_id, columns)`
- `get_chart_data(dataset_id, query_spec)`

Mỗi tool cần kiểm tra:

- Dataset tồn tại và người gọi có quyền truy cập.
- Tên cột và kiểu dữ liệu hợp lệ.
- Phép toán thuộc danh sách hỗ trợ.
- Giới hạn số dòng hoặc số nhóm trả về.

**An toàn**

Không chạy mã Python do LLM tạo ra bằng `eval`/`exec`. LLM chỉ được chọn tool và tạo tham số; phần tính toán là mã do ứng dụng kiểm soát.

**Điều kiện hoàn thành**

- Kết quả tool được kiểm chứng bằng test với fixture có đáp án đã biết.
- Tên cột không hợp lệ hoặc yêu cầu ngoài phạm vi trả lỗi/hỏi lại thay vì tự đoán.

### Giai đoạn 4 — LangGraph và sub-agent

Tạo graph điều phối có state rõ ràng, chẳng hạn:

```text
question
dataset_id
messages / conversation_id
analysis_plan
tool_results
document_sources
chart_config
final_answer
errors
```

**Luồng xử lý ban đầu**

```text
START
  → phân loại yêu cầu
  → Data Analyst subgraph hoặc Document Researcher subgraph
  → kiểm tra kết quả
  → định dạng câu trả lời
  → END
```

**Vai trò agent**

- **Supervisor:** định tuyến câu hỏi đến chuyên gia phù hợp bằng model có structured output.
- **Dataset Expert:** profile, liệt kê cột và thống kê mô tả.
- **Filter Expert:** chỉ lập kế hoạch lọc có điều kiện rõ ràng.
- **Analytics Expert:** tổng hợp, nhóm và tính metric.
- **Visualization Expert:** lập kế hoạch tổng hợp kèm cấu hình biểu đồ.
- **Document Researcher:** đã có LangGraph riêng để tìm nội dung hỗ trợ trong TXT/PDF và trả nguồn.
- **Chart Builder:** ban đầu nên là logic/tool tạo cấu hình biểu đồ, chưa cần agent riêng.
- **Reviewer:** chỉ thêm sau khi evaluation cho thấy cần kiểm tra kết quả độc lập.

Nếu câu hỏi đơn giản như “đếm số dòng”, graph nên gọi tool trực tiếp, không cần gọi nhiều agent.

**Đã triển khai trong MVP:** supervisor và bốn chuyên gia dữ liệu là các node riêng trong graph; mỗi chuyên gia chỉ được phép lập kế hoạch cho nhóm operation của mình. Model/provider được chọn qua `LLM_PROVIDER` và `LLM_MODEL` (mặc định Ollama/Qwen); để đổi model không cần sửa graph. Mỗi yêu cầu hiện cần một lần định tuyến và một lần lập kế hoạch.

**Điều kiện hoàn thành**

- Graph định tuyến đúng các nhóm câu hỏi cơ bản.
- Kết quả có thể lần theo: câu hỏi → agent → tool → kết quả.
- Lỗi tool và lỗi model không bị che thành câu trả lời thành công.

### Giai đoạn 5 — Vector store và hỏi đáp tài liệu

**Đã triển khai trong MVP**

- API nhận TXT/PDF, cắt đoạn, tạo embedding Ollama và lưu Chroma bền vững riêng với dữ liệu bảng.
- Metadata mỗi đoạn gồm document ID, tên file, số trang và số đoạn; câu hỏi chỉ truy vấn các document ID người dùng chọn.
- Document Researcher dùng LangGraph: truy xuất → nếu đủ liên quan thì trả lời có trích dẫn → nếu không đủ thì từ chối không gọi LLM.
- UI hỗ trợ nạp nhiều tài liệu, chọn một/nhiều tài liệu để hỏi, xem trích đoạn nguồn và xóa tài liệu.
- PDF scan cần OCR chưa được hỗ trợ; hỗ trợ hiện tại là văn bản trích xuất được từ PDF và TXT UTF-8.
- Lưu vector tại `data/vector_store/` mặc định; `AI_AGENT_VECTOR_DIR` đổi thư mục lưu.

**Việc còn lại trước production**

- Thêm xác thực và gắn document ID với chủ sở hữu/workspace; hiện API chưa có phân quyền.
- Bổ sung ingestion bất đồng bộ cho tài liệu lớn và bộ đánh giá retrieval/citation thực tế.

**Điều kiện hoàn thành**

- Test truy vấn tìm được tài liệu liên quan.
- Test câu hỏi không có thông tin không tạo câu trả lời bịa.
- Nguồn trích dẫn xác định được tài liệu/đoạn thực tế.

### Giai đoạn 6 — API và giao diện

**Đã triển khai trong MVP**

- API phân tích có schema thống nhất cho tổng hợp, lọc, thống kê và `trend` theo ngày/tuần/tháng/quý/năm; trả bảng, cấu hình biểu đồ và tóm tắt xu hướng.
- UI hỗ trợ bar, line, scatter, histogram, area, box, violin, pie; chọn trục và cột màu.
- Xu hướng được tính xác định từ cột thời gian và các metric, không để LLM tự suy diễn số liệu.

**API bổ sung**

- `POST /datasets/{dataset_id}/analyze`: nhận câu hỏi, trả kết quả hoặc mã job.
- `GET /jobs/{job_id}`: lấy trạng thái tác vụ dài.
- `GET /health`: kiểm tra dịch vụ.

Kết quả phân tích nên có cấu trúc ổn định:

```text
answer
table
chart: type, x, y, labels, title
sources
job_id hoặc request_id
```

**Giao diện Streamlit**

- Upload file và chọn sheet.
- Xem profile dữ liệu.
- Nhập câu hỏi.
- Hiển thị câu trả lời, bảng, nguồn và biểu đồ.
- Hiển thị lỗi, thời gian chờ và trạng thái job rõ ràng.

**Điều kiện hoàn thành**

- Người dùng đi được từ upload đến xem phân tích mà không cần gọi script thủ công.
- Biểu đồ được dựng từ dữ liệu trả về, không phụ thuộc vào nội dung tự do do LLM sinh ra.

### Giai đoạn 7 — Memory, checkpoint, cache và xử lý bất đồng bộ

**Memory/checkpoint**

- Dùng state của graph cho dữ liệu của một lượt chạy.
- Dùng checkpoint gắn với `conversation_id` để tiếp tục hội thoại.
- Chỉ lưu sở thích hoặc thông tin dài hạn khi thực sự cần; không đưa toàn bộ file hay dữ liệu nhạy cảm vào memory.

**Cache**

- Profile: hash file + sheet + tùy chọn đọc.
- Kết quả tính toán: phiên bản dataset + phép tính + tham số.
- Embedding: hash tài liệu + model embedding.

Nếu cache dùng chung nhiều người dùng, khóa phải bao gồm phạm vi sở hữu/quyền truy cập để không trả dữ liệu của người này cho người khác. Cache LLM cần thận trọng vì ngữ cảnh có thể chứa dữ liệu riêng tư.

**Async và job nền**

- Dùng async cho I/O khi client hỗ trợ.
- Phân tích pandas nặng không tự trở nên bất đồng bộ chỉ vì endpoint dùng `async`.
- Với tác vụ lâu, dùng queue/worker; Redis có thể làm queue và cache.
- API trả `job_id`, giao diện polling trạng thái.
- Giới hạn timeout, retry và số tác vụ đồng thời.

**Điều kiện hoàn thành**

- Câu hỏi ngắn vẫn trả trực tiếp nhanh.
- Tác vụ dài không giữ kết nối HTTP mở vô thời hạn.
- Cache được vô hiệu hóa đúng khi dataset thay đổi hoặc bị xóa.

### Giai đoạn 8 — Evaluation và tracing

Evaluation cần chạy xuyên suốt, không để đến cuối mới làm.

**Đã triển khai trong MVP:** fixture evaluation tổng hợp cho aggregation/trend/chart; runner gọi model đang cấu hình và chấm routing, operation, tool success cùng kết quả đầu ra. Pytest vẫn kiểm tra kết quả dữ liệu, graph, chart và RAG một cách xác định.

Tracing cục bộ mặc định bật, lưu vào SQLite và chỉ ghi tên graph/component, event, thời lượng cùng liên kết parent/child; không ghi input/output hay dữ liệu hàng. Có thể tắt bằng `AI_AGENT_TRACE_ENABLED=false`; xem trace gần nhất qua `GET /traces`.

**Bộ test**

1. **Tool/unit tests:** phép tính và kiểm tra đầu vào.
2. **Graph tests:** chọn đúng agent/tool, xử lý thiếu thông tin và lỗi.
3. **RAG tests:** tìm đúng tài liệu, trích nguồn và từ chối khi thiếu căn cứ.
4. **Chart tests:** loại biểu đồ, tên trục và dữ liệu đầu vào.
5. **API tests:** upload, phân quyền, status code và response schema.
6. **Performance tests:** thời gian, kích thước file và số request đồng thời.

**Chỉ số cần theo dõi**

- Độ chính xác phép tính so với đáp án chuẩn.
- Tỷ lệ chọn đúng tool và tham số.
- Tỷ lệ retrieval đúng và nguồn trích dẫn đúng.
- Tỷ lệ agent hỏi lại/từ chối đúng khi thiếu dữ liệu.
- Latency theo từng node, tổng latency, lỗi, token và chi phí.
- Cache hit rate và tỷ lệ job thất bại.

Bắt đầu bằng pytest và bộ câu hỏi chuẩn lưu trong repo. Có thể dùng LangSmith cho trace/evaluation nếu phù hợp với chính sách dữ liệu; nếu không, giữ log và bộ eval nội bộ, tránh ghi nội dung file nhạy cảm.

**Điều kiện hoàn thành**

- Có baseline trước khi đổi prompt/model/embedding.
- Mỗi thay đổi quan trọng chạy lại bộ evaluation.
- Phát hiện được nếu kết quả mới kém hơn baseline; chưa cần đặt ngưỡng số tùy ý khi chưa có dữ liệu thực tế.

### Giai đoạn 9 — Docker và triển khai máy khác

**Việc cần làm**

- Tạo Dockerfile cho ứng dụng; không đóng gói file tải lên hoặc credentials vào image.
- Dùng Docker Compose để chạy các service cần thiết: API, UI, PostgreSQL, Redis và worker khi đã có tác vụ nền.
- Dùng volume cho dữ liệu cần giữ sau khi container restart.
- Đưa model/API key vào biến môi trường hoặc secret store, không commit vào Git.
- Tạo health check, log có request/job ID và hướng dẫn cấu hình.
- Thử triển khai sạch trên máy khác: build image, khởi động, upload, phân tích, restart, kiểm tra dữ liệu còn/đã xóa đúng.

**Điều kiện hoàn thành**

- Máy mới chỉ cần Docker và file cấu hình môi trường mẫu để chạy.
- Restart container không làm mất dữ liệu cần lưu.
- Có hướng dẫn backup/restore cho PostgreSQL và dữ liệu file.

### Giai đoạn 10 — Bảo mật và vận hành trước production

- Thêm đăng nhập và phân quyền nếu có nhiều người dùng.
- Cô lập dataset, conversation, vector search và cache theo user/workspace.
- Giới hạn loại file, dung lượng, số dòng, thời gian chạy và output.
- Không ghi API key, nội dung nhạy cảm hoặc toàn bộ dữ liệu bảng vào log.
- Đặt chính sách xóa file, memory, checkpoint và vector embeddings.
- Theo dõi lỗi, latency, số job chờ và tài nguyên máy.
- Kiểm thử backup/restore và xử lý khi worker hoặc database khởi động lại.

## 4. Thứ tự MVP thực tế

1. **Mốc A:** upload CSV/XLSX → profile → API.
2. **Mốc B:** custom tools → phép tính có test → LangGraph Data Analyst.
3. **Mốc C:** Streamlit → bảng và biểu đồ.
4. **Mốc D:** Document Researcher → vector store → trích nguồn.
5. **Mốc E:** PostgreSQL checkpoint/memory → hội thoại nhiều lượt.
6. **Mốc F:** Redis cache/queue → worker cho tác vụ dài.
7. **Mốc G:** evaluation regression → Docker Compose → triển khai máy khác.
8. **Mốc H:** xác thực, phân quyền và hardening nếu đưa cho nhiều người dùng.

## 5. Nguyên tắc ưu tiên

- Làm kết quả phân tích đúng và có thể kiểm thử trước; sau đó mới tối ưu tốc độ.
- Cache một phép tính sai chỉ làm kết quả sai nhanh hơn.
- Không thêm nhiều sub-agent nếu một tool hoặc một node giải quyết được yêu cầu.
- Không đưa toàn bộ file vào prompt, memory hoặc log.
- Tách dữ liệu bảng khỏi luồng RAG tài liệu.
- Đo bottleneck trước khi thêm worker, cache hay hạ tầng phức tạp.

## 6. Giả định cần xác nhận trước production

- Dung lượng file tối đa và số dòng cần hỗ trợ.
- Số người dùng và số tác vụ đồng thời.
- Thời gian lưu file, kết quả, checkpoint và embeddings.
- Yêu cầu xác thực, phân quyền và mức độ nhạy cảm của dữ liệu.
- Model/LLM được phép dùng và chính sách gửi dữ liệu tới nhà cung cấp.

Các quyết định này ảnh hưởng đến việc có cần DuckDB, worker, phân vùng dữ liệu, hạ tầng nhiều máy và quy trình bảo mật nâng cao hay không.
