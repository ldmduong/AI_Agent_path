"""
BÀI TẬP RAG VỚI LANGGRAPH (100% LOCAL)
======================================

RAG = Retrieval-Augmented Generation = "Truy xuất rồi mới sinh câu trả lời".
LLM không tự biết tài liệu nội bộ của bạn, nên ta làm 2 việc:
    1. Tìm những đoạn tài liệu liên quan nhất tới câu hỏi (Retrieval)
    2. Đưa các đoạn đó + câu hỏi cho LLM để nó trả lời dựa trên đó (Generation)

Đồ thị trong file này:

    START -> retrieve -> grade ──(có đoạn liên quan)──> generate -> END
                           │
                           └──(không có đoạn nào)────> no_answer -> END

Chạy thử:
    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text      (hoặc: ollama pull bge-m3 rồi đổi EMBED_MODEL)
    pip install langchain-core langchain-text-splitters langchain-chroma langchain-ollama langgraph
    python rag_langgraph.py
"""

from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import StateGraph, START, END


# ==========================================================
# 0. CẤU HÌNH (chỉnh ở đây để thử nghiệm)
# ==========================================================
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"   # tiếng Việt tốt hơn: "bge-m3"
TOP_K = 3                          # lấy tối đa mấy đoạn gần nhất
MAX_DISTANCE = 0.55                # đoạn nào có khoảng cách > ngưỡng này bị coi là KHÔNG liên quan
                                   # (cosine distance: 0 = giống hệt, càng lớn càng khác)
                                   # Hãy nhìn điểm in ra khi chạy để chỉnh số này cho hợp model.


# ==========================================================
# 1. INDEXING: biến tài liệu thành vector và lưu vào DB
#    (làm MỘT LẦN trước khi hỏi đáp)
# ==========================================================
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0)

# Dữ liệu mẫu. Thực tế bạn sẽ đọc từ file .txt/.pdf rồi tạo Document tương tự.
raw_documents = [
    Document(
        page_content="Chính sách nghỉ phép năm 2026: Nhân viên chính thức có 15 ngày phép nguyên lương mỗi năm. "
                     "Làm thêm mỗi 3 năm được cộng 1 ngày.",
        metadata={"source": "so_tay_nhan_su.txt"},
    ),
    Document(
        page_content="Quy định làm việc từ xa (Work From Home): Nhân viên được WFH tối đa 2 ngày/tuần "
                     "vào Thứ 4 và Thứ 6, cần báo trước trên hệ thống trước 9h sáng.",
        metadata={"source": "so_tay_nhan_su.txt"},
    ),
    Document(
        page_content="Hỗ trợ thiết bị: Công ty cấp ngân sách 5.000.000 VNĐ/năm để nhân viên mua sắm chuột, "
                     "bàn phím, hoặc nâng cấp RAM/SSD phục vụ công việc.",
        metadata={"source": "chinh_sach_it.txt"},
    ),
    Document(
        page_content="Giờ làm việc: Từ Thứ 2 đến Thứ 6, 8h30 đến 17h30, nghỉ trưa 1 tiếng từ 12h đến 13h. "
                     "Đi trễ quá 3 lần/tháng sẽ bị nhắc nhở.",
        metadata={"source": "so_tay_nhan_su.txt"},
    ),
]

# Bước 1a: Cắt tài liệu dài thành các đoạn nhỏ (chunk).
#   - Vì sao cắt? Để mỗi vector đại diện cho MỘT ý, tìm kiếm chính xác hơn,
#     và không nhét cả tài liệu khổng lồ vào prompt.
#   - chunk_overlap: các đoạn gối nhau một chút để không cắt mất ngữ cảnh ở ranh giới.
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(raw_documents)

# Bước 1b: Embedding = biến mỗi chunk thành một vector số.
#   Hai đoạn có nghĩa gần nhau thì vector gần nhau. Vector được lưu vào ChromaDB.
#   "hnsw:space": "cosine" -> đo khoảng cách bằng cosine (dễ hiểu, từ 0 đến ~1).
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    # persist_directory="./chroma_db",   # bỏ comment nếu muốn lưu ra ổ cứng
)

print(f"[INDEX] {len(raw_documents)} tài liệu -> {len(chunks)} chunk đã được lưu vào ChromaDB\n")


# ==========================================================
# 2. STATE: "giỏ" dữ liệu đi xuyên suốt đồ thị
#    Mỗi node nhận state, trả về phần state muốn cập nhật.
# ==========================================================
class RAGState(TypedDict):
    question: str                # câu hỏi của người dùng
    context: List[Document]      # các đoạn tài liệu được giữ lại sau khi lọc
    distances: List[float]       # khoảng cách tương ứng (để quan sát/debug)
    answer: str                  # câu trả lời cuối cùng


# ==========================================================
# 3. CÁC NODE
# ==========================================================
def retrieve_node(state: RAGState):
    """Node 1 - RETRIEVAL: embed câu hỏi rồi tìm TOP_K chunk gần nhất."""
    print("--- 🔍 BƯỚC 1: TRUY XUẤT ---")
    results = vectorstore.similarity_search_with_score(state["question"], k=TOP_K)

    for doc, dist in results:
        print(f"    distance={dist:.3f} | {doc.metadata['source']} | {doc.page_content[:60]}...")

    return {
        "context": [doc for doc, _ in results],
        "distances": [dist for _, dist in results],
    }


def grade_node(state: RAGState):
    """Node 2 - LỌC: bỏ những đoạn quá xa câu hỏi.

    Nếu không có bước này, retriever LUÔN trả về TOP_K đoạn dù chẳng liên quan,
    và LLM nhỏ rất dễ bịa ra câu trả lời từ đống ngữ cảnh sai đó.
    """
    print("--- ⚖️  BƯỚC 2: LỌC THEO NGƯỠNG ---")
    kept = [
        doc for doc, dist in zip(state["context"], state["distances"])
        if dist <= MAX_DISTANCE
    ]
    print(f"    Giữ lại {len(kept)}/{len(state['context'])} đoạn (ngưỡng {MAX_DISTANCE})")
    return {"context": kept}


def generate_node(state: RAGState):
    """Node 3 - GENERATION: đưa ngữ cảnh + câu hỏi cho LLM."""
    print("--- 🤖 BƯỚC 3: LLM SINH CÂU TRẢ LỜI ---")

    # Ghép các đoạn tìm được, kèm tên nguồn để LLM có thể trích dẫn.
    context_text = "\n\n".join(
        f"[Nguồn: {d.metadata.get('source', 'N/A')}] {d.page_content}"
        for d in state["context"]
    )

    # System message đặt "luật chơi"; Human message chứa dữ liệu cụ thể.
    messages = [
        SystemMessage(content=(
            "Bạn là trợ lý nội bộ của công ty. Chỉ dựa vào NGỮ CẢNH được cung cấp để trả lời. "
            "Nếu ngữ cảnh không có thông tin, hãy nói rõ 'Tài liệu không đề cập', tuyệt đối không tự bịa. "
            "Trả lời ngắn gọn bằng tiếng Việt và nêu tên nguồn."
        )),
        HumanMessage(content=f"NGỮ CẢNH:\n{context_text}\n\nCÂU HỎI: {state['question']}"),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}


def no_answer_node(state: RAGState):
    """Node dự phòng: không có đoạn nào liên quan -> trả lời luôn, KHÔNG gọi LLM."""
    print("--- 🚫 KHÔNG TÌM THẤY TÀI LIỆU LIÊN QUAN ---")
    return {"answer": "Tài liệu không đề cập đến nội dung này."}


# ==========================================================
# 4. HÀM ĐỊNH TUYẾN (conditional edge)
#    Đọc state và trả về TÊN node tiếp theo.
# ==========================================================
def route_after_grade(state: RAGState) -> str:
    return "generate" if state["context"] else "no_answer"


# ==========================================================
# 5. RÁP ĐỒ THỊ
# ==========================================================
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.add_node("no_answer", no_answer_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    route_after_grade,
    {"generate": "generate", "no_answer": "no_answer"},
)
workflow.add_edge("generate", END)
workflow.add_edge("no_answer", END)

app = workflow.compile()


# ==========================================================
# 6. CHẠY THỬ
# ==========================================================
def ask(question: str):
    print(f"\n❓ Hỏi: {question}")
    result = app.invoke({"question": question})
    print(f"✅ Trả lời:\n{result['answer']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Có trong tài liệu
    ask("Tôi được làm ở nhà mấy ngày 1 tuần và vào thứ mấy?")
    ask("Công ty hỗ trợ bao nhiêu tiền để mua bàn phím?")

    # Ngoài tài liệu -> kỳ vọng đi nhánh no_answer
    ask("Công ty có thưởng Tết bằng cổ phiếu không?")

    # Gợi ý thử nghiệm để hiểu RAG:
    #  - Đổi MAX_DISTANCE thành 2.0 (không lọc) rồi hỏi câu về cổ phiếu: xem LLM có bịa không.
    #  - Đổi TOP_K = 1 rồi hỏi câu cần 2 nguồn: xem câu trả lời thiếu thế nào.
    #  - Đổi EMBED_MODEL = "bge-m3" và so sánh các giá trị distance.
