import sys
from pathlib import Path

# ===== FIX IMPORT PATH =====
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st

from backend.pdf_loader import load_handbook
from backend.vector_store import build_vector_store
from backend.rag import build_rag_chain
from backend.config import HANDBOOK_PATH


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Handbook Chatbot",
    page_icon="🎓",
    layout="centered"
)

# ================= SIDEBAR =================
st.sidebar.title("🎓 Student Handbook Chatbot")
st.sidebar.markdown(
    """
    Chatbot hỗ trợ sinh viên tra cứu thông tin  
    **Sổ tay sinh viên** của trường.

    **Lưu ý**
    - Trả lời dựa trên sổ tay
    - Không thay thế quyết định hành chính
    """
)

# ================= INIT SESSION =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    with st.spinner("🔄 Đang tải dữ liệu sổ tay..."):
        documents = load_handbook(HANDBOOK_PATH)
        vector_store = build_vector_store(documents)
        st.session_state.rag_chain = build_rag_chain(vector_store)

# ================= TITLE =================
st.title("💬 Hỏi đáp Sổ tay sinh viên")

# ================= CHAT HISTORY =================
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ================= USER INPUT =================
user_question = st.chat_input("Nhập câu hỏi của bạn...")

if user_question:
    # User message
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

    # AI response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Đang suy nghĩ..."):
            answer = st.session_state.rag_chain.invoke(user_question)
            st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))
