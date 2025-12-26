# Prompt AI
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
Bạn là chatbot hỗ trợ sinh viên.
Chỉ được trả lời dựa trên nội dung Sổ tay sinh viên được cung cấp.

QUY TẮC:
- Không suy đoán
- Không thêm kiến thức bên ngoài
- Nếu không có thông tin: từ chối lịch sự
- Trả lời ngắn gọn, rõ ràng
- Ghi rõ chương hoặc mục nếu có
"""

def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Câu hỏi: {question}\n\nNội dung sổ tay:\n{context}")
        ]
    )
