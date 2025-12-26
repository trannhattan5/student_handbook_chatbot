# FastAPI entry
from fastapi import FastAPI

from backend.pdf_loader import load_handbook
from backend.vector_store import build_vector_store
from backend.rag import build_rag_chain
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Student Handbook AI Chatbot")

# ===== INIT APP =====
documents = load_handbook("data/handbook.pdf")
vector_store = build_vector_store(documents)
rag_chain = build_rag_chain(vector_store)


@app.get("/ask")
def ask(question: str):
    answer = rag_chain.invoke(question)
    return {
        "question": question,
        "answer": answer
    }
