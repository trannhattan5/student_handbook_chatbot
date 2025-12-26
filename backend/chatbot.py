from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from .prompt import SYSTEM_PROMPT

def create_chatbot(vector_store):
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"system_prompt": SYSTEM_PROMPT}
    )
