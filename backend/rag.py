from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from backend.prompt import get_prompt


def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    llm = ChatOllama(
        model="phi3",   # hoặc "phi3"
        temperature=0
    )

    prompt = get_prompt()

    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
