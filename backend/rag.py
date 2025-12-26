from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from backend.prompt import get_prompt
from backend.config import MODEL_NAME, TOP_K


def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatOpenAI(
        model=MODEL_NAME,
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
