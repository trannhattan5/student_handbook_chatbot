from langchain_community.document_loaders import PyPDFLoader

def load_handbook(path: str = "data/handbook.pdf"):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents
