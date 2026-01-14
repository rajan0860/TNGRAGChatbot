from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import EMBEDDING_MODEL_NAME

def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

def create_index_semantic(text):
    """
    Creates a FAISS index using semantic chunking.
    """
    embeddings = get_embeddings()
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    docs = text_splitter.create_documents([text])
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def create_index_recursive(text):
    """
    Creates a FAISS index using recursive character text splitting.
    """
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = text_splitter.create_documents([text])
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def save_index(vector_store, path):
    vector_store.save_local(path)

def load_index(path):
    embeddings = get_embeddings()
    # FAISS.load_local requires allow_dangerous_deserialization=True if loading untrusted files
    # Since we create it ourselves, it's generally fine, but good to be aware.
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
