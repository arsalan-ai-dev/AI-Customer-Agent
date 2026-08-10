import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# Initialize lightweight embedding model (ONNX runtime uses < 150MB RAM)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def process_and_ingest_document(file_path: str) -> dict:
    """Ingest a single PDF or TXT file into ChromaDB."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)

    # 1. Select correct loader based on file extension
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")

    docs = loader.load()

    if not docs:
        return {"filename": filename, "chunks_added": 0}

    # Tag original filename in document metadata
    for doc in docs:
        doc.metadata["source"] = filename

    # 2. Split Document into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 3. Add Chunks to Chroma Vector Store
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    vector_store.add_documents(chunks)

    return {"filename": filename, "chunks_added": len(chunks)}


def get_active_documents() -> dict:
    """Fetch list of indexed files and total chunk count from ChromaDB."""
    if not os.path.exists(CHROMA_DB_DIR):
        return {"files": [], "total_chunks": 0}

    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        data = vector_store.get()
        metadatas = data.get("metadatas", [])
        
        files = list({m.get("source") for m in metadatas if m and "source" in m})
        total_chunks = len(data.get("ids", []))

        return {"files": files, "total_chunks": total_chunks}
    except Exception:
        return {"files": [], "total_chunks": 0}


def clear_knowledge_base():
    """Reset and clear the Chroma vector database directory."""
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)