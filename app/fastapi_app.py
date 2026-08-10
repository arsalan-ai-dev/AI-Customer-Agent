import os
import shutil
import tempfile
import traceback
from typing import List, AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------
# ⚡ FastAPI App Initialization
# ---------------------------------------------------
app = FastAPI(
    title="AI Customer Support Agent Backend",
    description="FastAPI Microservice with FlashRank Hybrid RAG Search & Safe Phoenix Telemetry.",
    version="3.5.0"
)

# ---------------------------------------------------
# 📊 Arize Phoenix Telemetry Tracing (Safe Non-Blocking)
# ---------------------------------------------------
try:
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006") + "/v1/traces"
    tracer_provider = register(endpoint=phoenix_endpoint, auto_instrument=False)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("🚀 Arize Phoenix Tracing initialized successfully!")
except Exception as e:
    print(f"⚠️ Arize Phoenix telemetry notice (non-blocking): {e}")

# ---------------------------------------------------
# 📦 LangChain Imports & Dynamic Version Fallbacks
# ---------------------------------------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Safe EnsembleRetriever import resolution across package versions
try:
    from langchain.retrievers import EnsembleRetriever  # type: ignore
except ImportError:
    try:
        from langchain.retrievers.ensemble_retriever import EnsembleRetriever  # type: ignore
    except ImportError:
        try:
            from langchain_classic.retrievers import EnsembleRetriever  # type: ignore
        except ImportError:
            from langchain_community.retrievers.ensemble_retriever import EnsembleRetriever  # type: ignore

# Safe ContextualCompressionRetriever import resolution across package versions
try:
    from langchain.retrievers import ContextualCompressionRetriever  # type: ignore
except ImportError:
    try:
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever  # type: ignore
    except ImportError:
        try:
            from langchain_classic.retrievers import ContextualCompressionRetriever  # type: ignore
        except ImportError:
            from langchain_community.retrievers import ContextualCompressionRetriever  # type: ignore

from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# ---------------------------------------------------
# ⚙️ Configuration & Schemas
# ---------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = []

all_documents = []  # Memory document chunk cache for BM25 search
indexed_filenames = set()  # Track unique active files

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN
)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    streaming=True
)

# FlashRank Cross-Encoder Compressor (rescores initial hybrid candidates down to top 3)
compressor = FlashrankRerank(top_n=3)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

system_prompt = (
    "You are an expert AI customer support assistant.\n"
    "Answer the user's question clearly and accurately using ONLY the context provided below. "
    "If the context does not contain the answer, state politely that you don't know.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

def get_current_retriever():
    """
    Dynamically creates an EnsembleRetriever (BM25 + Chroma) 
    and wraps it with FlashRank Cross-Encoder compression for precision reranking.
    """
    dense = vector_store.as_retriever(search_kwargs={"k": 8})
    
    if len(all_documents) > 0:
        bm25 = BM25Retriever.from_documents(all_documents)
        bm25.k = 8
        base_retriever = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])
    else:
        base_retriever = dense

    # Apply FlashRank Cross-Encoder compression to re-rank chunks
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return reranked_retriever

# ---------------------------------------------------
# 🌐 FastAPI Endpoints
# ---------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "vector_store_ready": vector_store is not None,
        "indexed_files": list(indexed_filenames),
        "total_chunks": len(all_documents),
        "reranker": "FlashRank Cross-Encoder active"
    }

@app.get("/documents")
def get_documents():
    """Returns active file list and chunk summary."""
    return {
        "files": list(indexed_filenames),
        "total_chunks": len(all_documents)
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global all_documents, indexed_filenames
    
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path) if file.filename.endswith(".pdf") else TextLoader(tmp_path)
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["source"] = file.filename

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # Synchronize vector storage and memory BM25 store
        vector_store.add_documents(chunks)
        all_documents.extend(chunks)
        indexed_filenames.add(file.filename)

        return {
            "status": "success", 
            "filename": file.filename, 
            "chunks_added": len(chunks),
            "total_files": len(indexed_filenames)
        }

    except Exception as e:
        print("❌ Upload Error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.delete("/clear")
def clear_knowledge_base():
    """Wipes vector store collection and resets memory BM25 index."""
    global all_documents, indexed_filenames, vector_store
    try:
        try:
            vector_store._collection.delete(where={})
        except Exception:
            pass

        all_documents = []
        indexed_filenames = set()

        return {"status": "success", "message": "Knowledge Base reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {str(e)}")

async def generate_chat_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    try:
        chat_history = [
            HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content)
            for msg in request.history
        ]

        current_retriever = get_current_retriever()
        history_aware_retriever = create_history_aware_retriever(llm, current_retriever, contextualize_q_prompt)
        active_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        async for chunk in active_rag_chain.astream({
            "input": request.question,
            "chat_history": chat_history
        }):
            if "answer" in chunk:
                yield chunk["answer"]

    except Exception as e:
        yield f"\n[Error during streaming: {str(e)}]"

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(generate_chat_stream(request), media_type="text/plain")