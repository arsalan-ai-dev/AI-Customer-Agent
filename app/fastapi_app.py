import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from hybrid_retriever import HybridRetriever
from multi_agent import run_agent

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

hybrid_retriever = HybridRetriever()

def hybrid_format_docs(query, top_k=3):
    docs = hybrid_retriever.retrieve(query, top_k=top_k)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(docs)

llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

template = """You are an expert customer support agent. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def get_context(query):
    return hybrid_format_docs(query)

app = FastAPI(title="AI Customer Agent API", description="Backend API for n8n Automation")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        if not request.question or request.question.strip() == "":
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        answer = run_agent(request.question)
        return QueryResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Agent is ready to take questions!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)