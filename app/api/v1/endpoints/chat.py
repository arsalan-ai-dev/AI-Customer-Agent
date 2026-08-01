from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.support_agent import CustomerSupportAgent

router = APIRouter()
agent = CustomerSupportAgent()

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    answer: str
    session_id: str

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise AI Customer Support Agent"
    }

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty."
        )
    
    try:
        session_id = request.session_id or "default_session"
        answer = agent.answer_question(request.question)
        return ChatResponse(answer=answer, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))