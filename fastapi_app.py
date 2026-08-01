from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from app.config.settings import settings
from app.config.logging_config import setup_logging
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.documents import router as doc_router

logger = setup_logging(settings.LOG_LEVEL if hasattr(settings, 'LOG_LEVEL') else "INFO")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}...")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Enterprise Multi-Agent RAG Customer Support API",
    lifespan=lifespan
)

# Include API Routers
app.include_router(chat_router, prefix="/api/v1", tags=["Chat & Support"])
app.include_router(doc_router, prefix="/api/v1/documents", tags=["Document Management"])

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)