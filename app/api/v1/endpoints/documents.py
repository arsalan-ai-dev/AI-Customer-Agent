import os
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.config.settings import settings
from app.rag.loaders import DocumentIngestor
from app.rag.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF or TXT file upload, saves it to data directory, 
    and triggers automatic vector database re-indexing.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File upload rejected: Unsupported extension '{file_ext}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_ext}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Save uploaded file to configured data directory
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    destination_path = os.path.join(settings.DATA_DIR, file.filename)
    
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File '{file.filename}' successfully saved to '{destination_path}'")
    except Exception as e:
        logger.error(f"Failed to save uploaded file '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded document."
        )
    finally:
        file.file.close()

    # Trigger dynamic ingestion pipeline
    try:
        logger.info("Triggering automatic dynamic re-indexing...")
        ingestor = DocumentIngestor()
        chunks = ingestor.load_and_split()
        
        if chunks:
            vector_manager = VectorStoreManager()
            vector_manager.create_vectorstore(chunks)
            logger.info(f"Successfully ingested '{file.filename}' into vector store.")
        else:
            logger.warning("No chunks extracted during upload ingestion.")
            
    except Exception as e:
        logger.error(f"Error during post-upload ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File saved, but indexing failed."
        )

    return {
        "message": f"File '{file.filename}' uploaded and indexed successfully.",
        "filename": file.filename,
        "status": "success"
    }