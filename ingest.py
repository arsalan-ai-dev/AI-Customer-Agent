import logging
from app.config.logging_config import setup_logging
from app.rag.loaders import DocumentIngestor
from app.rag.vectorstore import VectorStoreManager

logger = setup_logging("INFO")

def run_ingestion():
    """Executes the complete document ingestion, chunking, and embedding pipeline."""
    logger.info("Starting Enterprise Document Ingestion Pipeline...")
    
    ingestor = DocumentIngestor()
    chunks = ingestor.load_and_split()
    
    if not chunks:
        logger.warning("Ingestion aborted: No chunks generated.")
        return

    logger.info("Generating embeddings and persisting chunks to ChromaDB...")
    vector_manager = VectorStoreManager()
    vector_manager.create_vectorstore(chunks)
    logger.info("Ingestion Pipeline completed successfully!")

if __name__ == "__main__":
    run_ingestion()