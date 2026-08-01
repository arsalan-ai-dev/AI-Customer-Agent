from app.rag.loaders import DocumentIngestor
from app.rag.vectorstore import VectorStoreManager

class IngestionService:
    """Service layer that coordinates document loading, splitting, and vector database persistence."""

    def __init__(self):
        self.ingestor = DocumentIngestor()
        self.vector_manager = VectorStoreManager()

    def run_pipeline(self):
        """Executes the full document ingestion pipeline."""
        print("📁 Starting document ingestion pipeline...")
        
        # 1. Load & split documents
        chunks = self.ingestor.load_and_split()
        if not chunks:
            print("❌ Pipeline stopped: No document chunks were generated.")
            return False

        # 2. Persist to ChromaDB
        self.vector_manager.create_vectorstore(chunks)
        print("🎉 Ingestion Pipeline completed successfully!")
        return True