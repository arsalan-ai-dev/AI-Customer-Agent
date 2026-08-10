import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

try:
    from app.config.settings import settings
    CHROMA_DIR = getattr(settings, "CHROMA_DIR", "./chroma_db")
except Exception:
    CHROMA_DIR = "./chroma_db"

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Enterprise Vector Store Manager utilizing ChromaDB and FastEmbed Embeddings."""

    def __init__(self):
        self.persist_directory = CHROMA_DIR
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Creates and persists vector store from document chunks."""
        logger.info(
            f"Generating embeddings using FastEmbed (bge-small-en-v1.5) "
            f"and persisting to '{self.persist_directory}'..."
        )

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info("Vector database successfully built and persisted to disk.")
        return vectorstore

    def get_vectorstore(self) -> Chroma:
        """Loads existing persisted vector store from disk."""
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )