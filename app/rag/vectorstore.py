import logging
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Enterprise Vector Store Manager utilizing ChromaDB and HuggingFace Embeddings."""

    def __init__(self):
        self.persist_directory = settings.CHROMA_DIR
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Creates and persists vector store from document chunks."""
        logger.info(
            f"Generating embeddings using '{settings.EMBEDDING_MODEL_NAME}' "
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