import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app.config.settings import settings
from app.rag.vectorstore import VectorStoreManager
from app.rag.loaders import DocumentIngestor

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Enterprise Hybrid Context Retriever combining Dense Vector Search and Sparse BM25 Search."""

    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.vector_store = self.vector_manager.get_vectorstore()
        self.dense_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": settings.RETRIEVAL_TOP_K}
        )
        self.bm25_retriever = self._build_bm25_retriever()

    def _build_bm25_retriever(self):
        """Constructs BM25 Keyword Retriever from ingested documents."""
        try:
            ingestor = DocumentIngestor()
            chunks = ingestor.load_and_split()
            if chunks:
                bm25 = BM25Retriever.from_documents(chunks)
                bm25.k = settings.RETRIEVAL_TOP_K
                logger.info("BM25 Keyword Retriever initialized successfully.")
                return bm25
        except Exception as e:
            logger.warning(f"Could not initialize BM25 retriever: {str(e)}")
        return None

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieves and merges top relevant document chunks using hybrid search."""
        logger.info(f"Executing hybrid retrieval for query: '{query}'")
        
        # Dense Vector Search
        dense_docs = self.dense_retriever.invoke(query)
        
        # Sparse BM25 Search (if available)
        bm25_docs = []
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                logger.warning(f"BM25 execution skipped: {str(e)}")

        # Deduplicate and combine results while preserving rank
        seen_contents = set()
        combined_docs = []

        for doc in dense_docs + bm25_docs:
            content_hash = doc.page_content.strip()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                combined_docs.append(doc)

        # Return up to TOP_K results
        return combined_docs[:settings.RETRIEVAL_TOP_K]

    def get_formatted_context(self, query: str) -> str:
        """Retrieves and formats context string with metadata for the LLM prompt."""
        docs = self.get_relevant_documents(query)
        if not docs:
            logger.warning(f"No context retrieved for query: '{query}'")
            return "No relevant information found."

        formatted_chunks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "Unknown Source")
            formatted_chunks.append(f"[Source {i}: {source}]\n{doc.page_content}")

        return "\n\n".join(formatted_chunks)