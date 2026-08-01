import os
import logging
from datetime import datetime, timezone
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import settings

logger = logging.getLogger(__name__)

class DocumentIngestor:
    """Enterprise Document Loader, Metadata Enricher, and Splitter Engine."""

    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def load_single_document(self, file_path: str) -> List[Document]:
        """Loads a single PDF or TXT file and enriches metadata."""
        docs = []
        try:
            filename = os.path.basename(file_path)
            
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                raw_docs = loader.load()
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                raw_docs = loader.load()
            else:
                return []

            # Enrich metadata for each page/doc with timezone-aware UTC timestamp
            for doc in raw_docs:
                doc.metadata["filename"] = filename
                doc.metadata["source"] = file_path
                doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()
                docs.append(doc)

        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {str(e)}")
            
        return docs

    def load_and_split(self) -> List[Document]:
        """Scans data directory, loads supported files, enriches metadata, and splits into chunks."""
        logger.info(f"Scanning data directory: '{self.data_dir}'")
        all_documents = []
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                docs = self.load_single_document(file_path)
                if docs:
                    all_documents.extend(docs)

        if not all_documents:
            logger.warning("No valid documents found in data directory.")
            return []

        chunks = self.splitter.split_documents(all_documents)
        logger.info(f"Processed {len(all_documents)} pages/files into {len(chunks)} metadata-enriched chunks.")
        return chunks