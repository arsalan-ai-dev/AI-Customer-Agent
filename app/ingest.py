import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def main():
    print("🚀 Starting data ingestion pipeline...")
    
    data_dir = os.getenv("DATA_DIR", "./data")
    chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    
    # 1. Load Documents
    print(f"📁 Loading documents from: {data_dir}")
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} total pages from PDFs.")
    
    # 2. Split Documents
    print("✂️ Splitting documents into strategic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Created {len(chunks)} text chunks.")
    
    # 3. Initialize Free Open-Source Embeddings
    print("🤖 Initializing Free HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Create and Persist Vector Store
    print(f"💾 Saving vector store to: {chroma_db_dir}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_db_dir
    )
    print("✅ Data ingestion successfully completed locally!")

if __name__ == "__main__":
    main()