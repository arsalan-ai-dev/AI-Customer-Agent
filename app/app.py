import os
import sys
import tempfile
import streamlit as st

# Inject workspace directories into sys.path to resolve module imports on Render
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path in (current_dir, parent_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

# Direct module imports with fallback resolution
try:
    from multi_agent import run_agent
except ImportError:
    try:
        from app.multi_agent import run_agent
    except ImportError:
        def run_agent(query):
            return "Multi-agent module could not be loaded."

try:
    from services.ingestion import process_and_ingest_document, get_active_documents, clear_knowledge_base
except ImportError:
    try:
        from app.services.ingestion import process_and_ingest_document, get_active_documents, clear_knowledge_base
    except ImportError:
        def process_and_ingest_document(file_path):
            return {"filename": os.path.basename(file_path), "chunks_added": 0}
        def get_active_documents():
            return {"files": [], "total_chunks": 0}
        def clear_knowledge_base():
            pass

st.set_page_config(
    page_title="Enterprise Customer Support Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Customer Support Agent")
st.caption("Containerized Hybrid Retrieval-Augmented Generation (RAG) Architecture")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# 📁 Sidebar Management
# ---------------------------------------------------
with st.sidebar:
    st.header("📄 Knowledge Base Management")
    
    uploaded_file = st.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("📥 Process & Ingest Document", use_container_width=True):
            with st.spinner("Ingesting and indexing document..."):
                try:
                    # Save temporary file locally for vector processing
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    # Execute ingestion service directly in Python
                    result = process_and_ingest_document(temp_path)
                    
                    # Cleanup temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    chunks = result.get("chunks_added", "N/A") if isinstance(result, dict) else "N/A"
                    st.success(f"Successfully ingested **{uploaded_file.name}** ({chunks} chunks added)!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion Error: {e}")

    st.markdown("---")
    
    # Active Files Inspection
    st.subheader("📚 Active Indexed Documents")
    try:
        doc_info = get_active_documents()
    except Exception:
        doc_info = {"files": [], "total_chunks": 0}
    
    if doc_info and doc_info.get("files"):
        for file in doc_info["files"]:
            st.markdown(f"• `{file}`")
        st.caption(f"Total Chunks Indexed: **{doc_info.get('total_chunks', 0)}**")
    else:
        st.info("No documents active in Knowledge Base.")

    st.markdown("---")

    # Clear Knowledge Base Button
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True, type="secondary"):
        try:
            clear_knowledge_base()
            st.session_state.messages = []
            st.success("Knowledge Base cleared and chat reset!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing base: {e}")

# ---------------------------------------------------
# 💬 Chat Interface
# ---------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about your knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Analyzing query with multi-agent system..."):
                # Run the LangGraph multi-agent pipeline directly in-process
                response_text = run_agent(prompt)
                
            message_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            message_placeholder.error(f"Agent Execution Error: {e}")