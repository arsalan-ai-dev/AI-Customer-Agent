import os
import sys
import tempfile
import traceback
import streamlit as st

# Ensure root and subdirectories are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path in (current_dir, parent_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import backend modules directly
try:
    from multi_agent import run_agent
except ImportError:
    from app.multi_agent import run_agent

try:
    from services.ingestion import process_and_ingest_document, get_active_documents, clear_knowledge_base
except ImportError:
    from app.services.ingestion import process_and_ingest_document, get_active_documents, clear_knowledge_base

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
                # Create temp directory to keep original filename and extension
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                try:
                    # Write uploaded bytes to file preserving extension
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    print(f"[INGEST LOG] Processing file: {temp_path}")
                    
                    # Execute direct ingestion
                    result = process_and_ingest_document(temp_path)
                    print(f"[INGEST LOG] Result: {result}")

                    chunks = result.get("chunks_added", "N/A") if isinstance(result, dict) else "N/A"
                    st.sidebar.success(f"Ingested **{uploaded_file.name}** ({chunks} chunks added)!")
                    
                except Exception as e:
                    err_msg = traceback.format_exc()
                    print(f"[INGEST ERROR]\n{err_msg}")
                    st.sidebar.error(f"Ingestion Failed: {str(e)}")
                finally:
                    # Cleanup local temp files
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)

    st.markdown("---")
    
    # Active Files Inspection
    st.subheader("📚 Active Indexed Documents")
    try:
        doc_info = get_active_documents()
        if doc_info and doc_info.get("files"):
            for file in doc_info["files"]:
                st.markdown(f"• `{file}`")
            st.caption(f"Total Chunks Indexed: **{doc_info.get('total_chunks', 0)}**")
        else:
            st.info("No documents active in Knowledge Base.")
    except Exception as e:
        st.error(f"Error fetching active docs: {e}")

    st.markdown("---")

    # Clear Knowledge Base Button
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True, type="secondary"):
        try:
            clear_knowledge_base()
            st.session_state.messages = []
            st.success("Knowledge Base cleared!")
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
                response_text = run_agent(prompt)
            message_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            message_placeholder.error(f"Agent Execution Error: {e}")