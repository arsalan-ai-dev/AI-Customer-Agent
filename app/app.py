import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

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

# Fetch active documents from backend
def fetch_active_docs():
    try:
        res = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"files": [], "total_chunks": 0}

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
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Successfully ingested **{data['filename']}** ({data['chunks_added']} chunks added)!")
                        st.rerun()
                    else:
                        st.error(f"Failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.markdown("---")
    
    # Active Files Inspection
    st.subheader("📚 Active Indexed Documents")
    doc_info = fetch_active_docs()
    
    if doc_info["files"]:
        for file in doc_info["files"]:
            st.markdown(f"• `{file}`")
        st.caption(f"Total Chunks Indexed: **{doc_info['total_chunks']}**")
    else:
        st.info("No documents active in Knowledge Base.")

    st.markdown("---")

    # Clear Knowledge Base Button
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True, type="secondary"):
        try:
            res = requests.delete(f"{BACKEND_URL}/clear", timeout=10)
            if res.status_code == 200:
                st.session_state.messages = []
                st.success("Knowledge Base cleared and chat reset!")
                st.rerun()
            else:
                st.error("Failed to clear Knowledge Base.")
        except Exception as e:
            st.error(f"Error: {e}")

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
        full_response = ""

        try:
            payload = {
                "question": prompt,
                "history": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
            }

            with requests.post(
                f"{BACKEND_URL}/chat/stream",
                json=payload,
                stream=True,
                timeout=60
            ) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.error(f"Backend error: {response.status_code}")
        except Exception as e:
            message_placeholder.error(f"Error connecting to server: {e}")

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})