import os
import uuid
import requests
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="Enterprise AI Customer Support Agent",
    page_icon="🤖",
    layout="wide"
)

# Backend API Endpoints (environment variable fallback for Docker networking)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/v1")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/documents/upload"

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Header Section
st.title("🤖 Enterprise AI Customer Support Agent")
st.caption("Powered by Multi-Agent Dynamic Orchestration (Groq) & Hybrid Retrieval (Dense Vector + BM25)")

# Sidebar: Document Upload & Session Details
with st.sidebar:
    st.header("📌 Session Details")
    st.text(f"Session ID:\n{st.session_state.session_id}")
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.markdown("---")
    st.header("📄 Knowledge Base Upload")
    uploaded_file = st.file_uploader("Upload Policy or Spec Document (.pdf, .txt)", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Upload to Knowledge Base"):
            with st.spinner("Ingesting document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(UPLOAD_ENDPOINT, files=files)
                    
                    if response.status_code == 200:
                        st.success(f"Successfully uploaded: {uploaded_file.name}")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Upload failed')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {str(e)}")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Response Loop
if user_query := st.chat_input("Ask a question about policies, technical troubleshooting, or general info..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "question": user_query,
                    "session_id": st.session_state.session_id
                }
                response = requests.post(CHAT_ENDPOINT, json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No response generated.")
                else:
                    answer = f"⚠️ Server Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}"
            except Exception as e:
                answer = f"⚠️ Could not reach backend server: {str(e)}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})