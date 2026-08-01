import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq  # 🎯 Pure Groq integration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Customer Agent", page_icon="🤖", layout="centered")
st.title("🤖 AI Customer Support Agent")
st.write("Ask anything based on the ingested documentation!")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

try:
    vector_store = load_vectorstore()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # ⚡ Initialize Groq Llama-3.1 Model
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.1-8b-instant",
        groq_api_key="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"
    )   

    template = """You are an expert customer support agent. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.

Context:
{context}

Question: {question}
Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(user_query)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Error initializing agent: {e}")