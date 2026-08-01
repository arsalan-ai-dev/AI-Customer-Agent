from langchain_groq import ChatGroq
from hybrid_retriever import HybridRetriever
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

retriever = HybridRetriever()

def run_agent(question: str) -> str:
    # Search for relevant documents
    docs = retriever.retrieve(question, top_k=3)
    
    if not docs:
        return "No relevant information found in the documents."
    
    # Combine the documents
    context = "\n\n".join(docs)
    
    # Create a prompt
    prompt = f"""You are an expert customer support agent. Use the following context to answer the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    
    # Get response from LLM
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    result = run_agent("What is this document about?")
    print(result)