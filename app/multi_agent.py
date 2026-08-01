from typing import Literal, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
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

@tool
def search_documents(query: str) -> str:
    """Search the document database for relevant information."""
    docs = retriever.retrieve(query, top_k=3)
    if not docs:
        return "No relevant information found."
    return "\n\n".join(docs)

tools = [search_documents]
tool_node = ToolNode(tools)

def supervisor_agent(state: MessagesState) -> Dict[str, Any]:
    """Supervisor decides whether to search or end."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    if "search" in last_message.lower() or "find" in last_message.lower():
        return {"next": "researcher"}
    else:
        return {"next": "validator"}

def researcher_agent(state: MessagesState) -> Dict[str, Any]:
    """Researcher searches documents and returns findings."""
    messages = state["messages"]
    query = messages[-1].content if messages else ""
    
    docs = retriever.retrieve(query, top_k=3)
    if docs:
        context = "\n\n".join(docs)
        response = llm.invoke(f"Based on these documents, answer the user's question: {query}\n\nDocuments: {context}")
        return {"messages": [{"role": "assistant", "content": response.content}]}
    else:
        return {"messages": [{"role": "assistant", "content": "I couldn't find relevant information."}]}

def validator_agent(state: MessagesState) -> Dict[str, Any]:
    """Validator checks if the answer is correct and complete."""
    messages = state["messages"]
    last_response = messages[-1].content if messages else ""
    
    validation_prompt = f"""Check if this answer is accurate and complete based on the context.
    If it's inaccurate or incomplete, say "HALLUCINATION DETECTED" and explain why.
    If it's accurate, say "VALID" and summarize why.
    
    Answer: {last_response}
    """
    
    validation = llm.invoke(validation_prompt)
    return {"messages": [{"role": "validator", "content": validation.content}]}

def should_continue(state: Dict[str, Any]) -> Literal["researcher", "validator", END]:
    """Determine which node to go to next."""
    if state.get("next") == "researcher":
        return "researcher"
    elif state.get("next") == "validator":
        return "validator"
    else:
        return END

workflow = StateGraph(MessagesState)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("validator", validator_agent)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", should_continue)
workflow.add_edge("researcher", "validator")
workflow.add_edge("validator", END)

app = workflow.compile()

def run_agent(question: str) -> str:
    """Run the multi-agent system."""
    result = app.invoke({"messages": [{"role": "user", "content": question}]})
    
    for msg in result["messages"]:
        if msg["role"] == "assistant":
            return msg["content"]
    
    return "No response generated."

if __name__ == "__main__":
    response = run_agent("What is the purpose of this document?")
    print("Agent Response:", response)