import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Force load environment variables from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY is missing! Please ensure it is defined in your .env file."
    )

# 2. Initialize the shared LLM instance with the loaded API key
llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# -------------------------------------------------------------------
# Agent 1: Router / Intent Classifier
# Determines whether the question is general or requires document retrieval
# -------------------------------------------------------------------
router_prompt = ChatPromptTemplate.from_template(
    """You are an intent router for a customer support AI system.
Classify the user's input into one of two categories:
1. 'GENERAL': Greetings, small talk, or general questions not related to product docs.
2. 'SUPPORT': Questions asking about specific product details, policies, technical issues, or documentation.

Respond with ONLY one word: either 'GENERAL' or 'SUPPORT'.

User Input: {question}
Classification:"""
)

router_chain = router_prompt | llm | StrOutputParser()


# -------------------------------------------------------------------
# Agent 2: General Conversation Specialist
# Handles non-technical, conversational queries
# -------------------------------------------------------------------
general_prompt = ChatPromptTemplate.from_template(
    """You are a polite and helpful customer support assistant. 
Respond warmly and concisely to the customer's input.

User: {question}
Assistant:"""
)

general_agent_chain = general_prompt | llm | StrOutputParser()


# -------------------------------------------------------------------
# Agent 3: Technical / Document Support Specialist
# Formulates targeted answers based on retrieved context
# -------------------------------------------------------------------
support_prompt = ChatPromptTemplate.from_template(
    """You are an expert technical support specialist. Answer the user's question clearly based on customer support guidelines.
If the information requires specific internal database context, provide a clear and direct answer based on standard support protocol.

Question: {question}
Answer:"""
)

support_agent_chain = support_prompt | llm | StrOutputParser()


# -------------------------------------------------------------------
# Orchestrator Function (Called by FastAPI)
# -------------------------------------------------------------------
def run_agent(question: str) -> str:
    """
    Main entry point for multi-agent execution called by FastAPI /chat endpoint.
    """
    try:
        # Step 1: Route user query
        intent = router_chain.invoke({"question": question}).strip().upper()
        
        # Step 2: Delegate to specialized agent based on routing decision
        if "GENERAL" in intent:
            response = general_agent_chain.invoke({"question": question})
        else:
            response = support_agent_chain.invoke({"question": question})
            
        return response

    except Exception as e:
        # Graceful error fallback for API or LLM issues
        return f"Agent execution error: {str(e)}"