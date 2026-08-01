import logging
from typing import Dict, Any
from langchain_groq import ChatGroq
from app.config.settings import settings
from app.rag.retrievers import ContextRetriever
from app.agents.memory import memory_manager

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Enterprise Multi-Agent Router and Execution Orchestrator."""

    def __init__(self):
        self.retriever = ContextRetriever()
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
        )

    def classify_intent(self, query: str) -> str:
        """Classifies incoming user intent into POLICY, TECHNICAL, or GENERAL."""
        system_prompt = (
            "You are an Intent Classification Router. Categorize the user query into exactly ONE category:\n"
            "- POLICY: Queries regarding company policies, rules, refunds, terms, or incorporation documents.\n"
            "- TECHNICAL: Queries regarding errors, bugs, code issues, installation, or configuration.\n"
            "- GENERAL: Greetings, general conversation, or non-technical support.\n\n"
            "Respond with ONLY the category word: POLICY, TECHNICAL, or GENERAL."
        )

        try:
            response = self.llm.invoke(
                [("system", system_prompt), ("user", query)]
            )
            intent = response.content.strip().upper()
            if intent not in ["POLICY", "TECHNICAL", "GENERAL"]:
                intent = "POLICY"  # Safe default fallback
            logger.info(f"Query classified intent: '{intent}'")
            return intent
        except Exception as e:
            logger.warning(f"Intent classification failed: {str(e)}. Defaulting to POLICY.")
            return "POLICY"

    def execute_agent(self, query: str, session_id: str = "default_session") -> str:
        """Dynamically routes query to specialized agent prompt with history context."""
        intent = self.classify_intent(query)
        
        # Get formatted chat history from memory manager
        history = memory_manager.get_formatted_history(session_id)
        
        # Build specialized system prompts based on intent
        if intent == "POLICY":
            context = self.retriever.get_formatted_context(query)
            system_prompt = (
                "You are an Enterprise Policy Specialist Agent.\n"
                "Answer the user's question strictly using the provided context below.\n"
                "If the information is not present in the context, state politely that you do not have that policy on file.\n"
                "You have access to the conversation history below. Use it to answer follow-up questions.\n\n"
                f"--- CONTEXT ---\n{context}\n---------------\n\n"
                f"--- CHAT HISTORY ---\n{history}\n---------------"
            )

        elif intent == "TECHNICAL":
            context = self.retriever.get_formatted_context(query)
            system_prompt = (
                "You are a Senior Technical Support Specialist Agent.\n"
                "Provide step-by-step diagnostic and troubleshooting advice.\n"
                "Use the provided context and technical specifications where available.\n"
                "You have access to the conversation history below. Use it to answer follow-up questions.\n\n"
                f"--- CONTEXT ---\n{context}\n---------------\n\n"
                f"--- CHAT HISTORY ---\n{history}\n---------------"
            )

        else:  # GENERAL
            system_prompt = (
                "You are a friendly Enterprise AI Support Concierge.\n"
                "Provide helpful, concise, and professional responses to general inquiries.\n"
                "You have access to the conversation history below. Use it to answer any follow-up questions about previous messages.\n\n"
                f"--- CHAT HISTORY ---\n{history}\n---------------"
            )

        # Generate LLM response
        response = self.llm.invoke(
            [("system", system_prompt), ("user", query)]
        )
        answer = response.content.strip()

        # Update session memory
        memory_manager.add_message(session_id, "user", query)
        memory_manager.add_message(session_id, "assistant", answer)

        return answer