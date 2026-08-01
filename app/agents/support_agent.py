import logging
from app.agents.orchestrator import AgentOrchestrator  # type: ignore

logger = logging.getLogger(__name__)


class CustomerSupportAgent:
    """Main Entry Point for Enterprise Customer Support Agent System."""

    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        logger.info("CustomerSupportAgent initialized with Multi-Agent Orchestrator.")

    def answer_question(self, query: str, session_id: str = "default_session") -> str:
        """Delegates query execution to multi-agent orchestrator."""
        return self.orchestrator.execute_agent(query, session_id=session_id)