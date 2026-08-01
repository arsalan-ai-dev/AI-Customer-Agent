import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SessionMemoryManager:
    """In-memory session history storage for multi-turn conversations."""

    def __init__(self, max_history_turns: int = 5):
        self.max_history_turns = max_history_turns
        # Dictionary mapping session_id -> List of (role, content) tuples
        self._sessions: Dict[str, List[Tuple[str, str]]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        """Appends a message (user/assistant) to session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        self._sessions[session_id].append((role, content))
        
        # Trim history if it exceeds max turns (each turn = 1 user + 1 assistant message)
        max_messages = self.max_history_turns * 2
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]
            
        logger.debug(f"Updated memory for session '{session_id}'. Total stored messages: {len(self._sessions[session_id])}")

    def get_formatted_history(self, session_id: str) -> str:
        """Formats conversation history for LLM prompt context."""
        history = self._sessions.get(session_id, [])
        if not history:
            return "No previous conversation history."
        
        formatted_turns = []
        for role, content in history:
            formatted_turns.append(f"{role.capitalize()}: {content}")
            
        return "\n".join(formatted_turns)

    def clear_session(self, session_id: str):
        """Clears memory for a specific session ID."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared memory session '{session_id}'")

# Global singleton memory instance
memory_manager = SessionMemoryManager(max_history_turns=5)