"""Memory management module for GraphRAG."""

from typing import Optional, Dict
from graphrag.memory.user_memory import UserMemory
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Manages user memory instances and operations."""

    def __init__(self, uri: str):
        """Initialize the memory manager.

        Args:
            uri: The URI for the memory store.
        """
        self.uri = uri
        self._memory_cache: Dict[str, UserMemory] = {}

    def get_or_create(self, user_id: str) -> UserMemory:
        """Get or create a UserMemory instance for the given user.

        Args:
            user_id: The user identifier.

        Returns:
            UserMemory: The memory instance for the user.
        """
        if user_id not in self._memory_cache:
            self._memory_cache[user_id] = UserMemory(
                uri=self.uri,
                user=user_id,
            )
        return self._memory_cache[user_id]

    def get_memory_string(self, user_id: Optional[str], query: str) -> str:
        """Retrieve memory string for the user.

        Args:
            user_id: The user identifier or None.
            query: The current query.

        Returns:
            str: Formatted memory string.
        """
        if not user_id:
            return "No previous conversation history."

        try:
            memory = self.get_or_create(user_id)
            memory_str = memory.get(query)
            return memory_str or "No previous conversation history."

        except Exception as e:
            logger.error("Error retrieving memory: %s", e)
            return "No previous conversation history."

    def save(self, user_id: str, query: str, response: str) -> None:
        """Save query-response pair to user memory.

        Args:
            user_id: The user identifier.
            query: The user's query.
            response: The generated response.
        """
        try:
            memory = self.get_or_create(user_id)
            memory.add(query=query, response=response)
            logger.debug("Saved query-response to memory for user: %s", user_id)

        except Exception as e:
            logger.error("Error saving to memory for user %s: %s", user_id, e)

    def clear(self, user_id: str) -> bool:
        """Clear memory for a specific user.

        Args:
            user_id: The user identifier.

        Returns:
            bool: True if memory was cleared successfully, False otherwise.
        """
        try:
            if user_id in self._memory_cache:
                memory = self._memory_cache[user_id]
                memory.delete()
                del self._memory_cache[user_id]
                logger.info("Cleared memory for user: %s", user_id)
                return True

            logger.warning("No memory found for user: %s", user_id)
            return False

        except Exception as e:
            logger.error("Error clearing memory for user %s: %s", user_id, e)
            return False
