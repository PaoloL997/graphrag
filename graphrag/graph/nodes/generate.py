"""Response generation node."""

from typing import Dict, Optional, List
from langchain_core.documents import Document

from graphrag.config.prompts import GENERATE_RESPONSE_PROMPT
from graphrag.core.state import State
from graphrag.memory.manager import MemoryManager
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class GenerateNode:
    """Node for generating responses using LLM."""

    def __init__(self, llm, memory_manager: MemoryManager):
        """Initialize the generation node.

        Args:
            llm: The language model for generation.
            memory_manager: Memory manager for retrieving conversation history.
        """
        self.llm = llm
        self.memory_manager = memory_manager

    def __call__(self, state: State) -> Dict:
        """Generate response using LLM based on query and context.

        Args:
            state: The current state containing query and context.

        Returns:
            dict: Updated state with generated response.
        """
        context_str = self._prepare_context_string(state.get("context"))
        query = (
            str(state["refined_query"])
            if state.get("refined_query")
            else state["query"]
        )
        memory_str = self.memory_manager.get_memory_string(
            user_id=state.get("user_id"),
            query=query,
        )

        try:
            response = self.llm.invoke(
                GENERATE_RESPONSE_PROMPT.format(
                    query=state["refined_query"],
                    context=context_str,
                    memory=memory_str,
                )
            )

            logger.info("Generated response for query: %s", state["query"])
            return {"response": response.content}

        except Exception as e:
            logger.error("Error generating response: %s", e)
            return {
                "response": "I apologize, but I encountered an error while "
                "generating a response. Please try again."
            }

    def _prepare_context_string(self, context: Optional[List[Document]]) -> str:
        """Prepare context string from documents.

        Args:
            context: List of documents or None.

        Returns:
            str: Formatted context string.
        """
        if not context:
            return "No relevant context found."

        return "\n\n".join([doc.page_content for doc in context])
