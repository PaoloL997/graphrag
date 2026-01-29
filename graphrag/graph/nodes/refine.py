from langchain_core.language_models import BaseChatModel
from graphrag.core.state import State
from graphrag.memory.manager import MemoryManager
from graphrag.config.prompts import REFINE_QUERY_PROMPT


class RefineNode:
    """Refine User query."""

    def __init__(self, llm: BaseChatModel, memory_manager: MemoryManager):
        """Initialize the RefineNode."""
        self.llm = llm
        self.memory_manager = memory_manager

    def __call__(self, state: State):
        user_id = state.get("user_id")
        if not user_id:
            return {
                "refined_query": state.get("query")
            }  # Skip refinement if no user_id
        cache = self.memory_manager.get_or_create(user_id)
        short_memory = cache.short_term_memory()
        prompt = REFINE_QUERY_PROMPT.format(
            history=short_memory
            if short_memory
            else "No previous conversation history.",
            current_question=state.get("query"),
        )
        response = self.llm.invoke(prompt)
        if not response or not response.content:
            refined_query = state.get("query")
        else:
            refined_query = str(response.content).strip()
        return {"refined_query": refined_query}
