"""Agent module for GraphRAG workflow orchestration."""

from typing import Optional, Dict
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from graphrag.core.state import State
from graphrag.store.store import Store
from graphrag.memory.manager import MemoryManager
from graphrag.graph.nodes import (
    RetrieveNode,
    EvaluateNode,
    DrawNode,
    GenerateNode,
    RefineNode,
)
from graphrag.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class GraphRAG:
    """GraphRAG agent for orchestrating document retrieval and response generation."""

    def __init__(
        self,
        store: Store,
        llm: str,
        rerank: bool = False,
        draw_thinking_level: str = "low",
        draw_model: str = "gemini-3-flash-preview",
        draw_threshold_inches: float = 28.0,
    ):
        """Initialize the agent.

        Args:
            store: The Milvus store instance for document retrieval.
            llm: The language model to use for response generation.
            rerank: Whether to use a reranker during retrieval.
            draw_thinking_level: The thinking level for the TechDraw agent.
            draw_model: The Gemini model to use for the TechDraw agent.
            draw_threshold_inches: Minimum dimension to trigger zooming analysis.
        """
        self.store = store
        self.llm = init_chat_model(model=llm)
        self.memory_manager = MemoryManager(uri=self.store.uri)
        self.memory_manager._clear_redis()  # Clear short-term memory on initialization

        # Initialize nodes
        self.refine_node = RefineNode(
            init_chat_model(model="gpt-4.1-nano", temperature=0.5), self.memory_manager
        )
        self.retrieve_node = RetrieveNode(store, rerank)
        self.evaluate_node = EvaluateNode(self.llm)
        self.draw_node = DrawNode(
            thinking_level=draw_thinking_level,
            model=draw_model,
            threshold_inches=draw_threshold_inches,
        )
        self.generate_node = GenerateNode(self.llm, self.memory_manager)

        self.graph = self._compile_graph()

    def _route_retrieval(self, state: State) -> str:
        """Route based on retrieval results.

        Args:
            state: The current state with context.

        Returns:
            str: Next node to execute ('generate').
        """
        ctx = state.get("context")
        if not ctx or len(ctx) == 0:
            logger.info("No relevant context found for query: %s.", state.get("query"))
        return "generate"

    def _compile_graph(self):
        """Compile the LangGraph workflow.

        Returns:
            CompiledGraph: The compiled workflow graph.
        """
        graph_builder = StateGraph(State)

        graph_builder.add_node("refine", self.refine_node)
        graph_builder.add_node("retrieve", self.retrieve_node)
        graph_builder.add_node("evaluate", self.evaluate_node)
        graph_builder.add_node("context_from_draw", self.draw_node)
        graph_builder.add_node("generate", self.generate_node)

        graph_builder.set_entry_point("refine")
        graph_builder.add_edge("refine", "retrieve")
        graph_builder.add_edge("retrieve", "evaluate")
        graph_builder.add_edge("evaluate", "context_from_draw")
        graph_builder.add_conditional_edges(
            "context_from_draw",
            self._route_retrieval,
            {"generate": "generate", END: END},
        )
        graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def run(self, query: str, user_id: Optional[str] = None) -> Dict:
        """Run the GraphRAG workflow.

        Args:
            query: The user's query.
            user_id: Optional user identifier for memory management.

        Returns:
            dict: The final state containing the response.
        """
        initial_state: State = {
            "query": query,
            "refined_query": None,
            "context": None,
            "response": None,
            "user_id": user_id,
        }

        try:
            final_state = self.graph.invoke(initial_state)

            if user_id and final_state.get("response"):
                self.memory_manager.save(user_id, query, final_state["response"])
            return final_state

        except Exception as e:
            logger.error("Error during workflow execution: %s", e)
            return {
                "query": query,
                "refined_query": None,
                "context": None,
                "response": "An error occurred while processing your request.",
                "user_id": user_id,
            }
