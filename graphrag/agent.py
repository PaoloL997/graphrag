"""Agent module for GraphRAG workflow orchestration."""

from langgraph.graph import StateGraph, END
from graphrag.nodes import relevance_node
from graphrag.state import State


def route_relevance(state: State):
    """Route Agent based on relevance tool output.

    Args:
        state: The current state of the graph.

    Returns:
        str: The next node to execute ('continue') or END.
    """
    if state["relevant"]:
        return "continue"
    return END


graph_builder = StateGraph(State)
graph_builder.add_node("check_relevance", relevance_node)

graph_builder.add_conditional_edges(
    "check_relevance", route_relevance, {"continue": END, END: END}
)

graph = graph_builder.compile()
