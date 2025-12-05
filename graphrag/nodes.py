"""Node implementations for GraphRAG workflow."""

from graphrag.state import State
from graphrag.tools import check_relevance


def relevance_node(state: State):
    """Node that checks the relevance of the context to the query.

    Args:
        state: The current state containing at least query and summary.

    Returns:
        dict: Dictionary with 'relevant' key containing boolean result.
    """
    result = check_relevance.invoke(
        {"query": state["query"], "context": state["summary"]}
    )
    return {"relevant": result["relevant"]}
