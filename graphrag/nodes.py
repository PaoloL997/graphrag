from graphrag.state import State
from graphrag.tools import check_relevance



def relevance_node(state: State):
    """Node that checks the relevance of the context to the query."""
    result = check_relevance.invoke(
        {"query": state["query"],
         "context": state["summary"]}
    )
    return {"relevant": result["relevant"]}