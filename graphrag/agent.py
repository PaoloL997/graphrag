
from langgraph.graph import StateGraph, END
from graphrag.nodes import relevance_node
from graphrag.state import State

def route_relevance(state: State):
    if state["relevant"]:
        return "continue"
    return END


graph_builder = StateGraph(State)
graph_builder.add_node("check_relevance", relevance_node)

graph_builder.add_conditional_edges(
    "check_relevance",
    route_relevance,
    {
        "continue": END,
        END: END
    }
)

graph = graph_builder.compile()




