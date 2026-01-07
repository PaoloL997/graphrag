"""Agent module for GraphRAG workflow orchestration."""

from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from graphrag.state import State
from graphrag.store import Store


load_dotenv()


GENERATE_RESPONSE = """
You are an AI assistant that helps people find information. 
Use the provided context to answer the question as best as you can. 
If the context is empty or not relevant, say this explicitly.
Always format your answer using Markdown.
Question: {query}
Context: {context}
"""


class GraphRAG:
    """GraphRAG agent for orchestrating document retrieval and response generation."""

    def __init__(self, store: Store, llm: str, rerank: bool = False):
        """Initialize the agent.

        Args:
            store: The Milvus store instance for document retrieval.
            llm: The language model to use for response generation.
            rerank: Whether to use a reranker during retrieval.
        """
        self.store = store
        self.llm = init_chat_model(model=llm)
        self.graph = self._compile_graph()
        self.rerank = rerank

    def _retrieve_node(self, state: State):
        """Retrieve documents from the Milvus store based on query.

        Args:
            state: The current state containing the query.

        Returns:
            dict: Updated state with retrieved context.
        """
        if not self.rerank:
            context = self.store.retrieve(state["query"])
        else:
            context = self.store.retrieve_with_reranker(state["query"])
        return {"context": context}

    def _get_response_node(self, state: State):
        """Generate response using LLM based on query and context.

        Args:
            state: The current state containing query and context.

        Returns:
            dict: Updated state with generated response.
        """
        # Check if context is None
        if not state["context"]:
            context_str = ""
        else:
            context_str = "\n".join([doc.page_content for doc in state["context"]])
        response = self.llm.invoke(
            GENERATE_RESPONSE.format(query=state["query"], context=context_str)
        )

        return {"response": response.content}

    def _route_retrieval(self, state: State):
        """Route based on retrieval results.

        Args:
            state: The current state with context.

        Returns:
            str: Next node to execute ('generate') or END.
        """
        if state["context"] and len(state["context"]) > 0:
            return "generate"
        return END

    def _compile_graph(self):
        """Compile the LangGraph workflow."""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("retrieve", self._retrieve_node)
        graph_builder.add_node("generate", self._get_response_node)

        # Add edges
        graph_builder.set_entry_point("retrieve")
        graph_builder.add_conditional_edges(
            "retrieve", self._route_retrieval, {"generate": "generate", END: END}
        )
        graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def get_summary(self):
        """Retrieve the summary entry from the store."""
        try:
            qresult = self.store.query(
                expression='namespace == "summary"', fields=["text"], limit=1
            )
            return qresult[0]["text"]
        except Exception as e:
            print(f"Error retrieving summary: {e}")
            return None

    def run(self, query: str) -> str:
        summary = self.get_summary()
        if not summary:
            self.store.summarize()
            summary = self.get_summary()

        initial_state: State = {
            "query": query,
            "context": None,
            "response": None,
            # "relevant": None,
            "summary": summary,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state
