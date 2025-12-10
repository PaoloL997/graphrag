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
Question: {query}
Context: {context}
"""

CHECK_RELEVANCE = """
You are an evaluator tasked with determining whether a user's question is relevant to a summary of a collection.
Here is the summary of the collection: {collection_summary}
Here is the user's question: {question}
If the question is related to the content or main concepts present in the collection summary, consider it relevant.
Provide a binary 'yes' or 'no' score to indicate whether the question is relevant to the collection.
"""


# TODO: devi per forza aggiungere un nodo per controllare se la domanda è inerente o meno al contesto


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

    def _relevant_node(self, state: State):
        """Check if the query is relevant to the document collection summary.

        Args:
            state: The current state containing the query and summary.
        Returns:
            dict: Updated state with relevance boolean.
        """
        summary = state["summary"]  # TODO: da implementare
        query = state["query"]
        response = self.llm.invoke(
            CHECK_RELEVANCE.format(collection_summary=summary, question=query)
        ).content
        return {"relevant": "yes" in str(response).strip().lower(), "summary": summary}

    def _get_response_node(self, state: State):
        """Generate response using LLM based on query and context.

        Args:
            state: The current state containing query and context.

        Returns:
            dict: Updated state with generated response.
        """
        # Check if context is None
        if not state["context"]:
            # TODO: da implementare
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

    def _route_relevant(self, state: State):
        if state["relevant"]:
            return "retrieve"
        return END

    def _compile_graph(self):
        """Compile the LangGraph workflow."""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("relevant", self._relevant_node)
        graph_builder.add_node("retrieve", self._retrieve_node)
        graph_builder.add_node("generate", self._get_response_node)

        # Add edges
        graph_builder.set_entry_point("relevant")
        graph_builder.add_conditional_edges(
            "relevant", self._route_relevant, {"retrieve": "retrieve", END: END}
        )
        graph_builder.add_conditional_edges(
            "retrieve", self._route_retrieval, {"generate": "generate", END: END}
        )
        graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def run(self, query: str) -> str:
        initial_state: State = {
            "query": query,
            "context": None,
            "response": None,
            "relevant": None,
            "summary": None,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state["response"]
