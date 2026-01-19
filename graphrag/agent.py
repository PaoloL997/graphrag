"""Agent module for GraphRAG workflow orchestration."""

import json

from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from graphrag.state import State
from graphrag.store import Store
from graphrag.draw import ContextFromDraw


load_dotenv()


GENERATE_RESPONSE = """
You are an AI assistant that helps people find information. 
Use the provided context to answer the question as best as you can. 
If the context is empty or not relevant, say this explicitly.
Always format your answer using Markdown.
Question: {query}
Context: {context}
"""

EVALUATE_CONTEXT = """
You are an Information Retrieval Specialist. Your task is to filter a list of documents based on their relevance to a specific user query.

Task:
- Evaluate each document provided in the context.
- Determine if the document contains information necessary to answer the user query.
- Identify the EXACT Primary Key (pk) for every relevant document from the context provided.

Output Requirements:
- Return ONLY a valid JSON list of strings containing the EXACT "pk" values of the relevant documents as they appear in the context.
- The pk values must match exactly what is shown before the colon in each document (e.g., if you see "doc_123: content...", return "doc_123").
- If no documents are relevant, return an empty list: [].
- Do not include any explanations, greetings, or additional text.
- Do not use placeholder values like "pk" - use the actual pk values from the documents.

Question: {query}
Context: {context}

Examples of correct output format:
["doc_001", "doc_005"]
["material_spec_123"]
[]
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
        self.draw = (
            ContextFromDraw()
        )  # TODO: la size minima per richiedere lo zooming deve essere parametrizzabile

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

    def _context_from_draw_node(self, state: State):
        if not state["context"] or len(state["context"]) == 0:
            return {"context": []}
        context = state["context"][:]  # Create a copy to avoid mutation issues
        for i, document in enumerate(context):
            if document.metadata["type"] == "draw":
                new_context = self.draw.run(document=document, query=state["query"])
                context[i].page_content = new_context
        return {"context": context}

    def _evaluate_node(self, state: State):
        """Evaluate and filter documents based on relevance to query."""
        if not state["context"]:
            return {"context": []}
        docs_text = "\n\n".join(
            f"{doc.metadata['pk']}: {doc.page_content}" for doc in state["context"]
        )
        evaluation_result = self.llm.invoke(
            EVALUATE_CONTEXT.format(query=state["query"], context=docs_text)
        )

        try:
            relevant_pks = json.loads(str(evaluation_result.content))

            filtered_context = [
                doc
                for doc in state["context"]
                if str(doc.metadata["pk"])
                in [str(pk) for pk in relevant_pks]  # Converti entrambi a string
            ]
            return {"context": filtered_context}
        except Exception as e:
            print(f"Error parsing evaluation result: {e}")
            return {"context": state["context"]}

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
            # TODO:la context_str deve essere costruita in maniera più completa, in base a type
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
        graph_builder.add_node("evaluate", self._evaluate_node)
        graph_builder.add_node("context_from_draw", self._context_from_draw_node)

        # TODO: da aggiungere nodo per creare context a partire da draw
        graph_builder.add_node("generate", self._get_response_node)

        # Add edges
        graph_builder.set_entry_point("retrieve")
        graph_builder.add_edge("retrieve", "evaluate")
        graph_builder.add_edge("evaluate", "context_from_draw")
        graph_builder.add_conditional_edges(
            "context_from_draw",
            self._route_retrieval,
            {"generate": "generate", END: END},
        )
        graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def get_summary(self):
        """Retrieve the summary entry from the store."""
        try:
            qresult = self.store.query(
                expression='namespace == "summary"', fields=["text"], limit=1
            )
            if qresult and len(qresult) > 0:
                return qresult[0]["text"]
            return None
        except Exception as e:
            print(f"Error retrieving summary: {e}")
            return None

    def run(self, query: str) -> str:
        # Aggiungi sempre la summary ad uno store
        if not self.get_summary():
            self.store.summarize()
        initial_state: State = {
            "query": query,
            "context": None,
            "response": None,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state.get("response", "")
