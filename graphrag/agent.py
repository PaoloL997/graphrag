"""Agent module for GraphRAG workflow orchestration."""

import json
from typing import Optional

from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from graphrag.state import State
from graphrag.store import Store
from graphrag.draw import ContextFromDraw
from graphrag.logger import get_logger
from graphrag.memory import UserMemory

load_dotenv()

logger = get_logger(__name__)

GENERATE_RESPONSE = """
# Role
You are a helpful and natural AI Assistant. Your goal is to provide accurate answers by integrating retrieved technical information with the ongoing conversation history.

# Data Sources
### Primary Context (Knowledge Base)
{context}

### Conversational Memory (Past Interactions)
{memory}

# Guidelines
1. **Primary Source:** Use the "Primary Context" as your main factual reference.
2. **Context Integration:** Use "Conversational Memory" to maintain flow and personalization.
3. **Natural Language (CRITICAL):** Do NOT use phrases like "Based on the context provided," "According to the documents," or "In the memory." Speak directly to the user as a knowledgeable partner. 
4. **Authenticity:** If the information is not available in either source, politely inform the user without sounding mechanical.
5. **Formatting:** Use Markdown (bolding, lists) for clarity, but keep the prose conversational.

# User Query
Question: {query}

# Response
(Provide a direct, natural answer without referencing your internal data sources)
"""

EVALUATE_CONTEXT = """
You are an Information Retrieval Specialist. Your task is to filter a list of documents based on their relevance to a specific user query.

Task:
- Evaluate each document provided in the context.
- Determine if the document contains information necessary to answer the user query.
- If multiple documents have the type "draw", you must select ONLY the most significant or representative one. Do not include more than one document of type "draw" in the final list.
- Identify the EXACT Primary Key (pk) for every relevant document selected from the context provided.

Output Requirements:
- Return ONLY a valid JSON list of strings containing the EXACT "pk" values of the relevant documents as they appear in the context.
- The pk values must match exactly what is shown before the colon in each document (e.g., if you see "doc_123: content...", return "doc_123").
- If no documents are relevant, return an empty list: [].
- Do not include any explanations, greetings, or additional text.

Question: {query}
Context: {context}

Examples of correct output format:
["doc_001"]
["material_spec_123", "drawing_ref_02"]
[]
"""


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
            draw_thinking_level: The thinking level for the TechDraw agent ("low", "medium", "high").
            draw_model: The Gemini model to use for the TechDraw agent.
            draw_threshold_inches: Minimum dimension (in inches) to trigger zooming analysis.
        """
        self.store = store
        self.llm = init_chat_model(model=llm)
        self.graph = self._compile_graph()
        self.rerank = rerank
        self.draw = ContextFromDraw(
            thinking_level=draw_thinking_level,
            model=draw_model,
            inches=draw_threshold_inches,
        )

        # Cache per le istanze UserMemory
        # TODO: Sostituire con Redis per deployment production
        self._memory_cache: dict[str, UserMemory] = {}

    def _get_or_create_memory(self, user_id: str) -> UserMemory:
        """Get or create a UserMemory instance for the given user.

        Args:
            user_id: The user identifier.

        Returns:
            UserMemory: The memory instance for the user.
        """
        if user_id not in self._memory_cache:
            self._memory_cache[user_id] = UserMemory(
                uri=self.store.uri,
                user=user_id,
            )
        return self._memory_cache[user_id]

    def _retrieve_node(self, state: State):
        """Retrieve documents from the Milvus store based on query.

        Args:
            state: The current state containing the query.

        Returns:
            dict: Updated state with retrieved context.
        """
        context = (
            self.store.retrieve_with_reranker(state["query"])
            if self.rerank
            else self.store.retrieve(state["query"])
        )

        if context is None:
            context = []

        logger.info("Retrieved %d documents from store.", len(context))
        return {"context": context}

    def _context_from_draw_node(self, state: State):
        """Enhance context with draw-specific information.

        Args:
            state: The current state with context.

        Returns:
            dict: Updated state with enhanced context.
        """
        ctx = state.get("context")
        if not ctx:
            return {"context": []}

        context = ctx[:]  # Copy to avoid mutation
        updated_pks = []

        for i, document in enumerate(context):
            doc_type = document.metadata.get("type")
            if doc_type == "draw":
                try:
                    new_context = self.draw.run(document=document, query=state["query"])
                    context[i].page_content = new_context
                    updated_pks.append(str(document.metadata.get("pk", "unknown")))
                except Exception as e:
                    logger.error("Error processing draw document: %s", e)

        if updated_pks:
            logger.info(
                "Updated context with draw information for documents: %s. Total context length: %d",
                updated_pks,
                len(context),
            )

        return {"context": context}

    def _evaluate_node(self, state: State):
        """Evaluate and filter documents based on relevance to query.

        Args:
            state: The current state with context.

        Returns:
            dict: Updated state with filtered context.
        """
        if not state.get("context"):
            return {"context": []}

        # Prepare context for evaluation
        ctx = state.get("context")
        if not ctx:
            return {"context": []}

        docs_text = "\n\n".join(
            f"{doc.metadata['pk']}: type: {doc.metadata.get('type', 'unknown')} content: {doc.page_content}"
            for doc in ctx
        )

        try:
            evaluation_result = self.llm.invoke(
                EVALUATE_CONTEXT.format(query=state["query"], context=docs_text)
            )
            relevant_pks = json.loads(str(evaluation_result.content))

            # Filter context based on relevant PKs
            relevant_pks_str = [str(pk) for pk in relevant_pks]
            filtered_context = [
                doc for doc in ctx if str(doc.metadata["pk"]) in relevant_pks_str
            ]

            logger.info(
                "Filtered context from %d to %d relevant documents.",
                len(ctx),
                len(filtered_context),
            )
            return {"context": filtered_context}

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse evaluation result: %s. Using original context.",
                e,
            )
            return {"context": ctx}
        except Exception as e:
            logger.error("Error during evaluation: %s. Using original context.", e)
            return {"context": ctx}

    def _get_response_node(self, state: State):
        """Generate response using LLM based on query and context.

        Args:
            state: The current state containing query and context.

        Returns:
            dict: Updated state with generated response.
        """
        # Prepare context string
        context_str = ""
        ctx = state.get("context")
        if ctx:
            context_str = "\n\n".join([doc.page_content for doc in ctx])

        memory_str: str | None = None
        uid = state.get("user_id")
        if uid:
            try:
                memory = self._get_or_create_memory(uid)
                memory_str = memory.get(state["query"])
                logger.debug("Retrieved memory for user %s", uid)
            except Exception as e:
                logger.error("Error retrieving memory: %s", e)

        # Generate response
        response = self.llm.invoke(
            GENERATE_RESPONSE.format(
                query=state["query"],
                context=context_str or "No relevant context found.",
                memory=memory_str or "No previous conversation history.",
            )
        )

        logger.info("Generated response for query: %s", state["query"])
        return {"response": response.content}

    def _route_retrieval(self, state: State):
        """Route based on retrieval results.

        Args:
            state: The current state with context.

        Returns:
            str: Next node to execute ('generate') or END.
        """
        ctx = state.get("context")
        has_context = bool(ctx and len(ctx) > 0)
        return "generate" if has_context else END

    def _compile_graph(self):
        """Compile the LangGraph workflow.

        Returns:
            CompiledGraph: The compiled workflow graph.
        """
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("retrieve", self._retrieve_node)
        graph_builder.add_node("evaluate", self._evaluate_node)
        graph_builder.add_node("context_from_draw", self._context_from_draw_node)
        graph_builder.add_node("generate", self._get_response_node)

        # Define workflow
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

    def get_summary(self) -> Optional[str]:
        """Retrieve the summary entry from the store.

        Returns:
            Optional[str]: The summary text if found, None otherwise.
        """
        try:
            result = self.store.query(
                expression='namespace == "summary"', fields=["text"], limit=1
            )
            if result and len(result) > 0:
                logger.info("Retrieved summary from store.")
                return result[0]["text"]
            logger.info("No summary found in store.")
            return None
        except Exception as e:
            logger.error("Error retrieving summary: %s", e)
            return None

    def _save_to_memory(self, user_id: str, query: str, response: str) -> None:
        """Save query-response pair to user memory.

        Args:
            user_id: The user identifier.
            query: The user's query.
            response: The generated response.
        """
        try:
            memory = self._get_or_create_memory(user_id)
            memory.add(query=query, response=response)
        except Exception as e:
            logger.error("Error saving to memory: %s", e)

    def clear_user_memory(self, user_id: str) -> bool:
        """Clear memory for a specific user (e.g., on disconnect).

        Args:
            user_id: The user identifier.

        Returns:
            bool: True if memory was cleared successfully, False otherwise.
        """
        try:
            if user_id in self._memory_cache:
                memory = self._memory_cache[user_id]
                memory.delete()
                del self._memory_cache[user_id]
                logger.info("Cleared memory for user: %s", user_id)
                return True
            logger.warning("No memory found for user: %s", user_id)
            return False
        except Exception as e:
            logger.error("Error clearing memory for user %s: %s", user_id, e)
            return False

    def run(self, query: str, user_id: Optional[str] = None) -> dict:
        """Run the GraphRAG workflow.

        Args:
            query: The user's query.
            user_id: Optional user identifier for memory management.

        Returns:
            dict: The final state containing the response.
        """
        initial_state: State = {
            "query": query,
            "context": None,
            "response": None,
            "user_id": user_id,
        }

        logger.info("Starting workflow for query: %s", query)
        final_state = self.graph.invoke(initial_state)

        # Save to memory if user_id is provided
        if user_id and final_state.get("response"):
            self._save_to_memory(user_id, query, final_state["response"])

        logger.info("Workflow completed successfully.")
        return final_state
