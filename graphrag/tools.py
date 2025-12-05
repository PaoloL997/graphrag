"""Tools and utilities for GraphRAG operations."""

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from graphrag.config import Config

cfg = Config()


@tool
def check_relevance(query: str, context: str) -> dict:
    """Check if the context is relevant to the query.

    Args:
        query: The user's query string.
        context: The context to check relevance against.

    Returns:
        dict: Dictionary with 'relevant' key containing boolean result.
    """
    model = cfg.models("check_relevance")["model"]
    llm = init_chat_model(model=model)
    prompt = cfg.prompt("check_relevance")
    results = llm.invoke(prompt.format(query=query, context=context))
    results = results.content.lower().strip()
    return {"relevant": results == "true"}
