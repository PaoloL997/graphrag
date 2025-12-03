from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from openai import models
from graphrag.config import Config
cfg = Config()


@tool
def check_relevance(query: str, context: str) -> dict:
    """Check if the context is relevant to the query."""
    model = cfg.models("check_relevance")["model"]
    llm = init_chat_model(model=model)
    prompt = cfg.prompt("check_relevance")
    results = llm.invoke(
        prompt.format(query=query, context=context)
    )
    results = results.content.lower().strip()
    return {"relevant": results == "true"}

    