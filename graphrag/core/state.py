"""State management for GraphRAG workflow."""

from typing import Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class State(TypedDict):
    """
    State representation for graphRAG.

    Attributes:
    - query: The user's query string.
    - summary: Summary of the document collection.
    - context: Retrieved context relevant to the query.
    - response: Generated response from the LLM.
    """

    query: str
    context: Optional[list[Document]]
    response: Optional[str]
    user_id: Optional[str]
