"""State management for GraphRAG workflow."""

from typing import Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class State(TypedDict):
    """
    State representation for graphRAG.

    Attributes:
    - query: The user's query string.
    - summary: Milvus collection summary
    - relevant: True if query is relevant to the collection, False otherwise.
    - context: Retrieved context relevant to the query.
    """

    query: str
    summary: str
    relevant: Optional[bool]
    context: Optional[list[Document]]
