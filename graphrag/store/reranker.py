from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


class CohereReranker:
    """Cohere Reranker wrapper class."""

    def __init__(
        self,
        model: str = "rerank-v3.5",  # latest | multilingual
        top_n: int = 3,
    ):
        """Initialize the Cohere Reranker.

        Args:
            model: The Cohere rerank model to use.
            top_n: The number of top documents to return.
        """
        self.client = CohereRerank(model=model, top_n=top_n)

    def rerank(self, query: str, documents: list[Document]):
        """Rerank documents based on query using Cohere Reranker.

        Args:
            query: The search query string.
            documents: List of documents to rerank.

        Returns:
            list[Document]: Reranked documents.
        """
        # Ensure query is a string
        if isinstance(query, list):
            query = str(query[0]) if query else ""

        query = str(query).strip()

        if not query or not documents:
            return documents

        # Use keyword arguments to be explicit
        return self.client.compress_documents(documents=documents, query=query)
