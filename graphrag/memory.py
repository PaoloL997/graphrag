"""
- Utilizzerei il reranker anche tipo per recuperare solo le coppie query-response legate alla domanda
Dove salvare:
Possiamo creare un database memory
Ogni collection corrisponde ad un utente
Ogni documento nella collection è una coppia query-response
la collection viene eliminata quando l'utente si disconnette (TODO: da capire come fare)
"""

from langchain_core.documents import Document
from graphrag.store import Store, drop_collection


class UserMemory:
    """Manage user-specific memory using a vector store."""

    def __init__(
        self,
        uri: str,
        user: str,
        k: int = 5,
        embedding_model: str = "text-embedding-3-small",
    ):
        """Initialize the UserMemory with a vector store for the specific user.

        Args:
            uri: The URI of the vector store.
            user: The user identifier to create a dedicated collection.
            k: Number of similar documents to retrieve.
            embedding_model: The embedding model to use.
        """
        self.store = Store(
            uri=uri,
            database="memory",  # TODO: questo lo lascerei così per ora
            collection=user,
            k=k,
            embedding_model=embedding_model,
        )

    def add(self, query: str, response: str) -> None:
        """Add a query-response pair to the user's memory.
        Args:
            query: The user's query.
            response: The system's response to the query.
        """
        page_content = f"QUERY: {query}\nRESPONSE: {response}"
        document = Document(page_content=page_content)
        self.store.add([document])

    def get(self, query: str) -> list[Document]:
        """Retrieve documents related to the query from the user's memory.
        Args:
            query: The user's query.
        Returns:
            A list of Document objects relevant to the query.
        """
        results = self.store.retrieve(query)  # TODO: potrei anche fare senza reranker
        return results

    def delete(self) -> None:
        """Delete the user's memory collection."""
        try:
            drop_collection(
                uri=self.store.uri,
                database=self.store.database,
                collection=self.store.collection,
            )
        except Exception as e:
            raise e
