"""
- Utilizzerei il reranker anche tipo per recuperare solo le coppie query-response legate alla domanda
Dove salvare:
Possiamo creare un database memory
Ogni collection corrisponde ad un utente
Ogni documento nella collection è una coppia query-response
la collection viene eliminata quando l'utente si disconnette (TODO: da capire come fare)
"""

from collections import deque
from typing import Deque
from langchain_core.documents import Document
from graphrag.store import Store, drop_collection

LEN_SHORT_MEMORY = 5


class UserMemory:
    """Manage user-specific memory using a vector store."""

    short_memory: Deque[Document]

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
        self.short_memory = deque(
            maxlen=LEN_SHORT_MEMORY
        )  # TODO: capire se lasciare in memoria

    def add(self, query: str, response: str) -> None:
        """Add a query-response pair to the user's memory.
        Args:
            query: The user's query.
            response: The system's response to the query.
        """
        page_content = f"QUERY: {query}\nRESPONSE: {response}"
        document = Document(page_content=page_content)
        if (
            self.short_memory.maxlen is not None
            and len(self.short_memory) == self.short_memory.maxlen
        ):
            self.store.add([self.short_memory[0]])
        self.short_memory.append(document)

    def short_term_memory(self) -> str | None:
        if not self.short_memory:
            return None
        return "\n".join(doc.page_content for doc in self.short_memory)

    def long_term_memory(self, query: str) -> str | None:
        results = self.store.retrieve_with_reranker(query)
        if results:
            return "\n".join([doc.page_content for doc in results])
        return None

    def get(self, query: str) -> str | None:
        """Perform hybrid retrieval combining short-term and long-term memory.
        Args:
            query: The user's query.
        Returns:
            A combined string of relevant short-term and long-term memory, or None if no memory is found.
        """
        short_memory = self.short_term_memory()
        long_memory = self.long_term_memory(query)
        combined_memory = ""
        if short_memory:
            combined_memory += f"Short-term memory:\n{short_memory}\n\n"
        if long_memory:
            combined_memory += f"Long-term memory:\n{long_memory}\n\n"
        return combined_memory if combined_memory else None

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
