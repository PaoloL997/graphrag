from typing import cast, List
import redis

from langchain_core.documents import Document
from graphrag.store import Store, drop_collection

LEN_SHORT_MEMORY = 5


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
        self.long_memory_store = Store(
            uri=uri,
            database="memory",
            collection=user,
            k=k,
            embedding_model=embedding_model,
        )
        # Usa StrictRedis invece di Redis per forzare il comportamento sincrono
        self.short_memory_store = redis.StrictRedis(
            host="localhost",
            port=6379,
            decode_responses=True,
        )
        self.user = user

    def add(self, query: str, response: str) -> None:
        page_content = f"QUERY: {query}\nRESPONSE: {response}"
        # Cast esplicito del risultato
        length = cast(int, self.short_memory_store.llen(self.user))
        if length >= LEN_SHORT_MEMORY:
            oldest = cast(str, self.short_memory_store.lindex(self.user, 0))
            if oldest:
                document = Document(page_content=oldest)
                self.long_memory_store.add([document])
        self.short_memory_store.rpush(self.user, page_content)
        self.short_memory_store.ltrim(self.user, -LEN_SHORT_MEMORY, -1)

    def short_term_memory(self) -> str | None:
        # Cast esplicito del risultato a List
        conversations = cast(
            List[str], self.short_memory_store.lrange(self.user, 0, -1)
        )
        if conversations:
            return "\n".join(conversations)
        return None

    def long_term_memory(self, query: str) -> str | None:
        results = self.long_memory_store.retrieve_with_reranker(query)
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
        print(f"Short-term memory retrieved: {short_memory}")
        long_memory = self.long_term_memory(query)
        print(f"Long-term memory retrieved: {long_memory}")
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
                uri=self.long_memory_store.uri,
                database=self.long_memory_store.database,
                collection=self.long_memory_store.collection,
            )
        except Exception as e:
            raise e
