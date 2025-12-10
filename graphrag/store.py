"""Vector store implementation using Milvus for document storage and retrieval."""

from pymilvus import connections, db, MilvusException
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_milvus import BM25BuiltInFunction, Milvus
from dotenv import load_dotenv

from graphrag.reranker import CohereReranker


load_dotenv()


class Store:
    """Milvus vector store for storing and retrieving documents."""

    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        namespace: str,
        k: int = 4,
        embedding_model: str | None = None,
    ):
        """Initialize the Milvus store.

        Args:
            uri: The URI of the Milvus instance.
            database: The database name.
            collection: The collection name.
            k: The number of documents to retrieve. Defaults to 4.
            namespace: The namespace to use.
            embedding_model: The embedding model to use. Defaults to "text-embedding-3-small".
            reranker: Whether to use a reranker. Defaults to False.

        Raises:
            ConnectionError: If connection to Milvus fails.
            RuntimeError: If database initialization fails.
        """
        self.uri = uri
        self.database = database
        self.collection = collection
        self.k = k
        self.namespace = namespace
        self.embedding_model = (
            embedding_model if embedding_model else "text-embedding-3-small"
        )

        self._connect()
        self._initialize_database()

        self.embed = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = self._create_vstore()

        self.reranker = CohereReranker(top_n=self.k)

    def _connect(self):
        """Connect to Milvus instance.

        Raises:
            ConnectionError: If connection to Milvus fails.
        """
        try:
            host = self.uri.split("://")[1].split(":")[0]
            port = int(self.uri.split(":")[-1])
            connections.connect(host=host, port=port)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Milvus at {self.uri}: {e}"
            ) from e

    def _initialize_database(self):
        """Initialize the database.

        Raises:
            RuntimeError: If database initialization fails.
        """
        try:
            if self.database in db.list_database():
                db.using_database(self.database)
            else:
                db.create_database(self.database)
        except MilvusException as e:
            raise RuntimeError(
                f"Failed to initialize database {self.database}: {e}"
            ) from e

    def _create_vstore(self):
        (
            """Create and return a vector store.
        
        Returns:
            Milvus: Configured Milvus vector store instance.
        """
            ""
        )
        mstore = Milvus(
            embedding_function=self.embed,
            connection_args={"uri": self.uri, "db_name": self.database},
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            consistency_level="Strong",
            drop_old=False,  # TODO: vedi se aggiungere come configurabile
            collection_name=self.collection,
            auto_id=True,
            partition_key_field="namespace",
        )
        return mstore

    def add(self, docs: list[Document]):
        """
        Add LangChain Documents to the vector store.

        Args:
            docs: List of LangChain Documents to add.
        """
        _ = self.vector_store.add_documents(documents=docs)

    def get_retriever(
        self, ranker_type: str = "weighted", weigths: list[float] | None = None
    ) -> BaseRetriever:
        """Get a retriever from the vector store.

        Args:
            ranker_type: Type of ranker to use. Defaults to "weighted".
            weigths: Weights for the ranker. Defaults to [0.6, 0.4].

        Returns:
            BaseRetriever: Configured retriever instance.
        """
        if weigths is None:
            weigths = [0.6, 0.4]
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.k},
            ranker_type=ranker_type,
            ranker_params={"weights": weigths},
        )
        return retriever

    def retrieve(
        self,
        query: str,
        score: bool = False,
    ):
        """Similarity search across all documents.

        Searches all documents in the store while preserving namespace information
        in the returned documents for tracking their origin.

        Args:
            query: The query string.
            score: Whether to return scores with the documents. Defaults to False.

        Returns:
            list: List of documents or document-score tuples if score=True.
                  Each document contains namespace metadata for tracking origin.
        """
        if score:
            docs = self.vector_store.similarity_search_with_score(query, k=self.k)
        else:
            docs = self.vector_store.similarity_search(
                query,
                k=self.k,
            )
        return docs

    def retrieve_with_reranker(self, query: str):
        """Retrieve documents and rerank them using the Cohere Reranker.

        Args:
            query: The query string (must be a string, not a list).

        Returns:
            list: Reranked list of documents.
        """
        # Ensure query is a string
        if isinstance(query, list):
            query = query[0] if query else ""

        query = str(query)

        if not self.reranker:
            self.reranker = CohereReranker(top_n=self.k)

        # Get more candidates for reranking
        docs = self.vector_store.similarity_search(query, k=self.k * 4)

        # Rerank with proper string query
        reranked_docs = self.reranker.rerank(query, docs)
        return reranked_docs
