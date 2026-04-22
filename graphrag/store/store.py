"""Vector store implementation using Milvus for document storage and retrieval."""

import logging
from pymilvus import connections, db, MilvusException, Collection, utility
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from graphrag.store.reranker import CohereReranker


load_dotenv()

logger = logging.getLogger(__name__)

_MILVUS_VARCHAR_MAX = 65535
_SAFE_CHUNK_SIZE = 60_000
_CHUNK_OVERLAP = 200


# def drop_collection(name: str, database: str, uri: str = "http://localhost:19530"):
def drop_collection(uri: str, database: str, collection: str):
    """
    Drop a collection from the specified database.

    Args:
        uri: The URI of the Milvus instance.
        collection: The name of the collection to drop.
        database: The name of the database containing the collection.
    Returns:
        bool: True if the collection was dropped successfully, False otherwise.
    """
    try:
        # Establish connection to Milvus
        host = uri.split("://")[1].split(":")[0]
        port = int(uri.split(":")[-1])
        connections.connect(host=host, port=port)

        existing_dbs = db.list_database()
        if database not in existing_dbs:
            print(f"Database {database} does not exist.")
            return False
        db.using_database(database)
        collections = utility.list_collections()
        if collection not in collections:
            print(f"Collection {collection} does not exist in database {database}.")
            return False
        col = Collection(collection)
        col.drop()
        return True
    except MilvusException as e:
        print(f"Error dropping collection {collection} from database {database}: {e}")
        return False


def list_databases(uri: str) -> list[str]:
    """
    List all databases in the Milvus instance.

    Args:
        uri: The URI of the Milvus instance.
    Returns:
        list[str]: List of database names.
    """
    host = uri.split("://")[1].split(":")[0]
    port = int(uri.split(":")[-1])
    connections.connect(host=host, port=port)
    return db.list_database()


def create_database(uri: str, database: str) -> bool:
    """
    Create a new database in the Milvus instance.

    Args:
        uri: The URI of the Milvus instance.
        database: The name of the database to create.
    Returns:
        bool: True if the database was created successfully, False otherwise.
    """
    try:
        host = uri.split("://")[1].split(":")[0]
        port = int(uri.split(":")[-1])
        connections.connect(host=host, port=port)
        existing_dbs = db.list_database()
        if database in existing_dbs:
            print(f"Database {database} already exists.")
            return False
        db.create_database(database)
        return True
    except MilvusException as e:
        print(f"Error creating database {database}: {e}")
        return False


def list_collections(uri: str, database: str) -> list[str]:
    """
    List all collections in the specified database.

    Args:
        uri: The URI of the Milvus instance.
        database: The name of the database.
    Returns:
        list[str]: List of collection names.
    """
    host = uri.split("://")[1].split(":")[0]
    port = int(uri.split(":")[-1])
    connections.connect(host=host, port=port)
    db.using_database(database)
    return utility.list_collections()


class Store:
    """Milvus vector store for storing and retrieving documents."""

    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        k: int = 4,
        embedding_model: str | None = None,
    ):
        """Initialize the Milvus store.

        Args:
            uri: The URI of the Milvus instance.
            database: The database name.
            collection: The collection name.
            k: The number of documents to retrieve. Defaults to 4.
            embedding_model: The embedding model to use. Defaults to "text-embedding-3-small".

        Raises:
            ConnectionError: If connection to Milvus fails.
            RuntimeError: If database initialization fails.
        """
        # Suppress async warnings from Milvus client
        logging.getLogger("langchain_milvus").setLevel(logging.ERROR)
        self.uri = uri
        self.database = database
        self.collection = collection
        self.k = k
        self.embedding_model = (
            embedding_model if embedding_model else "text-embedding-3-small"
        )

        self._ensure_database()

        self.embed = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = self._create_vstore()

        self.reranker = CohereReranker(top_n=self.k)

    def _ensure_database(self):
        """Ensure the target database exists using a temporary connection.

        Uses a separate alias to avoid interfering with langchain_milvus
        connection management.

        Raises:
            ConnectionError: If connection to Milvus fails.
            RuntimeError: If database initialization fails.
        """
        alias = "_store_setup"
        try:
            host = self.uri.split("://")[1].split(":")[0]
            port = int(self.uri.split(":")[-1])
            connections.connect(alias=alias, host=host, port=port)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Milvus at {self.uri}: {e}"
            ) from e
        try:
            if self.database not in db.list_database(using=alias):
                db.create_database(self.database, using=alias)
        except MilvusException as e:
            raise RuntimeError(
                f"Failed to initialize database {self.database}: {e}"
            ) from e
        finally:
            try:
                connections.disconnect(alias)
            except Exception:
                pass

    def _create_vstore(self):
        """Create and return a vector store.

        `langchain_milvus.Milvus` uses `MilvusClient` for data operations, but
        internally falls back to `pymilvus.orm.Collection` (e.g. inside
        ``_extract_fields``) which requires a connection registered in
        ``pymilvus.orm.connections``. We register it explicitly here under the
        same alias the vector store will use; without this, the first
        ``add_documents`` call on a not-yet-existing collection raises
        ``ConnectionNotExistException``.
        """
        mstore = Milvus(
            embedding_function=self.embed,
            connection_args={"uri": self.uri, "db_name": self.database},
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            consistency_level="Strong",
            drop_old=False,
            collection_name=self.collection,
            auto_id=True,
            partition_key_field="namespace",
        )
        try:
            host = self.uri.split("://")[1].split(":")[0]
            port = int(self.uri.split(":")[-1])
            connections.connect(
                alias=mstore.alias,
                host=host,
                port=port,
                db_name=self.database,
            )
        except Exception as e:
            logger.warning(
                "Failed to register ORM connection for alias %s: %s",
                mstore.alias,
                e,
            )
        return mstore

    @staticmethod
    def _split_oversized(docs: list[Document]) -> list[Document]:
        """Split documents whose text exceeds Milvus VARCHAR max length.

        Args:
            docs: Input list of LangChain Documents.

        Returns:
            New list where every document has page_content <= _MILVUS_VARCHAR_MAX.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_SAFE_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )
        result: list[Document] = []
        for doc in docs:
            if len(doc.page_content) > _MILVUS_VARCHAR_MAX:
                chunks = splitter.split_documents([doc])
                result.extend(chunks)
            else:
                result.append(doc)
        return result

    def add(self, docs: list[Document]):
        """
        Add LangChain Documents to the vector store.

        Documents whose text exceeds the Milvus VARCHAR limit are automatically
        split into smaller chunks before insertion.

        Args:
            docs: List of LangChain Documents to add.
        """
        docs = self._split_oversized(docs)
        _ = self.vector_store.add_documents(documents=docs)

    def delete(self, namespace: str):
        """Delete documents from a specific namespace.
        Args:
            namespace: The namespace from which to delete documents.
        """
        try:
            docs = self.query(
                expression=f'namespace == "{namespace}"',
                fields=["pk"],
                limit=1000,  # TODO: valuta se aumentare
            )
            if docs:
                ids_to_delete = [item["pk"] for item in docs]
                collection = Collection(self.collection)
                collection.delete(expr=f"pk in {ids_to_delete}")
        except Exception as e:
            print(f"Error deleting documents in namespace {namespace}: {e}")

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
            list: List of documents (or document-score tuples if score=True), or an
                  empty list if the collection does not exist yet.
        """
        try:
            if score:
                return self.vector_store.similarity_search_with_score(query, k=self.k)
            return self.vector_store.similarity_search(query, k=self.k)
        except MilvusException as e:
            logger.warning("Retrieve failed on collection %s: %s", self.collection, e)
            return []

    def retrieve_with_reranker(self, query: str):
        """Retrieve documents and rerank them using the Cohere Reranker.

        Args:
            query: The query string (must be a string, not a list).

        Returns:
            list: Reranked list of documents, or an empty list if the collection
                  does not exist yet.
        """
        if isinstance(query, list):
            query = str(query[0]) if query else ""

        if not self.reranker:
            self.reranker = CohereReranker(top_n=self.k)

        try:
            docs = self.vector_store.similarity_search(query, k=self.k * 4)
        except MilvusException as e:
            logger.warning(
                "Retrieve-with-reranker failed on collection %s: %s",
                self.collection,
                e,
            )
            return []
        return self.reranker.rerank(query, docs)

    def query(self, expression: str, fields: list | None = None, limit: int = 10):
        collection = Collection(self.collection)
        collection.load()
        result = collection.query(
            expr=expression,
            output_fields=fields if fields else ["*"],
            limit=limit,
        )
        return result
