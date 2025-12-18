"""Vector store implementation using Milvus for document storage and retrieval."""

from pymilvus import connections, db, MilvusException, Collection, utility
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_milvus import BM25BuiltInFunction, Milvus
from dotenv import load_dotenv

from graphrag.reranker import CohereReranker


load_dotenv()

SUMMARY_PROMPT = """
You are a content analysis system. You will receive a text that is a concatenation of multiple entries from a collection. 

Your task is to produce a single, comprehensive, and accurate summary of the text. The summary must:

- ALWAYS Begin the text by explicitly stating that it is a summary (for example, start with "Summary:" or similar).
- Include all main topics, key concepts, important details, definitions, relationships, events, or entities relevant for understanding the content.
- Be complete and faithful to the input text.
- Contain only the summary, without any additional explanations or commentary.

TEXT TO SUMMARIZE:
{INPUT_TEXT}
"""


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
            reranker: Whether to use a reranker. Defaults to False.

        Raises:
            ConnectionError: If connection to Milvus fails.
            RuntimeError: If database initialization fails.
        """
        self.uri = uri
        self.database = database
        self.collection = collection
        self.k = k
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
        if isinstance(query, list):
            query = str(query[0]) if query else ""

        if not self.reranker:
            self.reranker = CohereReranker(top_n=self.k)

        docs = self.vector_store.similarity_search(query, k=self.k * 4)
        reranked_docs = self.reranker.rerank(query, docs)
        return reranked_docs

    @staticmethod
    def _delete_old_summary(collection: Collection):
        """Delete old summary entries from the collection.

        Args:
            collection: The Milvus collection instance.
        """
        try:
            summary_check = collection.query(
                expr='namespace == "summary"',
                output_fields=["pk"],
                limit=100,
            )
            if summary_check:
                ids_to_delete = [item["pk"] for item in summary_check]
                collection.delete(expr=f"pk in {ids_to_delete}")
        except Exception as e:
            print(f"Error deleting old summary entries: {e}")

    def query(self, expression: str, fields: list | None = None, limit: int = 10):
        collection = Collection(self.collection)
        collection.load()
        result = collection.query(
            expr=expression,
            output_fields=fields if fields else ["*"],
            limit=limit,
        )
        return result

    def summarize(self, model: str = "gpt-4.1-mini"):
        """Generate a summary of the entire collection.

        Args:
            model: The language model to use for summarization. Defaults to "gpt-4.1-mini".
        """
        collection = Collection(self.collection)
        collection.load()
        self._delete_old_summary(collection)
        results = collection.query(
            expr="",
            output_fields=["text"],
            limit=1000,  # TODO: se hai più di 1000 documenti magari seleziona randomicamente
        )
        full_text = ""
        for item in results:
            full_text += item["text"] + "\n"

        llm = init_chat_model(model)
        summary = llm.invoke(SUMMARY_PROMPT.format(INPUT_TEXT=full_text)).content
        doc = Document(
            page_content=str(summary),
            metadata={
                "path": "N/A",
                "page_start": "N/A",
                "page_end": "N/A",
                "type": "text",
                "name": "collection_summary",
                "namespace": "summary",
            },
        )
        self.add([doc])
