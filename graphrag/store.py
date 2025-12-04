from pymilvus import connections,  db, MilvusException
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_core.retrievers import BaseRetriever

from dotenv import load_dotenv

load_dotenv()

class Store:
    """Milvus vector store for storing and retrieving documents."""
    def __init__(
            self,
            uri: str,
            database: str,
            collection: str,
            namespace: str | None = None,
            embedding_model: str | None = None, 
        ):
        """
        Initialize the Milvus store.

        Args:
            uri: The URI of the Milvus instance.
            database: The database name.
            collection: The collection name.
            namespace: The namespace to use. Defaults to "namespace".
            embedding_model: The embedding model to use. Defaults to "text-embedding-3-small
        """
        self.uri = uri
        self.database = database
        self.collection = collection
        self.namespace = namespace if namespace else "namespace"
        self.embedding_model = embedding_model if embedding_model else "text-embedding-3-small"

        self._connect()
        self._initialize_database()

        self.embed = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = self._create_vstore()


    def _connect(self):
        """Connect to Milvus instance."""        
        try: 
            host = self.uri.split("://")[1].split(":")[0]
            port = int(self.uri.split(":")[-1])
            connections.connect(host=host, port=port)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus at {self.uri}: {e}")
    
    def _initialize_database(self):
        """Initialize the database."""
        try:
            if self.database in db.list_database():
                    db.using_database(self.database)
            else:
                db.create_database(self.database)
        except MilvusException as e:
            raise RuntimeError(f"Failed to initialize database {self.database}: {e}")
    
    def _create_vstore(self):
        """Create and return a vector store"""
        mstore = Milvus(
            embeedding_function=self.embed,
            connection_args={"uri": self.uri, "db_name": self.database},
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            consistency_level="Strong",
            drop_old=False, # TODO: vedi se aggiungere come configurabile
            collection_name=self.collection,
            auto_id=True,
            partition_key_field="namespace",
        )
        return mstore
    
    def add(self, docs: list[Document]):
        """
        Add LangChain Documents to the vector store.
        
        Args:
            - docs: List of LangChain Documents to add.
        """
        _ = self.vector_store.add_documents(documents=docs)
    
    def get_retriever(
            self,
            k: int,
            ranker_type: str = "weighted",
            weigths: list[float] = [0.6, 0.4]
            ) -> BaseRetriever:
        """
        Get a retriever from the vector store.
        
        Args:
            - k: Number of documents to retrieve.
            - ranker_type: Type of ranker to use. Defaults to "weighted".
            - weigths: Weights for the ranker. Defaults to [0.6, 0.4]. 
        """
        retriever = self.vector_store.as_retriever(
            search_kwargs= {
                "k": k,
                "expr": f"namespace == '{self.namespace}'"
            },
            ranker_type=ranker_type,
            ranker_params={"weights": weigths}
        )
        return retriever
    
    def retrieve(
            self,
            query: str,
            k: int,
            score: bool = False
            ):
        """
        Similarity search.
        
        Args:
            - query: The query string.
            - k: Number of documents to retrieve.
            - score: Whether to return scores with the documents. Defaults to False.
        """
        if score:
            docs = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                expr=f"namespace == '{self.namespace}'",
            )
        else:
            docs =  self.vector_store.similarity_search(
                query,
                k=k,
                expr=f"namespace == '{self.namespace}'",
            )
        return docs