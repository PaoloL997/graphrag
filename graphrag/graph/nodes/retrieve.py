from typing import Dict
from graphrag.core.state import State
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieveNode:
    """Node to retrieve documents from a store based on a query in the state."""

    def __init__(self, store, rerank: bool = False):
        self.store = store
        self.rerank = rerank

    def __call__(self, state: State) -> Dict:
        try:
            context = (
                self.store.retrieve_with_reranker(state["query"])
                if self.rerank
                else self.store.retrieve(state["query"])
            )

            if context is None:
                context = []

            logger.info("Retrieved %d documents from store.", len(context))
            return {"context": context}

        except Exception as e:
            logger.error("Error retrieving documents: %s", e)
            return {"context": []}
