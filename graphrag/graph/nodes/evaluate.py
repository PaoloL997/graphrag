"""Evaluation node for filtering relevant documents."""

import json
from typing import Dict, List
from langchain_core.documents import Document

from graphrag.config.prompts import EVALUATE_CONTEXT_PROMPT
from graphrag.core.state import State
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluateNode:
    """Node for evaluating and filtering documents based on relevance."""

    def __init__(self, llm):
        """Initialize the evaluation node.

        Args:
            llm: The language model for evaluation.
        """
        self.llm = llm

    def __call__(self, state: State) -> Dict:
        """Evaluate and filter documents based on relevance to query.

        Args:
            state: The current state containing query and context.

        Returns:
            dict: Updated state with filtered context.
        """
        ctx = state.get("context")
        if not ctx:
            return {"context": []}

        docs_text = self._format_context_for_evaluation(ctx)

        try:
            evaluation_result = self.llm.invoke(
                EVALUATE_CONTEXT_PROMPT.format(query=state["query"], context=docs_text)
            )

            content = evaluation_result.content
            if isinstance(content, str):
                relevant_pks = self._parse_evaluation_result(content)
            elif isinstance(content, list):
                relevant_pks = [str(pk) for pk in content]
            else:
                relevant_pks = []

            filtered_context = self._filter_context_by_pks(ctx, relevant_pks)

            logger.info(
                "Filtered context from %d to %d relevant documents.",
                len(ctx),
                len(filtered_context),
            )

            return {"context": filtered_context}

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse evaluation result: %s. Using original context.", e
            )
            return {"context": ctx}

        except Exception as e:
            logger.error("Error during evaluation: %s. Using original context.", e)
            return {"context": ctx}

    def _format_context_for_evaluation(self, context: List[Document]) -> str:
        """Format context documents for evaluation.

        Args:
            context: List of documents.

        Returns:
            str: Formatted context string.
        """
        return "\n\n".join(
            f"{doc.metadata['pk']}: "
            f"type: {doc.metadata.get('type', 'unknown')} "
            f"content: {doc.page_content}"
            for doc in context
        )

    def _parse_evaluation_result(self, content: str) -> List[str]:
        """Parse evaluation result from LLM.

        Args:
            content: The LLM response content.

        Returns:
            List[str]: List of relevant PKs.

        Raises:
            json.JSONDecodeError: If parsing fails.
        """
        relevant_pks = json.loads(str(content))
        return [str(pk) for pk in relevant_pks]

    def _filter_context_by_pks(
        self, context: List[Document], relevant_pks: List[str]
    ) -> List[Document]:
        """Filter context documents by relevant PKs.

        Args:
            context: List of documents.
            relevant_pks: List of relevant primary keys.

        Returns:
            List[Document]: Filtered documents.
        """
        return [doc for doc in context if str(doc.metadata["pk"]) in relevant_pks]
