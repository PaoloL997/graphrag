"""Draw context enhancement node."""

from typing import Dict
from graphrag.core.state import State
from graphrag.core.draw import ContextFromDraw
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class DrawNode:
    """Node for enhancing context with draw-specific information."""

    def __init__(
        self,
        thinking_level: str = "low",
        model: str = "gemini-3-flash-preview",
        threshold_inches: float = 28.0,
    ):
        """Initialize the draw node.

        Args:
            thinking_level: The thinking level for the TechDraw agent.
            model: The Gemini model to use.
            threshold_inches: Minimum dimension to trigger zooming analysis.
        """
        self.draw = ContextFromDraw(
            thinking_level=thinking_level,
            model=model,
            inches=threshold_inches,
        )

    def __call__(self, state: State) -> Dict:
        """Enhance context with draw-specific information.

        Args:
            state: The current state with context.

        Returns:
            dict: Updated state with enhanced context.
        """
        ctx = state.get("context")
        if not ctx:
            return {"context": []}

        context = list(ctx)
        updated_pks = []

        for i, document in enumerate(context):
            if (
                document.metadata.get("type") == "draw"
                and state["refined_query"] is not None
            ):
                try:
                    new_context = self.draw.run(
                        document=document, query=state["refined_query"]
                    )
                    context[i].page_content = new_context

                    pk = document.metadata.get("pk", "unknown")
                    updated_pks.append(str(pk))

                except Exception as e:
                    logger.error(
                        "Error processing draw document (pk: %s): %s",
                        document.metadata.get("pk", "unknown"),
                        e,
                    )

        if updated_pks:
            logger.info(
                "Updated context with draw information for documents: %s. "
                "Total context length: %d",
                updated_pks,
                len(context),
            )

        return {"context": context}
