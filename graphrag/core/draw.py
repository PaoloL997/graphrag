from techdraw.agent import TechDraw
from langchain_core.documents import Document
import fitz
import numpy as np

ZOOMING_PROMPT = """You are a technical drawing analyzer specialized in identifying the most relevant region.

Your task: Analyze this technical drawing and identify the SINGLE most relevant bounding box that contains the information needed to answer the user's question.

COORDINATE SYSTEM:
- x=0, y=0 is the TOP-LEFT corner
- x=1000, y=1000 is the BOTTOM-RIGHT corner
- All coordinates are integers between 0 and 1000

IMPORTANT INSTRUCTIONS:
1. Return EXACTLY ONE bounding box in NORMALIZED coordinates (0-1000 scale)
2. Format: xmin,ymin,xmax,ymax where:
   - xmin, ymin = top-left corner of the box
   - xmax, ymax = bottom-right corner of the box
   - All values are integers between 0 and 1000
3. BE GENEROUS with boundaries - include surrounding context (expand the box by ~20-30%)
4. Choose the MOST IMPORTANT region if multiple areas are relevant
5. Consider: labels, legends, dimension lines, notes, and related elements

REASONING PROCESS:
- First, understand what information the question requires
- Scan the entire drawing to locate ALL relevant areas
- Choose the SINGLE most important area that contains the answer
- Define ONE bounding box that includes:
  * The main element(s)
  * Associated dimensions and annotations  
  * Nearby reference markers
  * Related detail callouts

OUTPUT FORMAT (return ONLY the coordinates as 4 integers separated by commas, nothing else):
xmin,ymin,xmax,ymax

USER QUESTION: {query}

Now identify the single most relevant region in this technical drawing.
"""

RESPONSE_PROMPT = """
You are a technical drawing analyst. Analyze this technical drawing and answer the user's question with specific, relevant information.

IMPORTANT: 
- Focus ONLY on what the user is asking about
- If you cannot find the requested information, state "Information not found in this drawing"
- Be precise and factual based on what you can see in the drawing

USER QUESTION: {query}

Please provide a direct answer to this question based on the technical drawing.
"""


class ContextFromDraw:
    """Extract context from technical drawings using TechDraw agent."""

    def __init__(
        self,
        thinking_level: str = "low",
        model: str = "gemini-3-flash-preview",
        inches: float = 28.0,
    ):
        """
        Initialize the ContextFromDraw agent.

        Args:
            thinking_level: The thinking level for the TechDraw agent ("low", "medium", "high").
            model: The Gemini model to use for the TechDraw agent.
            inches: Minimum dimension (in inches) to trigger zooming analysis.
        """
        self.agent = TechDraw(thinking_level=thinking_level, model=model)
        self.inches = inches  # Min threshold for zooming

    @staticmethod
    def _get_size(page: fitz.Page):
        """Get the size of the page in points.
        Args:
            page: The fitz Page object.
        Returns:
            A tuple (width, height) in points.
        """
        rect = page.rect
        width = int(rect.width)
        height = int(rect.height)
        return width, height

    @staticmethod
    def _get_img(page: fitz.Page):
        """Render the page to an image with 3x zoom.
        Args:
            page: The fitz Page object.
        Returns:
            A tuple (image as numpy array, pixmap).
        """
        mat = fitz.Matrix(3, 3)  # 3x zoom
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        return img, pix

    def _zooming(self, width: int, height: int) -> bool:
        """Determine if zooming is needed based on page size.

        Args:
            width: Width of the page in points.
            height: Height of the page in points.
        Returns:
            True if zooming is needed, False otherwise.
        """
        w, h = int(width / 72), int(height / 72)  # Convert to inches
        max_d = max(w, h)
        return max_d > self.inches

    @staticmethod
    def _crop(image: np.ndarray, bbox: tuple, pix: fitz.Pixmap) -> np.ndarray:
        """Crop the image based on normalized bounding box coordinates.

        Args:
            image: The full image as a numpy array.
            bbox: A tuple (xmin, ymin, xmax, ymax) in normalized coordinates (0-1000).
            pix: The fitz Pixmap object for dimension reference.
        Returns:
            The cropped image as a numpy array.
        """
        xmin, ymin, xmax, ymax = bbox
        cropped_width = pix.width * (xmax - xmin) // 1000
        cropped_height = pix.height * (ymax - ymin) // 1000
        cropped_x = pix.width * xmin // 1000
        cropped_y = pix.height * ymin // 1000
        return image[
            cropped_y : cropped_y + cropped_height,
            cropped_x : cropped_x + cropped_width,
        ]

    def run(self, document: Document, query: str):
        """
        Extract context from the technical drawing document based on the query.

        Args:
            document: The Document object containing the technical drawing.
            query: The user's query.
        Returns:
            The extracted context as a string.
        """
        draw: str | np.ndarray | None = None
        path = document.metadata["path"]
        doc = fitz.open(path)
        page = doc[0]  # Analyze only the first page
        width, height = self._get_size(page)
        img, pix = self._get_img(page)
        zoom = self._zooming(width, height)
        if zoom:
            print(
                f"Zooming in on drawing ({width / 72:.2f}in x {height / 72:.2f}in)..."
            )
            response = self.agent.invoke(
                query=ZOOMING_PROMPT.format(query=query), attached=str(path)
            )
            bbox = map(int, response.text.split(","))
            draw = self._crop(img, tuple(bbox), pix)
        else:
            print(f"No zooming needed ({width / 72:.2f}in x {height / 72:.2f}in).")
            draw = str(path)

        out = self.agent.invoke(
            query=RESPONSE_PROMPT.format(query=query), attached=draw
        )
        return out.text
