from typing_extensions import TypedDict
from typing import Optional


class State(TypedDict):
    query: str
    summary: str
    relevant: Optional[bool]
    context: Optional[str]