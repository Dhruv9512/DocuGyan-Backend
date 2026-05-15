from typing import TypedDict, Annotated, Sequence
import operator


class ChatState(TypedDict):
    messages: Annotated[Sequence, operator.add]
    final_answer: str