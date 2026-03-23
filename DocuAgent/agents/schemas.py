import operator
from typing import Literal, Annotated
from pydantic import BaseModel, Field
from typing import TypedDict
from langgraph.graph import MessagesState

# ==========================================
# Pydantic Models (structured LLM output)
# ==========================================

class RAGClassification(BaseModel):
    strategy: Literal["vector", "graph", "vectorless"] = Field(
        description="The RAG strategy best suited for this document type."
    )
    reasoning: str = Field(
        description="Brief explanation of why this strategy was chosen."
    )

# ==========================================
# Graph States for Agents
# ==========================================

class ExtractorState(MessagesState):
    project_id: str
    reference_urls: list[str]
    original_questions: list[str]
    # Annotated so parallel Map-Reduce workers append to the list instead of overwriting
    extracted_doc_blob_url: Annotated[list[str], operator.add]
    refined_questions_blob_url: Annotated[list[str], operator.add]

class SupervisorState(MessagesState):
    project_id: str
    user_uuid: str
    reference_urls: list[str]
    rag_strategy: str
    rag_reasoning: str
    ingestion_done: bool
    original_questions: list[str]
    # Must match the Extractor exactly to seamlessly sync state
    extracted_doc_blob_url: Annotated[list[str], operator.add]
    refined_questions_blob_url: Annotated[list[str], operator.add]


# ==========================================
# Graph States for Parallel Workers
# ==========================================
class ExtractionWorkerState(TypedDict):
    url: str
    project_id: str