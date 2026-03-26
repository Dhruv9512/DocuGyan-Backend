from langgraph.graph import MessagesState
from typing import Annotated
import operator 
from typing import TypedDict

# ==========================================
# Graph States for Agents
# ==========================================
class ExtractorState(MessagesState):
    # Also those fields should be same as OrchestratorState to seamlessly sync state between them
    project_id: str
    reference_urls: list[str]
    original_questions: list[str]
    
    # Annotated so parallel Map-Reduce workers append to the list instead of overwriting
    extracted_doc_blob_url: Annotated[list[str], operator.add]
    refined_questions_blob_url: Annotated[list[str], operator.add]

class OrchestratorState(MessagesState):
    """
    The global state for the entire DocuGyan deterministic pipeline.
    """
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
    """
    Micro-state passed to individual URL extraction workers.
    """
    url: str
    project_id: str