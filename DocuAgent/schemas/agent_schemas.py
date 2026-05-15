from langgraph.graph import MessagesState
from typing import Annotated, Any
import operator 
from typing import TypedDict

from qdrant_client.models import Dict

from DocuAgent.schemas.llm_schemas import PlannerOutput, RetrievalGraderOutput

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
    extracted_questions_blob_url: Annotated[list[str], operator.add]

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
    extracted_questions_blob_url: Annotated[list[str], operator.add]

    domain: str
    final_answers_blob_url: list[str]

class AcademicAgentState(MessagesState):
    """Global state for the Academic Orchestrator. Fanning out and fanning in."""
    project_id: str
    extracted_questions_blob_url: list[str]
    extracted_doc_blob_url: list[str]
    
    # REDUCER: As each parallel worker finishes, it appends its formatted markdown here
    completed_answers: Annotated[list[str], operator.add]
    failed_questions: Annotated[list[str], operator.add]
    
    final_answers_blob_url: list[str]
  

# ==========================================
# Graph States for Parallel Workers
# ==========================================
class ExtractionWorkerState(TypedDict):
    """
    Micro-state passed to individual URL extraction workers.
    """
    url: str
    project_id: str

class QuestionState(TypedDict):
    """Local state for a SINGLE parallel worker processing one question."""
    project_id: str
    original_question: str
    
    # The planner output gets saved directly to this specific question's state
    plan: PlannerOutput
    
    # Pipeline execution data
    grader_assessment: RetrievalGraderOutput
    retrieved_docs: list[Any]
    retrieved_references: list[str]
 
    draft_answer: str
    diagram_mapping: Dict[str, str]
    
    # Output: Single item list to be reduced globally
    completed_answers: list[str] 
    failed_questions: list[str]