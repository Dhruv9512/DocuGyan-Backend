from ast import operator
from typing import Literal, Annotated
from pydantic import BaseModel, Field


# LangGraph imports
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


class DomainClassification(BaseModel):
    domain: Literal["academic", "financial", "audit"] = Field(
        description="The domain this document belongs to."
    )
    reasoning: str = Field(
        description="Brief explanation of why this domain was chosen."
    )


class ExtractedDocument(BaseModel):
    url: str
    content: str
    file_type: str


# ==========================================
# Graph States (one per agent)
# ==========================================
class SupervisorState(MessagesState):
    """State for the Supervisor agent graph."""
    project_id: str
    user_uuid: str
    reference_urls: list[str]
    extracted_docs: Annotated[list[ExtractedDocument], operator.add]
    rag_strategy: str
    rag_reasoning: str
    ingestion_done: bool
    original_questions: list[str]
    refined_questions: list[str]


class AcademicAgentState(MessagesState):
    """State for the Academic domain agent."""
    project_id: str
    extracted_docs: list[ExtractedDocument]
    # academic-specific fields...


class FinancialAgentState(MessagesState):
    """State for the Financial domain agent."""
    project_id: str
    extracted_docs: list[ExtractedDocument]
    # financial-specific fields...


class AuditAgentState(MessagesState):
    """State for the Audit domain agent."""
    project_id: str
    extracted_docs: list[ExtractedDocument]
    # audit-specific fields...