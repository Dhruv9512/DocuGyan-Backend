from typing import Literal

from pydantic import BaseModel, Field

# ==========================================================
# Question Refinement Output Schema from QuestionRefiner
# =========================================================
class RefinedQuestion(BaseModel):
    refined_question: str = Field(
        ..., 
        min_length=10,
        description="The improved, clear, and professional version of the extracted question."
    )

class RefinedBatch(BaseModel):
    questions: list[RefinedQuestion]



# =========================================================================
# RAG Strategy Classification Schema from DocuPipelineOrchestrator agents
# =========================================================================
class RAGClassification(BaseModel):
    strategy: Literal["vector", "graph", "vectorless"] = Field(
        description="The RAG strategy best suited for this document type."
    )
    reasoning: str = Field(
        description="Brief explanation of why this strategy was chosen."
    )