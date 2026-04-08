from typing import List, Literal, Optional

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

# =================================================================
# Domain Classification Schema from DocuPipelineOrchestrator agents
# =================================================================
class DomainClassification(BaseModel):
    domain: Literal["academic", "financial", "audit"] = Field(
        description="The specialized domain category of the documents and questions."
    )
    reasoning: str = Field(
        description="Brief explanation of why this domain was selected based on the references and questions."
    )



# ============================================================
# Academic Question Planning Output Schema from AcademicAgent
# ============================================================
class PlannerOutput(BaseModel):
    """Execution strategy and strict constraints for answering an academic question."""
    steps: List[str] = Field(
        description="A sequential list of steps the downstream agents should follow to answer the question effectively."
    )
    question_category: Literal["academic", "coding", "math", "factual", "analytical", "creative"] = Field(
        description="The fundamental category of the question. Used to route to the correct specialist persona."
    )
    allocated_marks: Optional[int] = Field(
        default=None,
        description="The specific marks or points allocated to this question if mentioned (e.g., 5, 10). If not mentioned, return null."
    )
    target_word_count: int = Field(
        description="The estimated ideal word count required to answer this question comprehensively based on its depth or allocated marks."
    )
    requires_code: bool = Field(
        description="True if the question asks for a programmatic solution, algorithm, syntax, or code snippet."
    )
    requires_diagram: bool = Field(
        description="True if the question inherently requires a visual diagram, architecture flow, or graph to explain properly."
    )
    is_comparison: bool = Field(
        description="True if the question asks to compare, contrast, or find the difference between two or more concepts."
    )
    core_entities: List[str] = Field(
        description="The absolute core concepts, algorithms, entities, or terms that MUST be present in the final drafted answer."
    )
# ====================================================================
# Retrieval Grader Output Schema from AcademicAgent's C-RAG workflow
# =====================================================================
class RetrievalGraderOutput(BaseModel):
    """
    Structured output for the Corrective RAG (C-RAG) grader.
    Determines if the retrieved documents are sufficient to answer the question.
    """
    binary_score: Literal["accurate", "not_found", "ambiguous"] = Field(
        description="Whether the retrieved context is relevant and sufficient."
    )
    reasoning: str = Field(
        description="Brief explanation of why this score was given."
    )


# ============================================================
# Diagram Output Schema from AcademicAgent's Diagram Fetcher
# ============================================================
class DiagramOutput(BaseModel):
    diagram_type: Literal["mermaid", "none"] = Field(
        ...,
        description=(
            "Use 'mermaid' for all diagrams. Use 'none' only if the concept "
            "genuinely cannot be represented as a diagram."
        )
    )
    diagram_code: str = Field(
        ...,
        description=(
            "Valid Mermaid diagram code ONLY — no markdown fences, no preamble. "
            "Start directly with the diagram type keyword e.g. 'flowchart TD' or 'classDiagram'. "
            "If diagram_type is 'none', set this to an empty string."
        )
    )
    caption: str = Field(
        ...,
        description="One sentence describing what the diagram shows. Max 15 words."
    )
    fallback_text: str = Field(
        ...,
        description=(
            "A plain-text description of the concept (2-3 sentences) to show "
            "if the diagram cannot be rendered."
        )
    )