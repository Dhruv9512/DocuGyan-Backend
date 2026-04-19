from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Execution plan produced by the question planner."""

    question: str = Field(
        default="",
        description="The original question — used as fallback for entity extraction."
    )

    question_category: Literal[
        "academic", "coding", "math", "factual",
        "analytical", "creative", "default"
    ] = Field(
        default="default",
        description=(
            "Question category — controls which answer template is used. "
            "Must be 'coding' if requires_code=True."
        )
    )

    allocated_marks: int = Field(
        default=5,
        ge=1,
        le=20,  
        description=(
            "Marks allocated for this question. "
            "Must be a positive integer between 1 and 20. Never null. Default is 5."
        )
    )

    requires_code: bool = Field(
        default=False,
        description=(
            "True if any code, algorithm, pseudocode, or implementation is needed — "
            "even if explanation is also required alongside the code."
        )
    )

    requires_diagram: bool = Field(
        default=False,
        description=(
            "True if a diagram, flowchart, or visual would help a student "
            "understand the concept more clearly than text alone. "
            "True for: processes, flows, architectures, hierarchies, cycles, "
            "state machines, or any sequential working. "
            "False only if the concept is purely theoretical with no visual structure."
        )
    )

    is_comparison: bool = Field(
        default=False,
        description=(
            "True if the question asks to compare, contrast, or distinguish concepts. "
            "When True, core_entities must list every concept being compared separately."
        )
    )

    core_entities: List[str] = Field(
        default_factory=list,
        description=(
            "Every core concept, tool, algorithm, or object the answer must address. "
            "One short noun phrase per entity. Never empty — minimum 1 entity always."
        )
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("allocated_marks", mode="before")
    @classmethod
    def coerce_marks(cls, v):
        """Coerce None or invalid marks to default 5."""
        if v is None:
            return 5
        try:
            return int(v)
        except (TypeError, ValueError):
            return 5

    @field_validator("core_entities", mode="before")
    @classmethod
    def ensure_non_empty_entities(cls, v, info):
        """
        Guarantee core_entities is never empty.
        Falls back to extracting meaningful words from the question.
        """
        if not v:
            question = info.data.get("question", "")
            if question:
                stop_words = {
                    "what", "is", "are", "how", "does", "do", "explain",
                    "describe", "define", "discuss", "the", "a", "an",
                    "of", "in", "and", "or", "with", "for", "to", "that"
                }
                words = question.replace("?", "").replace(",", "").split()
                meaningful = [w for w in words if w.lower() not in stop_words]
                if meaningful:
                    return [" ".join(meaningful[:4])]
            return ["Unknown — re-extract from question"]
        return v

    @model_validator(mode="after")
    def sync_coding_category(self):
        """If requires_code=True, category must be coding."""
        if self.requires_code and self.question_category != "coding":
            self.question_category = "coding"
        return self

# ====================================================================
# Retrieval Grader Output Schema from AcademicAgent's C-RAG workflow
# =====================================================================
class RetrievalGraderOutput(BaseModel):
    binary_score: Literal["accurate", "ambiguous", "not_found"] = Field(
        description=(
            "accurate   → context fully and directly answers the question; no inference needed. "
            "ambiguous  → context is partially relevant but missing critical details or sub-parts. "
            "not_found  → context is irrelevant or off-topic; keyword overlap is not enough."
        )
    )
    reasoning: str = Field(
        description=(
            "1-3 sentences citing specific content from the context "
            "OR naming exactly what is missing. Never speak in generalities."
        )
    )