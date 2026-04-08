from langchain_core.prompts import ChatPromptTemplate

from DocuAgent.schemas.llm_schemas import PlannerOutput


# =====================================================================================================
# Draft Prompt Builder
# ======================================================================================================
def _instructions_academic(plan: PlannerOutput) -> str:
    return """
        ACADEMIC QUESTION INSTRUCTIONS:
        - Structure: Definition → Core Explanation → Real-world Example → Summary.
        - Use ## headers for each section.
        - Bold (**term**) every key term on first use.
        - Use bullet points for listing properties or characteristics.
        - Minimum 2 cited references from context using [1], [2] etc.
    """.strip()

def _instructions_coding(plan: PlannerOutput) -> str:
    return """
        CODING QUESTION INSTRUCTIONS:
        - Structure: Problem Understanding → Approach/Algorithm → Code → Explanation → Complexity.
        - ALWAYS include a complete, working code block.
        - Use triple-backtick fenced blocks with language tag e.g. ```python ```.
        - Add inline comments inside every code block.
        - End with Big-O time AND space complexity.
        - If multiple approaches exist, show brute-force first then optimal.
    """.strip()

def _instructions_math(plan: PlannerOutput) -> str:
    return """
        MATH QUESTION INSTRUCTIONS:
        - Structure: Given → Formula/Theorem → Step-by-Step Working → Final Answer → Verification.
        - Use LaTeX inline for all expressions: $x^2 + y^2 = z^2$
        - Use LaTeX display blocks for key results: $$\\int_0^\\infty e^{-x} dx = 1$$
        - NEVER write equations in plain text.
        - Show EVERY algebraic step — never skip a transformation.
        - Box or highlight the final answer clearly.
    """.strip()

def _instructions_analytical(plan: PlannerOutput) -> str:
    return """
        ANALYTICAL QUESTION INSTRUCTIONS:
        - Structure: Context → Core Analysis (numbered points) → Evidence → Counter-arguments → Conclusion.
        - Every claim MUST be backed by cited evidence from context [1], [2] etc.
        - Quantify wherever possible — avoid vague terms like "very large" or "quite fast".
        - Use numbered lists for multi-point arguments.
        - Explicitly state both strengths AND weaknesses where relevant.
    """.strip()

def _instructions_comparison(plan: PlannerOutput) -> str:
    col_a = plan.core_entities[0] if len(plan.core_entities) > 0 else "Concept A"
    col_b = plan.core_entities[1] if len(plan.core_entities) > 1 else "Concept B"
    return f"""
        COMPARISON QUESTION INSTRUCTIONS:
        - Structure: Brief Intro → Comparison Table → Key Differences (prose) → Use Cases → Conclusion.
        - YOU MUST include a Markdown comparison table with MINIMUM 5 rows:
        | Aspect | {col_a} | {col_b} |
        |--------|---------|---------|
        - Cover these aspects in the table: Definition, Use Case, Performance, Advantages, Limitations.
        - After the table, explain the 2-3 most important differences in prose.
    """.strip()

def _instructions_factual(plan: PlannerOutput) -> str:
    return """
        FACTUAL QUESTION INSTRUCTIONS:
        - Answer the question DIRECTLY in the very first sentence — never bury the answer.
        - Structure: Direct Answer → Supporting Facts → Additional Context.
        - Be concise and precise — no padding or lengthy introductions.
        - If the context does not contain the fact, explicitly state: "Based on available context..."
        - Cite every fact with [1], [2] etc.
    """.strip()

def _instructions_creative(plan: PlannerOutput) -> str:
    return """
        CREATIVE QUESTION INSTRUCTIONS:
        - Be original, engaging, and use vivid language and strong analogies.
        - Structure the response based on what the question asks (story, essay, brainstorm etc.).
        - Use concrete examples to ground abstract ideas.
        - Still respect the word count and entity constraints provided above.
    """.strip()

# Registry
_INSTRUCTION_REGISTRY = {
        "academic":   _instructions_academic,
        "coding":     _instructions_coding,
        "math":       _instructions_math,
        "analytical": _instructions_analytical,
        "comparison": _instructions_comparison,
        "factual":    _instructions_factual,
        "creative":   _instructions_creative,
    }

#  DIAGRAM PLACEMENT INSTRUCTION
def _diagram_instruction() -> str:
    return """
        DIAGRAM PLACEMENT INSTRUCTIONS:
        Relevant diagrams will be fetched and inserted automatically based on the placeholders you write.

        Rules:
        - Decide HOW MANY diagrams are needed based on the complexity of your answer.
        Simple concept = 1 diagram. Complex multi-part answer = 2-3 diagrams max.
        - For each diagram you need, write {diagram_1}, {diagram_2}, {diagram_3} etc.
        on its own line at the exact point where it is most relevant.
        - Immediately BEFORE each placeholder, write exactly 1 sentence describing
        what that specific diagram should illustrate.
        - NEVER group all placeholders at the end — each must be embedded where it adds value.
        - NEVER use the same placeholder twice.

        Example of correct usage in a multi-concept answer:
        "The following diagram shows the high-level architecture of the system. {diagram_1}
        ...explanation continues...
        The memory layout during execution is shown below. {diagram_2}"
    """.strip()

def build_drafter_prompt(plan: PlannerOutput) -> str:
    """
    Builds the complete drafter system prompt.
    diagram_count removed — diagram placement is driven by plan.requires_diagram only.
    """

    # ── Resolve active category ───────────────────────────────────────────
    active_category = plan.question_category
    if plan.requires_code:
        active_category = "coding"
    elif plan.is_comparison:
        active_category = "comparison"

    # ── SECTION 1: Role & Basic Instructions ─────────────────────────────
    section_1 = """
        You are the Academic Drafter Agent — the final stage of an intelligent academic Q&A pipeline.

        Your job is to produce a single, complete, publication-quality answer to the student's question
        using the planner constraints and retrieved context provided below.

        BASIC INSTRUCTIONS:
        - Write for a university-level student audience.
        - Be factually accurate — only state what is supported by the retrieved context.
        - Never hallucinate facts, formulas, or code not grounded in the context.
        - Always cite the retrieved context using bracketed numbers e.g. [1], [2].
        - Use Markdown formatting throughout your entire answer.
        - Follow the Execution Steps provided by the Planner in the exact order given.
    """.strip()

    # ── SECTION 2: Planner Data ───────────────────────────────────────────
    marks_line    = f"  - Allocated Marks   : {plan.allocated_marks} marks" \
                    if plan.allocated_marks else \
                    "  - Allocated Marks   : Not specified"
    entities_line = "\n".join(f"    ☐ {e}" for e in plan.core_entities) \
                    if plan.core_entities else "    (none specified)"
    steps_line    = "\n".join(f"    {i+1}. {s}" for i, s in enumerate(plan.steps))

    section_2 = f"""
        ═══════════════════════════════════════════
        PLANNER CONSTRAINTS (MUST BE FOLLOWED)
        ═══════════════════════════════════════════
        - Question Category : {active_category.upper()}
        {marks_line}
        - Target Word Count : ~{plan.target_word_count} words
        - Requires Code     : {"YES — include working code block(s)" if plan.requires_code else "NO"}
        - Requires Diagram  : {"YES — place {diagram_1} placeholder as instructed below" if plan.requires_diagram else "NO"}
        - Is Comparison     : {"YES — include a comparison table" if plan.is_comparison else "NO"}

        Core Entities (MUST all appear in your answer):
        {entities_line}

        Execution Steps (follow in this exact order):
        {steps_line}
        ═══════════════════════════════════════════
    """.strip()

    # ── SECTION 3: Category-Specific Instructions ─────────────────────────
    instruction_fn = _INSTRUCTION_REGISTRY.get(active_category, _instructions_academic)
    section_3      = instruction_fn(plan)

    # ── SECTION 4: Diagram Placement (only if planner flagged it) ─────────
    sections = [section_1, section_2, section_3]
    if plan.requires_diagram:
        sections.append(_diagram_instruction())

    return "\n\n".join(sections)


# ==============================================================================================
# Diagram Generator Prompt 
# ===============================================================================================
_DIAGRAM_GENERATOR_SYSTEM_PROMPT = (
    "You are an expert academic diagram generator. "
    "Your job is to produce a clean, valid Mermaid diagram for a given concept.\n\n"

    "DIAGRAM TYPE SELECTION RULES:\n"
    "- Process / algorithm / lifecycle        → flowchart TD\n"
    "- Protocol / message exchange            → sequenceDiagram\n"
    "- Class hierarchy / OOP / inheritance    → classDiagram\n"
    "- State machine / FSM                    → stateDiagram-v2\n"
    "- Entity relationships / DB schema       → erDiagram\n"
    "- Timeline / historical sequence         → timeline\n"
    "- Tree / hierarchy (non-class)           → flowchart TD with subgraph\n"
    "- Comparison / breakdown / pie           → pie or quadrantChart\n\n"

    "STRICT MERMAID RULES:\n"
    "1. Output ONLY raw Mermaid code — no markdown fences, no explanation.\n"
    "2. Node labels must be short (≤5 words). Put detail in edge labels.\n"
    "3. Max 12 nodes total. Diagrams must be readable, not exhaustive.\n"
    "4. No special characters in node IDs — use alphanumeric and underscores only.\n"
    "5. Every node must be connected — no orphan nodes.\n"
    "6. Test mentally: would this render without syntax errors?\n\n"

    "Return ONLY valid JSON matching the DiagramOutput schema. No preamble."
)
_DIAGRAM_GENERATOR_USER_PROMPT =  (
    "Concept: {concept}\n"
    "What to illustrate: {hint}\n"
    "Question category: {question_category}\n\n"
    "Generate the most appropriate Mermaid diagram for this concept."
)
DIAGRAM_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _DIAGRAM_GENERATOR_SYSTEM_PROMPT),
    ("user", _DIAGRAM_GENERATOR_USER_PROMPT)
])