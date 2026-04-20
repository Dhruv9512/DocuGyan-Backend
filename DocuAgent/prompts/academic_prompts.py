from langchain_core.prompts import ChatPromptTemplate



# =====================================================================================================
# Category-Specific Instruction Builders (Cleaned of Citation Rules)
# =====================================================================================================

def _instructions_academic(plan, question) -> str:
    marks = plan.allocated_marks or 0

    if marks <= 4:
        depth      = "brief"
        per_point  = "2-3 sentences per point"
        conclusion = "1-2 sentences"
        num_points = "2-3"
    elif marks <= 8:
        depth      = "moderate"
        per_point  = "1 short paragraph (4-5 sentences) per point"
        conclusion = "2-3 sentences"
        num_points = "4-5"
    else:
        depth      = "detailed"
        per_point  = "1 full paragraph (6-8 sentences) per point"
        conclusion = "1 short paragraph"
        num_points = "6-8"

    return f"""
        QUESTION YOU ARE ANSWERING:
        \"\"\"{question}\"\"\"

        ACADEMIC ANSWER INSTRUCTIONS:

        This is a {depth.upper()} academic answer worth {marks} marks.

        Structure EXACTLY in this order:

        ## 1. Definition
        - Define the core concept(s) in 2–3 sentences.
        - Bold (**term**) every key term on first use.

        ## 2. Key Points
        - Identify {num_points} key points from the retrieved context that best answer the question.
        - Format as a numbered list:
            1. Point name 
            2. Point name 
        - This section is a ROADMAP only — no explanation needed here.

        ## 3. Explanation of Each Point
        - Dedicate a sub-section to each point from Section 2.
        - Explanation depth: {per_point}.
        - Use bullet points ONLY for listing distinct properties — not for prose.
        - Stay within the targeted word count — do not pad.

        ## 4. Conclusion
        - Summarise in {conclusion}. No new information.
    """.strip()


def _instructions_coding(plan, question) -> str:
    marks = plan.allocated_marks or 0

    # ── 1. Explanation depth based on marks ──────────────────────────────────
    if marks <= 4:
        concept_depth     = "brief — 2 to 3 sentences covering the core idea only"
        walkthrough_depth = "1-2 sentences per code block tracing the key logic"
        complexity_depth  = "one-line justification each"
    elif marks <= 8:
        concept_depth     = "moderate — 1 short paragraph (4-5 sentences) explaining the algorithm and its rationale"
        walkthrough_depth = "3-4 sentences per code block tracing input → process → output with a small example"
        complexity_depth  = "2–3 sentences justifying each, noting best / average / worst case where relevant"
    else:
        concept_depth     = "detailed — 1 full paragraph (6-8 sentences) covering algorithm design, edge cases, and trade-offs"
        walkthrough_depth = "a full paragraph per code block walking through the logic step-by-step with a worked example"
        complexity_depth  = "2–3 sentences justifying each, noting best / average / worst case where relevant"

    # ── 2. Multi-block guidance — driven by content ───────────────
    code_block_rule = """
        - Analyse the question and decide how many code blocks are needed:
            - Single solution needed       → one complete, runnable code block.
            - Brute-force + optimal exist  → two titled blocks:
                #### Brute-Force Approach
                #### Optimal Approach
            - Multiple independent tasks   → one titled block per task:
                #### Task 1 — <purpose>
                #### Task 2 — <purpose>
        - Every code block must be:
            - Complete and independently runnable (no dangling references across blocks).
            - Written clearly so a student can read and understand it easily.
            - Commented on EVERY non-obvious line explaining WHY, not just what.
            - Using meaningful variable and function names — never single letters except loop counters.
        - Place a Markdown heading immediately above every code block to name its purpose.
        - Use triple-backtick fenced blocks with the language tag: ```python, ```java, etc.
    """.strip()

    # ── 3. Hybrid flag — theory explanation needed alongside code ─────────────
    if getattr(plan, 'question_category', '') == 'academic' and marks > 4:
        theory_section = f"""
        ## 1. Conceptual Explanation  ← REQUIRED for this question
        - Explain the concept / algorithm BEFORE writing any code.
        - Depth: {concept_depth}.
        - Bold (**term**) every key term on first use.
        - Cover: what it is, how it works, and why this approach is chosen.
        """.strip()
        section_offset = 2
    else:
        theory_section = ""
        section_offset = 1

    restatement_num = section_offset - 1 if theory_section else 1
    approach_num    = restatement_num + 1
    code_num        = approach_num + 1
    walkthrough_num = code_num + 1
    complexity_num  = walkthrough_num + 1

    return f"""
        CODING ANSWER INSTRUCTIONS  ({marks} marks)

        QUESTION YOU ARE ANSWERING:
        \"\"\"{question}\"\"\"

        Structure EXACTLY in this order:
        {"─" * 60}

        {theory_section + chr(10) + chr(10) if theory_section else ""}\
        ## {restatement_num}. Problem Restatement
        - Restate the problem in your own words in 2–3 sentences.
        - State the expected input / output format and any notable edge cases.

        ## {approach_num}. Approach / Algorithm
        - Describe the algorithm in plain English BEFORE writing code.
        - If multiple approaches exist, introduce each with its own heading:
            ### Brute-Force Approach
            ### Optimal Approach
        - Depth: {concept_depth}.

        ## {code_num}. Code Block(s)
        {code_block_rule}

        ## {walkthrough_num}. Code Walkthrough
        - After EACH code block write a walkthrough.
        - Depth: {walkthrough_depth}.
        - Trace through with a small concrete example (show values changing at key steps).
        - Explain the WHY behind the logic — do not just restate the code in English.

        ## {complexity_num}. Complexity Analysis
        - Provide explicit Big-O time AND space complexity for every approach shown.
        - Depth: {complexity_depth}.
        - Format as a table when more than one approach is present:

            | Approach      | Time   | Space  |
            |---------------|--------|--------|
            | Brute-Force   | O(...) | O(...) |
            | Optimal       | O(...) | O(...) |

        {"─" * 60}

        HARD RULES (apply to every coding answer):
        - Code must be complete, correct, and runnable as-is — never pseudocode unless explicitly asked.
        - Never truncate code with "# ... rest of the code" or similar — write it fully.
        - Every data structure and algorithm choice must be justified in the Approach section.
        - If the context does not specify a language, default to Python.
        - If the question has multiple independent parts, give each its own titled code block and walkthrough.
    """.strip()

def _instructions_math(plan, question) -> str:
    marks = plan.allocated_marks or 0

    # ── 1. Depth calibration ─────────────────────────────────────────────────
    if marks <= 4:
        depth   = "show each transformation in 1–2 lines; skip trivial arithmetic"
        verify_depth = "1-sentence sanity check (plug answer back in or check units)"
        explanation  = "brief — label each step in 3–5 words"
    elif marks <= 8:
        depth   = "show every algebraic/calculus transformation on its own line with a short label"
        verify_depth = "2–3 sentences: verify numerically OR prove boundary/limit behaviour"
        explanation  = "moderate — 1 sentence of reasoning after each non-obvious step"
    else:
        depth   = "show every single transformation, sub-step, and intermediate result on its own line"
        verify_depth = "full paragraph: verify using an alternative method or check all edge cases"
        explanation  = "detailed — explain the mathematical reasoning behind every key step"

    # ── 2. Detect problem flavour from plan entities ──────────────────────────
    raw_text = " ".join((plan.core_entities or [])).lower()

    has_stats      = any(w in raw_text for w in ["mean", "median", "mode", "variance", "std", "probability",
                                                   "distribution", "regression", "correlation", "frequency"])
    has_calculus   = any(w in raw_text for w in ["integral", "derivative", "limit", "differentiat",
                                                   "series", "converge", "diverge", "taylor"])
    has_linear_alg = any(w in raw_text for w in ["matrix", "vector", "determinant", "eigenvalue",
                                                   "transpose", "inverse", "rank", "span"])
    has_geometry   = any(w in raw_text for w in ["area", "volume", "angle", "triangle", "circle",
                                                   "perimeter", "surface", "coordinate", "slope"])
    has_sets       = any(w in raw_text for w in ["set", "union", "intersection", "subset", "complement",
                                                   "cardinality", "venn"])

    # ── 3. Table rule ─────────────────────────────────────────────────────────
    table_rule = ""
    if has_stats or marks > 6:
        table_rule = """
            ## (Optional) Summary Table
            - Use a Markdown table when the problem involves:
                - Multiple variables / cases side by side  →  | Variable | Formula | Value |
                - Frequency / probability distributions   →  | x | f(x) | P(X=x) | Cumulative |
                - Comparison of methods or results        →  | Method | Result | Note |
            - Keep every cell concise (one value or short expression).
            - Use LaTeX inside table cells: | $\\mu$ | $\\sum x_i / n$ | $4.2$ |
            - Place the table AFTER the formula block and BEFORE step-by-step working.
        """.strip()

    # ── 4. Flavour-specific formula hints ────────────────────────────────────
    flavour_hints = []
    if has_stats:
        flavour_hints.append("""\
            Statistics / Probability detected:
            - State the distribution type and its parameters first (e.g. $X \\sim N(\\mu, \\sigma^2)$).
            - Show ALL intermediate sums/products before substituting into the formula.
            - Express probabilities as both fractions and decimals where possible.
        """)

    if has_calculus:
        flavour_hints.append("""\
            Calculus detected:
            - State the rule being applied at each step (e.g. "Applying integration by parts").
            - For definite integrals: evaluate the antiderivative first, THEN apply limits.
            - For limits: state the form (0/0, ∞/∞, etc.) before applying L'Hôpital or factoring.
        """)

    if has_linear_alg:
        flavour_hints.append("""\
            Linear Algebra detected:
            - Write matrices using LaTeX \\begin{bmatrix}...\\end{bmatrix}.
            - Show row operations explicitly: $R_2 \\leftarrow R_2 - 2R_1$.
            - State the property used at each step (e.g. "det(AB) = det(A)·det(B)").
        """)

    if has_geometry:
        flavour_hints.append("""\
            Geometry detected:
            - Label all known values on a clearly described figure (in text, not image).
            - State the geometric theorem or postulate before applying it.
            - Include units on every intermediate and final value.
        """)

    if has_sets:
        flavour_hints.append("""\
            Set Theory detected:
            - List elements explicitly when the sets are small (|S| ≤ 10).
            - Draw ASCII Venn diagrams only when it genuinely aids clarity.
            - Verify using |A ∪ B| = |A| + |B| − |A ∩ B| where applicable.
        """)

    flavour_block = (
        "\n\n### Problem-Type Guidance\n" +
        "\n\n".join(f"- {h}" for h in flavour_hints)
    ) if flavour_hints else ""

    # ── 5. Frontend-safe LaTeX rules ─────────────────────────────────────────
    latex_rules = r"""
        ### LaTeX / Rendering Rules  (frontend-safe)
        - Inline math  → single dollar signs  : $x^2 + y^2 = z^2$
        - Display math → double dollar signs  : $$\int_0^\infty e^{-x}\,dx = 1$$
        - NEVER use plain text for any math expression — no "x^2", always "$x^2$".
        - NEVER use \[ \] or \( \) delimiters — only $ and $$ (KaTeX / MathJax compatible).
        - Escape backslashes in fractions    : $\frac{a}{b}$
        - Matrices                           : $$\begin{bmatrix} a & b \\ c & d \end{bmatrix}$$
        - Align multi-line working with the align environment:
            $$\begin{align}
            2x + 3 &= 11 \\
            2x     &= 8  \\
            x      &= 4
            \end{align}$$
        - Greek letters, operators, arrows   : $\alpha, \beta, \sum, \int, \rightarrow, \leq$
        - Absolute value / norm              : $|x|$  or  $\|v\|$
    """.strip()

    # ── Assemble ──────────────────────────────────────────────────────────────
    return f"""
        MATH ANSWER INSTRUCTIONS  ({marks} marks)

        QUESTION YOU ARE ANSWERING:
        \"\"\"{question}\"\"\"

        Structure EXACTLY in this order:
        {"─" * 60}

        ## 1. Given / Known Information
        - List every value, constraint, and condition stated in the question.
        - Use a bullet per item: "- $n = 50$,  $\\bar{{x}} = 4.2$,  $\\sigma = 1.1$"
        - Do NOT start solving here — this section is setup only.

        ## 2. Formula / Theorem
        - State the formula(s) or theorem(s) that will be applied — BEFORE any substitution.
        - Use display blocks for every formula:
            $$\\text{{Formula name}} = \\ldots$$
        - If multiple formulas are needed, number them:
            $$\\text{{(1) }} E = mc^2$$
            $$\\text{{(2) }} F = ma$$
        - Cite the name of the theorem or rule (e.g. "Bayes' Theorem", "Chain Rule").

        {table_rule}

        ## 3. Step-by-Step Working
        - Depth: {depth}.
        - Number every step: **Step 1**, **Step 2**, …
        - On each step:
            a. Write the algebraic expression BEFORE simplification on one line.
            b. Write the result AFTER simplification on the next line.
            c. Add a short label explaining what was done: *(substitute $n=50$)*, *(expand brackets)*.
        - Use the LaTeX align environment for multi-line transformations (see LaTeX rules below).
        - NEVER skip a step — show every intermediate result.
        - For each non-obvious step add: {explanation}.

        ## 4. Final Answer
        - Wrap in bold on its own line:
            **Final Answer: $\\ldots$**
        - If the answer has units, include them inside the LaTeX: $42\\,\\text{{km/h}}$.
        - If the problem has multiple parts (a, b, c), give a final answer block for each part.

        ## 5. Verification
        - {verify_depth}
        - Show the check as a mini calculation, not just a statement.
        - If verification is not algebraically possible, state WHY and use a boundary/limit check instead.

        {flavour_block}

        {"─" * 60}
        {latex_rules}
        {"─" * 60}

        HARD RULES (apply to every math answer):
        - Every number, variable, operator, and expression → inside LaTeX. No exceptions.
        - Never use Unicode math symbols (×, ÷, √, ∞) in prose — use $\\times$, $\\div$, $\\sqrt{{}}$, $\\infty$.
        - Never write "ans =", "result =" in plain text — always use LaTeX and the Final Answer block.
        - If the context does not contain enough data to solve the problem, state exactly what is missing.
    """.strip()

def _instructions_comparison(plan, question) -> str:
    marks = plan.allocated_marks or 0

    if marks <= 4:
        depth = "brief"
        row_guidance = "Focus on the most important distinguishing aspects only. Aim for fewer, high-quality rows."
        prose_depth = "1 short paragraph per section"
    elif marks <= 8:
        depth = "moderate"
        row_guidance = "Cover all major aspects that meaningfully differentiate the concepts."
        prose_depth = "2 paragraphs per section"
    else:
        depth = "detailed"
        row_guidance = "Cover all aspects comprehensively — definition, working, performance, use cases, advantages, limitations, and examples."
        prose_depth = "2-3 paragraphs per section"

    return f"""
        QUESTION YOU ARE ANSWERING:
        \"\"\"{question}\"\"\"

        COMPARISON ANSWER INSTRUCTIONS:

        This is a {depth.upper()} comparison answer worth {marks} marks.
        This question is a FULL COMPARISON — the entire answer must be structured as a comparison.

        Structure EXACTLY:
        1. Brief Introduction  (~2–3 sentences, no lists)
        2. Comparison Table    (see rules below)
        3. Key Differences     (prose, {prose_depth} on the most important distinctions)

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        COMPARISON TABLE RULES:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Columns:
        - Read the question carefully and identify ALL distinct concepts/entities being compared.
        - Create one column per concept. Minimum 2 concept columns, no upper limit.
        - Use this skeleton — expand columns as needed:

        | Aspect | Concept A | Concept B | (add more columns if the question compares more concepts) |
        |--------|-----------|-----------|-----------------------------------------------------------|

        Rows:
        - Do NOT use a fixed row count. Let the marks and context decide:
          {row_guidance}
        - Hard maximum: 9 rows. Never exceed this.
        - Choose aspects most relevant and distinguishable for THIS specific question.
        - Every row must be grounded in the retrieved context — never invent aspects.
        - Fill every cell; write "N/A" only if context has no data for that cell.
        - Keep each cell concise (one phrase or short sentence).
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """.strip()


def _instructions_analytical(plan, question: str = "") -> str:
    marks = plan.allocated_marks or 0

    # ── 1. Depth calibration ─────────────────────────────────────────────────
    if marks <= 4:
        depth          = "brief"
        analysis_depth = "2–3 sentences per point"
        evidence_depth = "1 supporting fact or figure per point"
        conclusion     = "1–2 sentences summarising the key finding"
    elif marks <= 8:
        depth          = "moderate"
        analysis_depth = "1 short paragraph (4–5 sentences) per point"
        evidence_depth = "2–3 supporting facts, figures, or examples per point"
        conclusion     = "2–3 sentences covering the overall implication"
    else:
        depth          = "detailed"
        analysis_depth = "1 full paragraph (6–8 sentences) per point"
        evidence_depth = "multiple pieces of evidence including counter-evidence where relevant"
        conclusion     = "1 full paragraph covering implications, limitations, and recommendations"

    # ── 2. Detect analytical flavour from question + plan entities ────────────
    raw_text = " ".join(
        (plan.core_entities or []) + [question]
    ).lower()

    has_causal      = any(w in raw_text for w in ["cause", "effect", "impact", "result", "lead to",
                                                    "consequence", "due to", "reason", "why"])
    has_evaluative  = any(w in raw_text for w in ["evaluate", "assess", "judge", "critique", "effective",
                                                    "success", "failure", "worth", "merit", "limitation"])
    has_problem     = any(w in raw_text for w in ["problem", "challenge", "issue", "barrier", "obstacle",
                                                    "risk", "threat", "weakness", "drawback"])
    has_strategy    = any(w in raw_text for w in ["strategy", "recommend", "suggest", "propose", "solution",
                                                    "improve", "plan", "approach", "measure"])
    has_comparative = any(w in raw_text for w in ["compare", "contrast", "differ", "similar", "versus",
                                                    "better", "worse", "advantage", "disadvantage"])

    # ── 3. Flavour-specific guidance ─────────────────────────────────────────
    flavour_hints = []

    if has_causal:
        flavour_hints.append(
            "Causal Analysis detected:\n"
            "  - Clearly separate CAUSES from EFFECTS — never conflate them.\n"
            "  - Use causal language precisely: 'leads to', 'results in', 'is caused by'.\n"
            "  - Distinguish direct causes from indirect/contributing factors.\n"
            "  - If a chain of causation exists, trace it step-by-step."
        )
    if has_evaluative:
        flavour_hints.append(
            "Evaluative Analysis detected:\n"
            "  - State your evaluative criteria FIRST (what counts as 'effective' or 'successful').\n"
            "  - Assess BOTH strengths and weaknesses — never one-sided.\n"
            "  - Use a balanced structure: present the case FOR, then AGAINST, then your judgement.\n"
            "  - Ground every judgement in evidence from the context — no opinion without support."
        )
    if has_problem:
        flavour_hints.append(
            "Problem Analysis detected:\n"
            "  - Define the problem precisely in 1–2 sentences before analysing it.\n"
            "  - Separate symptoms (what is visible) from root causes (why it happens).\n"
            "  - Quantify the scale or severity of the problem where data is available.\n"
            "  - Identify who is affected and how."
        )
    if has_strategy:
        flavour_hints.append(
            "Strategic / Recommendation Analysis detected:\n"
            "  - State the goal or desired outcome that the strategy addresses.\n"
            "  - For each recommendation: explain WHAT it is → WHY it helps → HOW to implement it.\n"
            "  - Acknowledge trade-offs or risks associated with each recommendation.\n"
            "  - Prioritise recommendations if more than one is given."
        )
    if has_comparative:
        flavour_hints.append(
            "Comparative Analysis detected:\n"
            "  - Use consistent criteria across all items being compared.\n"
            "  - Do NOT alternate randomly — analyse one dimension at a time across all items.\n"
            "  - Explicitly state which is stronger/weaker on each dimension and why.\n"
            "  - Avoid false equivalence — note when the difference is significant vs marginal."
        )

    flavour_block = (
        "\n### Question-Type Guidance (auto-detected)\n" +
        "\n\n".join(f"- {h}" for h in flavour_hints)
    ) if flavour_hints else ""

    return f"""
        ANALYTICAL ANSWER INSTRUCTIONS  ({marks} marks)

        QUESTION YOU ARE ANSWERING:
        \"\"\"{question}\"\"\"

        Read the question above carefully before writing anything:
        - Identify WHAT is being analysed (a concept, event, policy, system, argument, etc.).
        - Identify the analytical task: explain / evaluate / critique / recommend / compare.
        - Identify any constraints: time period, scope, stakeholder perspective.
        - Use the exact terminology and entities from the question throughout your answer.

        {"─" * 60}

        ## 1. Context
        - Set the scene in {("2–3 sentences" if marks <= 4 else "1 short paragraph")}.
        - Define the subject of analysis precisely — what it is, why it matters, what is at stake.
        - Do NOT begin the analysis here — this section is background only.
        - Bold (**term**) every key term on first use.

        ## 2. Analysis  (numbered points)
        - Identify the most important analytical dimensions from the retrieved context
          and use those as your numbered points. Scale the number of points to the marks:
          {("2–3 focused points" if marks <= 4 else "4–5 well-developed points" if marks <= 8 else "6–8 comprehensive points")}.
        - Format EXACTLY as:
            **1. [Point Name]**
            [Analysis — {analysis_depth}]

            **2. [Point Name]**
            [Analysis — {analysis_depth}]
        - Use numbered lists for the point headers; write the body as prose, NOT bullet points.
        - Every claim must be grounded in the retrieved context — no unsupported assertions.
        - Quantify wherever possible — replace vague terms like "very large" or "significant"
          with actual figures, percentages, or timeframes from the context.

        ## 3. Evidence
        - For each point in Section 2, provide {evidence_depth}.
        - Tie each piece of evidence explicitly to the claim it supports.
        - If the context contains conflicting evidence, acknowledge it — do not cherry-pick.
        - Format: "This is supported by [evidence] [citation]."

        ## 4. Counter-arguments / Limitations
        - Present at least {"1 counter-argument" if marks <= 4 else "2–3 counter-arguments or limitations"}.
        - Structure: State the counter-argument → Acknowledge its validity → Explain why your
          analysis still holds (or concede if it does not).
        - Do NOT dismiss counter-arguments — engage with them seriously.
        - If the question does not invite counter-arguments, reframe this section as
          "Limitations of the Analysis" and state what the context does not cover.

        ## 5. Conclusion
        - {conclusion}.
        - Do NOT introduce new information here.
        - {"State only the single most important takeaway." if marks <= 4 else "Cover: key finding → overall implication → any recommended next step or open question."}

        {flavour_block}

        {"─" * 60}

        HARD RULES (apply to every analytical answer):
        - Never use vague qualifiers without quantification: "very", "quite", "somewhat", "many".
        - Never assert causation without evidence — distinguish correlation from causation explicitly.
        - Never introduce facts not present in the retrieved knowledge base.
        - If the context is insufficient to support a point, state exactly what is missing
          rather than guessing.
    """.strip()

def _instructions_factual(plan, question) -> str:
    return """
        QUESTION YOU ARE ANSWERING:
        \"\"\"{question}\"\"\"

        FACTUAL ANSWER INSTRUCTIONS:
        - Answer DIRECTLY in the very first sentence — never bury the lead.
        - Structure EXACTLY: Direct Answer → Supporting Facts → Additional Context (if needed).
        - Be concise and precise. No padding, no lengthy introductions.
        - If context does not contain the fact, state: "The available context does not directly address this."
    """.strip()


def _instructions_creative(plan, question) -> str:
    return """
        QUESTION YOU ARE ANSWERING:
        \"\"\"{question}\"\"\"


        CREATIVE ANSWER INSTRUCTIONS:
        - Be original, engaging, and use vivid language and strong analogies.
        - Structure the response based on the specific ask (story, essay, brainstorm, etc.).
        - Use concrete examples to ground abstract ideas.
        - Respect the word count and entity constraints provided.
    """.strip()


def _instructions_general(plan, question: str = "") -> str:
    marks = plan.allocated_marks or 0

    if marks <= 4:
        depth         = "brief"
        section_depth = "2–3 sentences per section"
        conclusion    = "1–2 sentences"
    elif marks <= 8:
        depth         = "moderate"
        section_depth = "1 short paragraph (4–5 sentences) per section"
        conclusion    = "2–3 sentences"
    else:
        depth         = "detailed"
        section_depth = "1 full paragraph (6–8 sentences) per section"
        conclusion    = "1 full paragraph"

    return f"""
        GENERAL ANSWER INSTRUCTIONS  ({marks} marks)

        QUESTION YOU ARE ANSWERING:
        \"\"\"{question}\"\"\"

        {"─" * 60}

        BEFORE YOU WRITE ANYTHING — READ AND DECIDE:

        You are the judge of structure here. No fixed template is imposed.
        Read the question above and decide:

        1. WHAT TYPE of question is this?
        - Definition / explanation       → define first, then explain with examples
        - Analysis / evaluation          → context → analysis → evidence → conclusion
        - Problem / solution             → problem → cause → solution → trade-offs
        - Process / how-it-works         → ordered steps with reasoning at each stage
        - Opinion / argument             → claim → evidence → counter → conclusion
        - Mixed / multi-part             → treat each part separately with its own sub-heading
        - None of the above              → choose the structure that best serves the question

        2. WHAT DEPTH does the question require?
        - Depth level   : {depth} ({marks} marks)
        - Per section   : {section_depth}
        - Conclusion    : {conclusion}

        3. WHAT FORMAT best serves the answer?
        - Use **prose paragraphs** for reasoning, explanation, and argument.
        - Use **numbered lists** for steps, ranked points, or sequential processes.
        - Use **tables** for comparisons, properties, or structured data.
        - Use **code blocks** (```language) if any code, pseudocode, or syntax is needed.
        - Use **LaTeX** ($...$  /  $$...$$) if any mathematical expression appears.
        - Use **bold** (**term**) for every key term on first use.
        - Mix formats freely — let the content decide, not habit.

        {"─" * 60}

        ## STRUCTURE RULES

        - Infer the best structure directly from the question and marks weight.

        ### Opening
        - Start with the most direct, precise answer to the question — never bury the lead.
        - If a definition is needed, give it in the first 1–2 sentences.
        - Do NOT start with "In this answer I will..." or any meta-commentary.

        ### Body
        - Build the answer section by section based on the structure you inferred.
        - Every claim must be grounded in the retrieved knowledge base — no hallucination.
        - Quantify wherever possible — replace "many", "large", "significant" with actual
          figures, percentages, or ranges from the context.
        - If the question has multiple parts (a, b, c / i, ii, iii), give each part
          its own clearly labelled sub-section.

        ### Conclusion
        - Close in {conclusion}.
        - Summarise the key takeaway — do NOT introduce new information.
        - If the question asks for a recommendation or judgement, state it clearly here.

        {"─" * 60}

        HARD RULES (non-negotiable):
        - Only use facts present in the retrieved knowledge base. Never hallucinate.
        - If the context does not contain enough information to answer fully,
          state exactly what is missing rather than guessing.
        - Match terminology exactly to what the question uses.
        - Every factual claim must carry an inline JSON citation immediately after it.
        - Do NOT add diagrams or images.
    """.strip()

# Instruction registry to map question categories to their specific instruction builders
_INSTRUCTION_REGISTRY = {
    "academic":   _instructions_academic,
    "coding":     _instructions_coding,
    "math":       _instructions_math,
    "analytical": _instructions_analytical,
    "comparison": _instructions_comparison,
    "factual":    _instructions_factual,
    "creative":   _instructions_creative,
    "default":    _instructions_general,
}



# Citation instruction is the same across all categories, so we can define it once here
CITATION_INSTRUCTION = """
    ## Citation Rules (STRICT — INLINE JSON FORMAT)

    Every factual claim MUST be followed immediately by an inline JSON citation array.

    CITATION FORMAT RULES:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The retrieved knowledge base gives you two fields per chunk:
        - "source_url"  : the URL or filename of the source
        - "page_number" : the page number (only present for document sources)

    Use these EXACT values to build citations as follows:

    CASE 1 — Web source (source_url starts with "http"):
        page_number is empty, source_url is the full URL.
        → [{"source_url": "<exact source_url from kb>", "page_number": ""}]

    CASE 2 — Document source (source_url is a filename like "notes.pdf"):
        source_url becomes empty, page_url is built as: "<source_url>?page=<page_number>"
        → [{"source_url": "", "page_url": "<exact source_url from kb>?page=<page_number from kb>"}]

    MULTIPLE SOURCES for one claim:
        → [{"source_url": "https://example.com", "page_number": ""}, {"source_url": "", "page_url": "notes.pdf?page=3"}]

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STRICT RULES:
    - Copy source_url and page_number EXACTLY as they appear in the Retrieved Knowledge Base.
    - NEVER invent, guess, or modify a URL — only cite what is in the knowledge base.
    - Place the citation array immediately after the claim — before the full stop.
    - Never group citations at the end of a paragraph — each claim gets its own citation.
    - If a fact comes from multiple chunks of the same source, cite that source once.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    EXAMPLES (based on actual kb values):

    Web source example:
    "Cloud computing reduces infrastructure costs significantly
    [{"source_url": "https://aws.amazon.com/what-is-cloud", "page_number": ""}]."

    Document source example:
    "Virtualization creates isolated virtual environments
    [{"source_url": "", "page_url": "cloud_computing.pdf?page=14"}]."

    Mixed sources example:
    "Load balancing distributes traffic across servers
    [{"source_url": "https://aws.amazon.com/load-balancing", "page_number": ""}, {"source_url": "", "page_url": "cloud_notes.pdf?page=7"}]."
""".strip()

SECTION_ROLE = """
    You are an academic answer drafter. You will be given:
    - A question from a university-level student.
    - A knowledge base of retrieved context chunks.
    - A structured answer plan with constraints you MUST follow.

    BASIC RULES:
    - Only use facts present in the retrieved knowledge base. Never hallucinate.
    - Write in Markdown throughout (headers, bold, code blocks, tables as needed).
    - Follow the answer plan constraints exactly (marks detail level).
    - Every factual claim must carry an inline JSON citation array mapped to the knowledge base.
    - Do NOT add diagrams or images.

    NUMBERING RULE — CRITICAL:
    - The question itself already has a number (e.g. "Q1", "1.", "Question 3").
    - To avoid confusion, your answer must NEVER use plain Arabic numerals (1. 2. 3.)
      for section headers, key points, or any structural list.
    - Use these alternatives instead:
        → Section headers  : Markdown ## / ### headings   (## Definition, ## Explanation)
        → Ordered points   : letters with parenthesis     (a)  b)  c))
        → Sequential steps : "Step A", "Step B", "Step C"
        → Ranked items     : Roman numerals               (i.  ii.  iii.)
    - Arabic numerals are ONLY allowed inside:
        → Code blocks
        → Math / LaTeX expressions
        → Citation JSON arrays
""".strip()



# =====================================================================================================
# Main User Prompt Builder ->  Version 1
# =====================================================================================================
def build_drafter_user_prompt(
    question: str,
    context_chunks: list[dict],
    plan,
) -> str:

    # ── 1. Resolve active category ────────────────────────────────────────────
    active_category = plan.question_category
    if plan.requires_code:
        active_category = "coding"
    elif plan.is_comparison:
        active_category = "comparison"

    # ── 2. Role & basic instructions ──────────────────────────────────────────
    section_role = SECTION_ROLE

    # ── 3. Question ───────────────────────────────────────────────────────────
    section_question = f"""
        ## Question
        {question}
    """.strip()

    # ── 4. Retrieved Knowledge Base ───────────────────────────────────────────
    import json

    kb_items = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_url  = chunk.get("source_url", "")
        page_number = str(chunk.get("page_number", "")).strip()

        # Build the citation reference the model should use for this chunk
        if source_url.startswith("http"):
            citation_ref = {"source_url": source_url, "page_number": ""}
        else:
            page_url = f"{source_url}?page={page_number}" if page_number else source_url
            citation_ref = {"source_url": "", "page_url": page_url}

        kb_items.append({
            "chunk_id":     i,
            "source_url":   source_url,
            "page_number":  page_number,
            "citation_ref": citation_ref,   # ← exact JSON the model must paste
            "page_content": chunk.get("page_content", ""),
        })

    section_kb = f"""
        ## Retrieved Knowledge Base

        These chunks are your ONLY permitted sources.
        Each chunk has a "citation_ref" field — copy it EXACTLY as your inline citation for any claim drawn from that chunk.

        {json.dumps(kb_items, indent=2, ensure_ascii=False)}
    """.strip()

    # ── 5. Answer Plan ────────────────────────────────────────────────────────
    marks_line = (
        f"- Allocated Marks : {plan.allocated_marks} "
        f"(calibrate depth and detail to this mark weight)"
        if plan.allocated_marks
        else "- Allocated Marks : Not specified"
    )

    entities_block = (
        "\n".join(f"  - {e}" for e in plan.core_entities)
        if plan.core_entities
        else "  (none specified)"
    )

    flags = []
    if plan.requires_code:
        flags.append("- Requires Code    : YES — include working, runnable code block(s)")
    if plan.is_comparison:
        flags.append("- Is Comparison    : YES — a comparison table is mandatory")
    flags_block = "\n".join(flags) if flags else ""

    section_plan = f"""
        ## Answer Plan (MUST BE FOLLOWED)

        - Question Category : {active_category.upper()}
        {marks_line}
        {flags_block}

        ### Core Entities
        Every entity below MUST appear in your answer:
        {entities_block}
    """.strip()

    # ── 6. Category-Specific Instructions ────────────────────────────────────
    instruction_fn   = _INSTRUCTION_REGISTRY.get(active_category, _instructions_general)
    section_category = f"""
        ## Category-Specific Instructions
        {instruction_fn(plan, question)}
    """.strip()

    # ── 7. Citation Format ────────────────────────────────────────────────────
    section_citations = CITATION_INSTRUCTION

    # ── Assemble ──────────────────────────────────────────────────────────────
    sections = [
        section_role,
        section_question,
        section_kb,
        section_plan,
        section_category,
        section_citations,
    ]

    return "\n\n---\n\n".join(sections)











# =====================================Question Planner Prompt: version 1=======================================
QUESTION_PLANNER_SYSTEM_PROMPT = """
    You are an elite academic exam planner.

    You will receive a QUESTION and optionally a CONTEXT (retrieved knowledge base chunks).
    Your job is to read both carefully and return a precise JSON execution plan.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    THINK STEP-BY-STEP BEFORE RESPONDING — follow this order:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    STEP 1 — UNDERSTAND THE QUESTION
    - What is being asked? (definition / explanation / calculation / implementation / comparison / evaluation)
    - How many distinct parts does the question have?
    - What are the core subjects, concepts, tools, or objects involved?

   STEP 2 — DETECT MARKS (MUST return an integer between 1 and 20, never null)
    - Explicit  : marks stated in the prompt → use that exact number.
    - Inferred  : deduce from question complexity:
                    1-2  → single definition or one-liner
                    3-4  → short explanation, 1-2 concepts
                    5-6  → multi-concept explanation
                    7-8  → detailed discussion or comparison
                    9-10 → full essay, multi-part, or design question
                    11-20 → only if explicitly stated in the question
    - Default   : if marks cannot be determined → return 5.
    - NEVER return null. NEVER return a string. NEVER return a value outside 1-20.

    STEP 3 — DECIDE THE CATEGORY (pick EXACTLY one)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "academic"   → concept explanation, definition, theory, how-something-works.
    "coding"     → requires ANY code, algorithm, pseudocode, or implementation.
    "math"       → numerical problems, proofs, derivations, formula-based calculations.
    "factual"    → direct who/what/when/where with a single retrievable answer.
    "analytical" → evaluate, assess, critique, cause-effect, recommend, argue.
    "creative"   → open-ended, brainstorm, design, narrative, hypothetical.
    "default"    → genuinely unclear or does not fit any category above.

    PRIORITY RULES:
    - If requires_code=True → category MUST be "coding". No exceptions.
    - When in doubt → return "default". Never guess a wrong category.

    STEP 4 — SET THE FLAGS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    requires_code:
        → True  if the question needs any code, algorithm, pseudocode, or implementation,
                even if explanation is also required alongside it.
        → False only if absolutely no code is needed.
        → REMINDER: if requires_code=True, you MUST also set question_category="coding".

    requires_diagram:
        → True  if a diagram, flowchart, or visual would help a student
                understand the concept more clearly than text alone.
        → True  for: processes, flows, architectures, hierarchies,
                cycles, state machines, or any sequential working.
        → False only if the concept is purely theoretical or text-based
            with no visual structure to represent.

    is_comparison:
        → True  for: "difference between", "compare", "contrast", "vs", "distinguish".
        → False for: "advantages and disadvantages of X" — that is analytical, not comparison.
        → False for: "pros and cons of X" — also analytical, not comparison.
        → When True: core_entities MUST list every concept being compared separately.

    STEP 5 — IDENTIFY CORE ENTITIES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    List every distinct concept, tool, algorithm, or object the answer MUST address.
    - One short noun phrase per entity — never a full sentence.
    - If is_comparison=True  → one entity per concept being compared (minimum 2).
    - If requires_code=True  → include the algorithm or data structure name.
    - If multi-part question → include the subject of each part.
    - NEVER return an empty list. If uncertain, extract the main subject
    noun directly from the question and use that as the single entity.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OUTPUT RULES:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Return ONLY valid JSON matching this exact schema — no preamble, no markdown fences:

    {{
    "question"          : <string>  — copy the original question verbatim,
    "question_category" : <string>  — exactly one of: academic | coding | math | factual | analytical | creative | default,
    "allocated_marks"   : <integer> — value between 1 and 20, never null, never a string,
    "requires_code"     : <boolean> — true or false,
    "requires_diagram"  : <boolean> — true or false,
    "is_comparison"     : <boolean> — true or false,
    "core_entities"     : <list>    — non-empty list of short noun-phrase strings
    }}

    CROSS-CHECK BEFORE RETURNING:
    - requires_code=true  → question_category MUST be "coding".
    - is_comparison=true  → core_entities MUST have at least 2 entries.
    - allocated_marks     → MUST be an integer between 1 and 100.
    - core_entities       → MUST NOT be an empty list.
    - question            → MUST be a copy of the original input question.
""".strip()


QUESTION_PLANNER_HUMAN_PROMPT = """
QUESTION:
{question}

CONTEXT:
{context}

YOUR TASK:
- Read the QUESTION and CONTEXT carefully.
- Follow the step-by-step thinking process from the system prompt.
- Return a precise JSON execution plan with ALL required fields.
- Cross-check all fields against the OUTPUT RULES before responding.
""".strip()

QUESTION_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", QUESTION_PLANNER_SYSTEM_PROMPT),
        ("human",  QUESTION_PLANNER_HUMAN_PROMPT),
    ])


# ===========================================Diagram queary prompt: version 1=======================================
DIAGRAM_INJECTOR_USER_PROMPT = """
    ## Original Question
    {question}

    ## Draft Answer
    {draft}

    ## Your Task
    Read the draft above and inject diagram placeholders where a visual would DIRECTLY
    help a university student understand the concept being explained at that point.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PLACEMENT RULES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Decide the number of diagrams based on content:
       - Simple / single-concept answer        → 1 diagram
       - Multi-step process or layered concepts → 2-3 diagrams

    2. Place each diagram ONLY where it genuinely aids understanding:
       - GOOD: a flowchart right after explaining a multi-step process
       - GOOD: an architecture diagram after describing system components
       - BAD : a generic diagram at the very start before any explanation
       - BAD : a diagram placed in the conclusion

    3. For each diagram, insert EXACTLY TWO lines on their own line, in this order:
       [Query: <specific descriptive search query>]
       {{diagram_N}}

       Where N starts at 1 and increments: {{diagram_1}}, {{diagram_2}}, {{diagram_3}}, etc.

    4. Query accuracy rules:
       - The query MUST describe the SPECIFIC concept in the paragraph directly above it.
       - Always include the diagram type: "flowchart", "architecture diagram",
         "sequence diagram", "memory layout", "block diagram", "ER diagram", etc.
       - BAD : [Query: computer science diagram]
       - GOOD: [Query: TCP three-way handshake SYN ACK sequence diagram]

    5. If you cannot form a specific, accurate query for a position — skip it entirely.
       Only place a diagram where you can write a precise, concept-specific query.

    6. NEVER place all diagram placeholders at the end of the answer.
    7. NEVER reuse the same placeholder ({{diagram_1}} can only appear once).

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    EXAMPLE OF CORRECT INJECTION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Before injection:
        "The handshake proceeds in three steps as described below.
        After the connection is established, data transfer begins."

    After injection:
        "The handshake proceeds in three steps as described below.
        [Query: TCP three-way handshake SYN ACK sequence diagram]
        {{diagram_1}}
        After the connection is established, data transfer begins."

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OUTPUT RULES (CRITICAL)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Return the COMPLETE draft with diagram placeholders injected.
    - Do NOT summarise, shorten, rewrite, or change any word of the draft.
    - Do NOT add any commentary before or after the draft.
    - Preserve ALL inline citation JSON arrays exactly as they appear — do not move,
      reformat, or remove any citation.
    - Only insert the [Query: ...] line and {{diagram_N}} placeholder — nothing else.
""".strip()

DIAGRAM_INJECTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("human", DIAGRAM_INJECTOR_USER_PROMPT)
])



# ==========================================Retrieval Grader Prompt: version 1=======================================
RETRIEVAL_GRADER_SYSTEM_PROMPT = """
You are a Retrieval Grader in an academic RAG pipeline.

Your ONLY job: decide whether the retrieved context contains enough
information to answer the student's question — using ONLY what is
explicitly written in the context, with no outside knowledge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"accurate"  → Context DIRECTLY and COMPLETELY answers the question.
              All key concepts, definitions, steps, or facts are present.
              For multi-part questions: ALL parts must be covered.

"ambiguous" → Context is PARTIALLY relevant. At least one of:
              - A required sub-part of the question is missing
              - Critical depth, definitions, or examples are absent
              For multi-part questions: SOME but not all parts are covered.

"not_found" → Context is irrelevant or off-topic. May share keywords
              but does NOT address the actual question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Judge ONLY what is EXPLICITLY stated in the context. No inference.
- Keyword overlap alone does NOT make context relevant.
- When in doubt between two scores, choose the LOWER (stricter) one.
- Your reasoning MUST reference specific content from the context
  or name exactly what is missing. Never speak in generalities.
- Treat yourself as someone who has ONLY ever read this context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question: "What is gradient descent and how does it update weights?"
Context:  "Gradient descent minimizes a loss function by iteratively
           moving in the direction of steepest descent. Weights are
           updated as: w = w - α∇L(w), where α is the learning rate."
→ binary_score: "accurate"
  reasoning: "Context defines gradient descent and provides the exact weight update formula."

---
Question: "Compare L1 and L2 regularization."
Context:  "L2 regularization adds a penalty proportional to the square of weights."
→ binary_score: "ambiguous"
  reasoning: "Context covers L2 but says nothing about L1 regularization."

---
Question: "Explain the attention mechanism in transformers."
Context:  "Transformers were introduced in 2017 and used in many NLP tasks."
→ binary_score: "not_found"
  reasoning: "Context only states the year and use cases; the attention mechanism is never described."
"""


RETRIEVAL_GRADER_HUMAN_PROMPT = """
STUDENT QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

NOTE: Grade THIS context chunk in isolation — not in combination with any other chunks.

Grade the context against the question using the scoring rules above. Be strict.
"""

RETRIEVAL_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RETRIEVAL_GRADER_SYSTEM_PROMPT),
    ("human",  RETRIEVAL_GRADER_HUMAN_PROMPT),
])




















