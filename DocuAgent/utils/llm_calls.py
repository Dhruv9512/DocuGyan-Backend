import base64
import logging
from typing import List

# Import the neutral Plumbing
from core.utils.llm_engine import LLMEngine
from DocuAgent.schemas.llm_schemas import PlannerOutput, RetrievalGraderOutput, RefinedBatch, DiagramOutput
from DocuAgent.prompts.DocuExtractor_Prompts import REFINE_QUESTIONS_PROMPT
from DocuAgent.prompts.academic_prompts import build_drafter_prompt,DIAGRAM_GENERATOR_PROMPT

logger = logging.getLogger(__name__)

class DocuAgentLLMCalls:
    """
    DocuAgent Specialist: Handles specialized workflows for Document Intelligence.
    Uses LLMEngine to fetch clients and implements resilient fallback logic.

    All LLM chains are lazy-initialized once at the class level and reused
    across all calls — never reconstructed per invocation.
    """

    _planner_chain = None   
    _vision_chain  = None   
    _grader_chain  = None
    _drafter_chain = None
    _refiner_chain = None
    _diagram_chain = None

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 1: Scan-Based PDF Extraction (Vision)
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_vision_chain(cls):
        """
        Lazy-initializes the vision fallback chain once.

        Model strategy (Groq-first, cross-provider safety):
        1. PRIMARY: Llama-4-Scout-17B-16E (Groq) - Best for dense academic PDFs.
        2. FB 1: Llama-4-Maverick-17B-128E (Groq) - Best for complex layouts and tables.
        3. FB 2: Llama-3.2-90B-Vision-Instruct (HuggingFace) - Cross-provider safety net.
        4. FB 3: Llama-3.2-11B-Vision-Instruct (HuggingFace) - Lightweight safety net.
        """
        if cls._vision_chain is not None:
            return cls._vision_chain

        primary    = LLMEngine.get_groq_client(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0,
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.0,
        )
        fallback_2 = LLMEngine.get_huggingface_chat_client(
            model_name="meta-llama/Llama-3.2-90B-Vision-Instruct",
            temperature=0.0,
        )
        fallback_3 = LLMEngine.get_huggingface_chat_client(
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            temperature=0.0,
        )

        cls._vision_chain = primary.with_fallbacks(
            [fallback_1, fallback_2, fallback_3]
        )
        logger.info("[DocuAgentLLMCalls] Vision chain initialized.")
        return cls._vision_chain

    @classmethod
    def VisionExtractorLLM(cls, image_bytes_list: List[bytes]) -> str:
        """
        Takes a list of JPEG image bytes and returns formatted Markdown.
        """
        from DocuAgent.prompts.DocuExtractor_Prompts import PDF_EXTRACTION_PROMPT

        lc_payload = cls._build_langchain_payload(
            PDF_EXTRACTION_PROMPT, image_bytes_list
        )

        try:
            result = cls._get_vision_chain().invoke(lc_payload)
            logger.info(
                "[VisionExtractor] Success | Pages: %d", len(image_bytes_list)
            )
            return result.content

        except Exception as e:
            logger.error("FATAL: All Vision LLMs failed. Error: %s", e)
            raise RuntimeError(f"All Vision LLMs failed: {e}") from e

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 2: Question Planner
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_planner_chain(cls):
        """
        Lazy-initializes the planner fallback chain once.
        """
        if cls._planner_chain is not None:
            return cls._planner_chain

        primary    = LLMEngine.get_groq_client(model_name="deepseek-r1-distill-llama-70b", temperature=0.1)
        fallback_1 = LLMEngine.get_groq_client(model_name="llama-3.3-70b-versatile", temperature=0.1)
        fallback_2 = LLMEngine.get_groq_client(model_name="llama-3.1-70b-versatile", temperature=0.1)
        fallback_3 = LLMEngine.get_groq_client(model_name="gemma2-9b-it", temperature=0.1)
        fallback_hf = LLMEngine.get_huggingface_chat_client(model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.1)

        cls._planner_chain = (
            primary.with_structured_output(PlannerOutput)
            .with_fallbacks([
                fallback_1.with_structured_output(PlannerOutput),
                fallback_2.with_structured_output(PlannerOutput),
                fallback_3.with_structured_output(PlannerOutput),
                fallback_hf.with_structured_output(PlannerOutput, method="json_mode"),  # ← fix
            ])
        )

        logger.info("[DocuAgentLLMCalls] Planner chain initialized.")
        return cls._planner_chain

    @classmethod
    def call_question_planner(cls, question: str) -> PlannerOutput:
        """
        Analyzes a question and produces a full execution plan.
        """
        system_prompt = (
            "You are an elite academic exam planner. "
            "Analyze the question below and produce a precise execution plan "
            "covering retrieval strategy, formatting constraints, routing category, "
            "and a quality assurance checklist. "
            "Think step-by-step before responding. "
            "Return ONLY valid JSON matching the requested schema. No preamble."
        )

        messages = [
            ("system", system_prompt),
            ("user", f"Question: {question}"),
        ]

        try:
            result: PlannerOutput = cls._get_planner_chain().invoke(messages)
            logger.debug(
                "[Planner] category=%s | words=%d",
                result.question_category,
                result.target_word_count,
            )
            return result

        except Exception as e:
            logger.error(
                "CRITICAL: All planner fallbacks failed. question=%r | error=%s",
                question[:80], e,
            )
            raise  RuntimeError(f"All Question Planner LLMs failed: {e}") from e
    


    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 3: C-RAG Retrieval Grader
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_grader_chain(cls):
        """
        Lazy-initializes the C-RAG grader fallback chain once.

        Model strategy (temperature=0.0 — fully deterministic judgements):
        1. PRIMARY:    DeepSeek-R1-Distill-Llama-70B (Groq)
                    — Chain-of-thought reasoning before verdict.
                        Best for nuanced "ambiguous" vs "accurate" decisions.

        2. FALLBACK 1: Llama-3.3-70B-Versatile (Groq)
                    — Strong instruction following, reliable binary output.
                        Different weights from DeepSeek — avoids systemic failures.

        3. FALLBACK 2: Gemma-2-9B-IT (Groq)
                    — Highly obedient for strict JSON schema at 9B.
                        Fast and cheap emergency fallback within Groq.

        4. FALLBACK 3: Qwen2.5-72B-Instruct (HuggingFace)
                    — Cross-provider safety net if Groq is fully down.
                        Strong reasoning, reliable structured output.
        """
        if cls._grader_chain is not None:
            return cls._grader_chain

        primary    = LLMEngine.get_groq_client(
            model_name="deepseek-r1-distill-llama-70b", temperature=0.0
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="llama-3.3-70b-versatile", temperature=0.0
        )
        fallback_2 = LLMEngine.get_groq_client(
            model_name="gemma2-9b-it", temperature=0.0
        )
        fallback_hf = LLMEngine.get_huggingface_chat_client(
            model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.0
        )

        cls._grader_chain = (
            primary.with_structured_output(RetrievalGraderOutput)
            .with_fallbacks([
                fallback_1.with_structured_output(RetrievalGraderOutput),
                fallback_2.with_structured_output(RetrievalGraderOutput),
                fallback_hf.with_structured_output(RetrievalGraderOutput, method="json_mode"),
            ])
        )

        logger.info("[DocuAgentLLMCalls] Grader chain initialized.")
        return cls._grader_chain


    @classmethod
    def call_retrieval_grader(
        cls,
        question: str,
        context: str,
    ) -> RetrievalGraderOutput:
        """
        Evaluates whether retrieved documents are sufficient to answer
        the question. Returns a structured verdict with reasoning.

        Verdict logic the LLM is instructed to follow:
        - accurate   → context directly and completely answers the question
        - ambiguous  → context is partially relevant but has gaps
        - not_found  → context is irrelevant or completely off-topic

        Args:
            question: The original exam/academic question.
            context:  Formatted retrieved chunks (numbered, with sources).

        Returns:
            RetrievalGraderOutput with binary_score + reasoning.

        Raises:
            Exception if all fallbacks fail — caller should handle gracefully.
        """
        system_prompt = (
            "You are a strict Retrieval Grader for an academic RAG system.\n\n"
            "Your ONLY job is to evaluate whether the provided context contains "
            "enough relevant information to answer the question accurately.\n\n"
            "Scoring rules:\n"
            "- 'accurate'  → The context directly addresses the question. "
            "Key concepts, definitions, or facts needed are clearly present.\n"
            "- 'ambiguous' → The context is partially relevant. Some useful "
            "information exists but critical details, examples, or depth are missing.\n"
            "- 'not_found' → The context is irrelevant, off-topic, or contains "
            "no useful information for this question whatsoever.\n\n"
            "Be strict. Do NOT mark as 'accurate' if the answer requires "
            "significant inference or missing facts. "
            "Return ONLY valid JSON matching the schema. No preamble."
        )

        messages = [
            ("system", system_prompt),
            (
                "user",
                f"Question:\n{question}\n\n"
                f"Retrieved Context:\n{context}\n\n"
                f"Grade the context strictly using the scoring rules above."
            ),
        ]

        try:
            result: RetrievalGraderOutput = cls._get_grader_chain().invoke(messages)
            logger.debug(
                "[Grader] score=%s | reasoning=%s",
                result.binary_score,
                result.reasoning[:80],
            )
            return result

        except Exception as e:
            logger.error(
                "CRITICAL: All grader fallbacks failed. question=%r | error=%s",
                question[:80], e,
            )
            raise RuntimeError(f"All Retrieval Grader LLMs failed: {e}") from e
    

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 4: Academic Answer Drafter
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_drafter_chain(cls):
        """
        Lazy-initializes the drafter fallback chain once.

        Model strategy (temperature=0.4 — creative but grounded):
        1. PRIMARY:    Llama-3.3-70B-Versatile (Groq)
                    — Best instruction-following + long-form generation on Groq.
                        Handles all 6 categories cleanly. Fast enough for parallel workers.

        2. FALLBACK 1: DeepSeek-R1-Distill-Llama-70B (Groq)
                    — Chain-of-thought improves answer structure quality.
                        Slightly slower but stronger for math and analytical questions.

        3. FALLBACK 2: Llama-3.1-70B-Versatile (Groq)
                    — Stable same-family safety net if 3.3 hits rate limits.

        4. FALLBACK 3: Qwen2.5-72B-Instruct (HuggingFace)
                    — Cross-provider safety net if Groq is fully down.
                        Excellent long-form generation and instruction adherence.

        5. FALLBACK 4: Llama-3.1-8B-Instant (Groq)
                    — Ultra-fast last resort. Lower quality but never fails.
                        Ensures the pipeline always produces SOMETHING.
        """
        if cls._drafter_chain is not None:
            return cls._drafter_chain

        primary    = LLMEngine.get_groq_client(
            model_name="llama-3.3-70b-versatile", temperature=0.4
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="deepseek-r1-distill-llama-70b", temperature=0.4
        )
        fallback_2 = LLMEngine.get_groq_client(
            model_name="llama-3.1-70b-versatile", temperature=0.4
        )
        fallback_3 = LLMEngine.get_huggingface_chat_client(
            model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.4
        )
        fallback_4 = LLMEngine.get_groq_client(
            model_name="llama-3.1-8b-instant", temperature=0.4
        )

        cls._drafter_chain = primary.with_fallbacks([
            fallback_1,
            fallback_2,
            fallback_3,
            fallback_4,
        ])

        logger.info("[DocuAgentLLMCalls] Drafter chain initialized.")
        return cls._drafter_chain


    @classmethod
    def call_answer_drafter(
        cls,
        question:       str,
        context:        str,
        plan:           PlannerOutput
    ) -> str:
        """
        Generates the final drafted answer for a single question.

        Args:
            question:      The original student question.
            context:       Formatted retrieved context string (numbered chunks).
            plan:          PlannerOutput — drives the entire prompt construction.

        Returns:
            Raw drafted answer string (Markdown).

        Raises:
            RuntimeError if all fallbacks fail.
        """
        system_prompt = build_drafter_prompt(plan)

        user_prompt = (
            f"## Question\n{question}\n\n"
            f"## Retrieved Context\n{context}"
        )

        messages = [
            ("system", system_prompt),
            ("user",   user_prompt),
        ]

        try:
            result = cls._get_drafter_chain().invoke(messages)
            draft  = result.content if hasattr(result, "content") else str(result)

            logger.debug(
                "[Drafter] category=%s | words≈%d",
                plan.question_category,
                len(draft.split()),
            )
            return draft

        except Exception as e:
            logger.error(
                "CRITICAL: All drafter fallbacks failed. question=%r | error=%s",
                question[:80], e,
            )
            raise RuntimeError(f"All Drafter LLMs failed: {e}") from e
        
    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 5: Question Refiner
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_refiner_chain(cls):
        """
        Lazy-initializes the question refiner fallback chain once.

        Task profile: structured extraction + grammar correction on raw text.
        Needs strong instruction-following and reliable JSON output.
        Temperature = 0.0 — purely deterministic cleanup, no creativity needed.

        Model strategy:
        1. PRIMARY:    Llama-3.3-70B-Versatile (Groq)
                    — Best instruction-following on Groq for structured extraction.
                    Handles multi-part MCQ stripping and fill-in-the-blank
                    conversion cleanly. Fast enough for batched parallel workers.

        2. FALLBACK 1: DeepSeek-R1-Distill-Llama-70B (Groq)
                    — Chain-of-thought pre-reasoning improves handling of
                    ambiguous or malformed question blocks before output.
                    Different weight family from primary — avoids correlated failures.

        3. FALLBACK 2: Gemma-2-9B-IT (Groq)
                    — Highly schema-obedient at 9B. Fast emergency fallback
                    within Groq if the 70B models hit rate limits.

        4. FALLBACK 3: Qwen2.5-72B-Instruct (HuggingFace)
                    — Cross-provider safety net if Groq is fully down.
                    Strong multilingual instruction-following; handles
                    mixed-language exam papers well.

        5. FALLBACK 4: Llama-3.1-70B-Versatile (Groq)
                    — Same-family safety net for Llama-3.3. Slightly lower
                    quality but virtually never fails while Groq is up.
        """
        if cls._refiner_chain is not None:
            return cls._refiner_chain

        primary = LLMEngine.get_groq_client(model_name="llama-3.3-70b-versatile", temperature=0.0)
        fallback_1 = LLMEngine.get_groq_client(model_name="deepseek-r1-distill-llama-70b", temperature=0.0)
        fallback_2 = LLMEngine.get_groq_client(model_name="gemma2-9b-it", temperature=0.0)
        fallback_3 = LLMEngine.get_huggingface_chat_client(model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.0)
        fallback_4 = LLMEngine.get_groq_client(model_name="llama-3.1-70b-versatile", temperature=0.0)

        def make_refiner_chain(llm, use_json_mode=False):
            structured = (
                llm.with_structured_output(RefinedBatch, method="json_mode")
                if use_json_mode
                else llm.with_structured_output(RefinedBatch)
            )
            return REFINE_QUESTIONS_PROMPT | structured

        cls._refiner_chain = make_refiner_chain(primary).with_fallbacks([
            make_refiner_chain(fallback_1),
            make_refiner_chain(fallback_2),
            make_refiner_chain(fallback_3, use_json_mode=True), 
            make_refiner_chain(fallback_4),
        ])

        logger.info("[DocuAgentLLMCalls] Refiner chain initialized.")
        return cls._refiner_chain



    @classmethod
    def call_refine_questions(cls, batch_text: str) -> RefinedBatch:
        """
        Cleans and refines a batch of raw question blocks into structured output.

        The prompt handles all question types (MCQ stripping, fill-in-the-blank
        conversion, multi-part splitting, True/False, Assertion-Reason, etc.)
        as defined in REFINE_QUESTIONS_PROMPT.

        Args:
            batch_text: Raw question blocks joined by double newlines,
            each prefixed with '- ' (as formatted by _process_batch).

        Returns:
            RefinedBatch containing a list of RefinedQuestion objects.

        Raises:
            RuntimeError if all fallbacks are exhausted.
        """
        try:
            
            result: RefinedBatch = cls._get_refiner_chain().invoke(
                {"questions": batch_text}
            )

            if not result or not result.questions:
                logger.warning("[Refiner] LLM returned an empty RefinedBatch.")
                raise ValueError("LLM returned an empty RefinedBatch.")

            logger.debug(
                "[Refiner] Successfully refined %d questions.", len(result.questions)
            )
            return result

        except Exception as e:
            logger.error(
                "CRITICAL: All refiner fallbacks failed. batch_preview=%r | error=%s",
                batch_text[:80], e,
            )
            raise RuntimeError(f"All Refiner LLMs failed: {e}") from e
    
    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 6: Academic Diagram Generator
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_diagram_chain(cls):
        """
        Lazy-initializes the diagram generator fallback chain once.

        Task profile: structured diagram code generation from a concept + hint.
        Needs strong code generation and reliable JSON schema output.
        Temperature = 0.1 — nearly deterministic, tiny variance for layout variety.

        Model strategy:
        1. PRIMARY:    Llama-3.3-70B-Versatile (Groq)
                    — Best instruction-following + code gen on Groq.
                    Handles Mermaid syntax reliably. Fast enough for
                    parallel placeholder resolution.

        2. FALLBACK 1: DeepSeek-R1-Distill-Llama-70B (Groq)
                    — Chain-of-thought pre-reasoning produces cleaner
                    diagram structure for complex multi-entity concepts.
                    Different weight family — avoids correlated failures.

        3. FALLBACK 2: Gemma-2-9B-IT (Groq)
                    — Highly schema-obedient at 9B. Fast emergency fallback
                    within Groq if 70B slots are rate-limited.

        4. FALLBACK 3: Qwen2.5-72B-Instruct (HuggingFace)
                    — Cross-provider safety net if Groq is fully down.
                    Strong structured code generation, reliable JSON output.

        5. FALLBACK 4: Llama-3.1-70B-Versatile (Groq)
                    — Same-family safety net. Lower quality ceiling than 3.3
                    but virtually always available.
        """
        if cls._diagram_chain is not None:
            return cls._diagram_chain

        primary = LLMEngine.get_groq_client(model_name="llama-3.3-70b-versatile", temperature=0.1)
        fallback_1 = LLMEngine.get_groq_client(model_name="deepseek-r1-distill-llama-70b", temperature=0.1)
        fallback_2 = LLMEngine.get_groq_client(model_name="gemma2-9b-it", temperature=0.1)
        fallback_3 = LLMEngine.get_huggingface_chat_client(model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.1)
        fallback_4 = LLMEngine.get_groq_client(model_name="llama-3.1-70b-versatile", temperature=0.1)

        def make_diagram_chain(llm, use_json_mode=False):
            structured = (
                llm.with_structured_output(DiagramOutput, method="json_mode")
                if use_json_mode
                else llm.with_structured_output(DiagramOutput)
            )
            return DIAGRAM_GENERATOR_PROMPT | structured 

        cls._diagram_chain = make_diagram_chain(primary).with_fallbacks([
            make_diagram_chain(fallback_1),
            make_diagram_chain(fallback_2),
            make_diagram_chain(fallback_3, use_json_mode=True),  
            make_diagram_chain(fallback_4),
        ])

        logger.info("[DocuAgentLLMCalls] Diagram chain initialized.")
        return cls._diagram_chain


    @classmethod
    def call_diagram_generator(
        cls,
        concept: str,
        hint: str,
        question_category: str,
    ) -> DiagramOutput:
        """
        Generates a Mermaid diagram for a given academic concept.

        Args:
            concept:The core entity/concept to diagram
                    (e.g. "TCP three-way handshake").
            hint:The drafter's one-sentence description of what
                this specific diagram should illustrate
                (extracted from the sentence just before the placeholder).
            question_category: From PlannerOutput — drives diagram style choice
                            (flowchart for process questions, classDiagram for
                            OOP, sequenceDiagram for protocol questions, etc.)

        Returns:
            DiagramOutput with diagram_type, diagram_code, caption, fallback_text.

        Raises:
            RuntimeError if all fallbacks are exhausted.
        """
        
        try:
            
            result: DiagramOutput = cls._get_diagram_chain().invoke(
                {
                    "concept": concept,
                    "hint": hint,
                    "question_category": question_category,
                }
            )
            logger.debug(
                "[DiagramGen] concept=%r | type=%s | nodes≈%d",
                concept[:60],
                result.diagram_type,
                result.diagram_code.count("\n"),
            )
            return result

        except Exception as e:
            logger.error(
                "CRITICAL: All diagram generator fallbacks failed. concept=%r | error=%s",
                concept[:60], e,
            )
            raise RuntimeError(f"All Diagram Generator LLMs failed: {e}") from e

    # ══════════════════════════════════════════════════════════════════════
    # SHARED UTILITY
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _build_langchain_payload(
        cls, prompt_text: str, image_bytes_list: List[bytes]
    ) -> list:
        """
        Builds a LangChain HumanMessage with text + multiple Base64 images.
        """
        from langchain_core.messages import HumanMessage

        content_blocks = [{"type": "text", "text": prompt_text}]

        for img_bytes in image_bytes_list:
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url":    f"data:image/jpeg;base64,{b64_image}",
                    "detail": "high",
                },
            })

        return [HumanMessage(content=content_blocks)]