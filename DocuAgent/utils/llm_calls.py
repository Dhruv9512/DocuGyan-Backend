import base64
import logging
from typing import List

# Import the neutral Plumbing
from core.utils.llm_engine import LLMEngine
from DocuAgent.schemas.llm_schemas import PlannerOutput, RetrievalGraderOutput, RefinedBatch
from DocuAgent.prompts.DocuExtractor_Prompts import REFINE_QUESTIONS_PROMPT
from DocuAgent.prompts.academic_prompts import build_drafter_user_prompt, RETRIEVAL_GRADER_PROMPT, QUESTION_PLANNER_PROMPT, DIAGRAM_INJECTOR_PROMPT

logger = logging.getLogger(__name__)


class DocuAgentLLMCalls:
    """
    DocuAgent Specialist: Handles specialized workflows for Document Intelligence.
    Uses LLMEngine to fetch clients and implements resilient fallback logic.
    """

    # Primary chain
    _planner_chain = None
    _vision_chain  = None
    _grader_chain  = None
    _drafter_chain = None
    _refiner_chain = None
    _diagram_chain = None

    # Fallback chain
    _planner_chain_backup = None
    _vision_chain_backup  = None
    _grader_chain_backup  = None
    _drafter_chain_backup = None
    _refiner_chain_backup = None
    _diagram_chain_backup = None

    # ══════════════════════════════════════════════════════════════════════
    # SHARED UTILITY — HuggingFace structured output fix
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _make_hf_structured(llm, schema_class):
        """
        Wraps a HuggingFace LLM's json_mode output into the target Pydantic class.

        WHY THIS IS NEEDED:
            HuggingFace's with_structured_output(method="json_mode") returns a raw
            dict, NOT an instance of the schema class. This causes AttributeErrors
            like 'dict' object has no attribute 'binary_score'.

            This wrapper pipes the raw dict through the Pydantic constructor so
            callers always receive a properly typed object regardless of provider.

        Usage:
            fallback_hf = DocuAgentLLMCalls._make_hf_structured(
                LLMEngine.get_huggingface_chat_client(...), RetrievalGraderOutput
            )
        """
        from langchain_core.runnables import RunnableLambda

        raw_chain = llm.with_structured_output(schema_class, method="json_mode")

        def _parse(result):
            # If HF returned a dict, parse it into the Pydantic model
            if isinstance(result, dict):
                return schema_class(**result)
            # Already the right type (e.g. Groq returning Pydantic directly)
            return result

        return raw_chain | RunnableLambda(_parse)

    # ══════════════════════════════════════════════════════════════════════
    # SHARED UTILITY — Refiner chain builder
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _make_refiner_chain(llm, use_json_mode: bool = False):
        """
        Builds a single refiner chain by composing REFINE_QUESTIONS_PROMPT with
        a structured-output LLM.

        Extracted from _get_refiner_chain to avoid re-defining the helper
        function on every call and to make it independently testable.

        Args:
            llm:Any LangChain-compatible chat model.
            use_json_mode: If True, uses method="json_mode" (needed for some HF models).

        Returns:
            A Runnable that accepts {"questions": str} and returns RefinedBatch.
        """
        structured = (
            llm.with_structured_output(RefinedBatch, method="json_mode")
            if use_json_mode
            else llm.with_structured_output(RefinedBatch)
        )
        return REFINE_QUESTIONS_PROMPT | structured

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 1: Scan-Based PDF Extraction (Vision)
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_vision_chain(cls, use_backup: bool = False):
        """
        Lazy-initializes the vision fallback chain once per pool.
        ALL models must be vision-capable (multimodal) — text-only models cannot
        process image payloads and must never appear in this chain.

        Model strategy:
        1. PRIMARY:    llama-4-scout-17b-16e-instruct          (Groq)
        2. FALLBACK 1: llama-4-maverick-17b-128e-instruct      (Groq)
        3. FALLBACK 2: Llama-3.2-90B-Vision-Instruct           (HuggingFace)
        4. FALLBACK 3: Llama-3.2-11B-Vision-Instruct           (HuggingFace)
        5. FALLBACK 4: Llama-3.2-3B-Vision-Instruct            (HuggingFace)  ← smallest, last resort
        6. FALLBACK 5: Llama-3.2-1B-Vision-Instruct            (HuggingFace)  ← absolute last resort
        """
        if use_backup and cls._vision_chain_backup is not None:
            return cls._vision_chain_backup
        if not use_backup and cls._vision_chain is not None:
            return cls._vision_chain

        primary    = LLMEngine.get_groq_client(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0, use_backup=use_backup
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.0, use_backup=use_backup
        )
        fallback_2 = LLMEngine.get_huggingface_chat_client(
            model_name="meta-llama/Llama-3.2-90B-Vision-Instruct", temperature=0.0, use_backup=use_backup
        )
        fallback_3 = LLMEngine.get_huggingface_chat_client(
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", temperature=0.0, use_backup=use_backup
        )
        fallback_4 = LLMEngine.get_huggingface_chat_client(
            model_name="meta-llama/Llama-3.2-3B-Vision-Instruct", temperature=0.0, use_backup=use_backup
        )
        fallback_5 = LLMEngine.get_huggingface_chat_client(
            model_name="meta-llama/Llama-3.2-1B-Vision-Instruct", temperature=0.0, use_backup=use_backup
        )

        chain = primary.with_fallbacks([fallback_1, fallback_2, fallback_3, fallback_4, fallback_5])

        if use_backup:
            cls._vision_chain_backup = chain
        else:
            cls._vision_chain = chain

        return chain

    @classmethod
    def VisionExtractorLLM(cls, image_bytes_list: List[bytes]) -> str:
        from DocuAgent.prompts.DocuExtractor_Prompts import PDF_EXTRACTION_PROMPT

        lc_payload = cls._build_langchain_payload(PDF_EXTRACTION_PROMPT, image_bytes_list)
        try:
            result = cls._get_vision_chain().invoke(lc_payload)
            logger.info("[VisionExtractor] Success | Pages: %d", len(image_bytes_list))
            return result.content
        except Exception as e:
            try:
                logger.info("Attempting VisionExtractor backup chain...")
                backup_chain = cls._get_vision_chain(use_backup=True)
                result = backup_chain.invoke(lc_payload)
                logger.info("[VisionExtractor Backup] Success | Pages: %d", len(image_bytes_list))
                return result.content
            except Exception as backup_e:
                logger.error("FATAL: VisionExtractor backup chain also failed. Error: %s", backup_e)
                raise RuntimeError(f"All Vision LLMs failed: {e}") from e

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 2: Question Planner
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_planner_chain(cls, use_backup: bool = False):
        """
        Lazy-initializes the planner fallback chain once per pool.

        Model strategy (temperature=0.1):
        1. PRIMARY:    llama-3.3-70b-versatile                 (Groq)
        2. FALLBACK 1: llama3-groq-70b-8192-tool-use-preview   (Groq)
        3. FALLBACK 2: qwen-qwq-32b                            (Groq)
        4. FALLBACK 3: gemma2-9b-it                            (Groq)
        5. FALLBACK 4: llama-3.1-70b-versatile                 (Groq)
        6. FALLBACK 5: Qwen/Qwen2.5-72B-Instruct               (HuggingFace)
        """
        if use_backup and cls._planner_chain_backup is not None:
            return cls._planner_chain_backup
        if not use_backup and cls._planner_chain is not None:
            return cls._planner_chain

        primary    = LLMEngine.get_groq_client(
            model_name="llama-3.3-70b-versatile", temperature=0.1, use_backup=use_backup
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="llama3-groq-70b-8192-tool-use-preview", temperature=0.1, use_backup=use_backup
        )
        fallback_2 = LLMEngine.get_groq_client(
            model_name="qwen-qwq-32b", temperature=0.1, use_backup=use_backup
        )
        fallback_3 = LLMEngine.get_groq_client(
            model_name="gemma2-9b-it", temperature=0.1, use_backup=use_backup
        )
        fallback_4 = LLMEngine.get_groq_client(
            model_name="llama-3.1-70b-versatile", temperature=0.1, use_backup=use_backup
        )
        hf_llm = LLMEngine.get_huggingface_chat_client(
            model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.1, use_backup=use_backup
        )

        chain = (
            QUESTION_PLANNER_PROMPT | primary.with_structured_output(PlannerOutput)
            .with_fallbacks([
                QUESTION_PLANNER_PROMPT | fallback_1.with_structured_output(PlannerOutput),
                QUESTION_PLANNER_PROMPT | fallback_2.with_structured_output(PlannerOutput),
                QUESTION_PLANNER_PROMPT | fallback_3.with_structured_output(PlannerOutput),
                QUESTION_PLANNER_PROMPT | fallback_4.with_structured_output(PlannerOutput),
                QUESTION_PLANNER_PROMPT | cls._make_hf_structured(hf_llm, PlannerOutput),
            ])
        )

        if use_backup:
            cls._planner_chain_backup = chain
        else:
            cls._planner_chain = chain

        return chain

    @classmethod
    def call_question_planner(cls, question: str, context: str = "") -> PlannerOutput:

        try:
            result: PlannerOutput = cls._get_planner_chain().invoke({
                "question": question,
                "context": context[:6000],
            })
            if isinstance(result, dict):
                result = PlannerOutput(**result)
            logger.info("[Planner] Success | Category: %s | Marks: %s", result.question_category, result.allocated_marks)
            return result

        except Exception as e:
            try:
                logger.info("Attempting Planner backup chain...")
                backup_chain = cls._get_planner_chain(use_backup=True)
                result: PlannerOutput = backup_chain.invoke({
                    "question": question,
                    "context": context[:6000],
                })
                if isinstance(result, dict):
                    result = PlannerOutput(**result)
                logger.info("[Planner Backup] Success | Category: %s | Marks: %s", result.question_category, result.allocated_marks)
                return result
            except Exception as backup_e:
                logger.error(
                    "CRITICAL: Planner backup chain also failed. question=%r | error=%s",
                    question[:80], backup_e,
                )
                raise RuntimeError(f"All Question Planner LLMs failed: {e}") from e

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 3: C-RAG Retrieval Grader
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_grader_chain(cls, use_backup: bool = False):
        """
        Lazy-initializes the C-RAG grader fallback chain once per pool.

        Model strategy (temperature=0.0 — fully deterministic):
        1. PRIMARY:    llama-3.3-70b-versatile                 (Groq)
        2. FALLBACK 1: llama3-groq-70b-8192-tool-use-preview   (Groq)
        3. FALLBACK 2: qwen-qwq-32b                            (Groq)
        4. FALLBACK 3: gemma2-9b-it                            (Groq)
        5. FALLBACK 4: llama-3.1-70b-versatile                 (Groq)
        6. FALLBACK 5: Qwen/Qwen2.5-72B-Instruct               (HuggingFace)
        """
        if use_backup and cls._grader_chain_backup is not None:
            return cls._grader_chain_backup
        if not use_backup and cls._grader_chain is not None:
            return cls._grader_chain

        primary    = LLMEngine.get_groq_client(
            model_name="llama-3.3-70b-versatile", temperature=0.0, use_backup=use_backup
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="llama3-groq-70b-8192-tool-use-preview", temperature=0.0, use_backup=use_backup
        )
        fallback_2 = LLMEngine.get_groq_client(
            model_name="qwen-qwq-32b", temperature=0.0, use_backup=use_backup
        )
        fallback_3 = LLMEngine.get_groq_client(
            model_name="gemma2-9b-it", temperature=0.0, use_backup=use_backup
        )
        fallback_4 = LLMEngine.get_groq_client(
            model_name="llama-3.1-70b-versatile", temperature=0.0, use_backup=use_backup
        )
        hf_llm = LLMEngine.get_huggingface_chat_client(
            model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.0, use_backup=use_backup
        )

        chain = (
            RETRIEVAL_GRADER_PROMPT | primary.with_structured_output(RetrievalGraderOutput)
            .with_fallbacks([
                RETRIEVAL_GRADER_PROMPT | fallback_1.with_structured_output(RetrievalGraderOutput),
                RETRIEVAL_GRADER_PROMPT | fallback_2.with_structured_output(RetrievalGraderOutput),
                RETRIEVAL_GRADER_PROMPT | fallback_3.with_structured_output(RetrievalGraderOutput),
                RETRIEVAL_GRADER_PROMPT | fallback_4.with_structured_output(RetrievalGraderOutput),
                RETRIEVAL_GRADER_PROMPT | cls._make_hf_structured(hf_llm, RetrievalGraderOutput),
            ])
        )

        if use_backup:
            cls._grader_chain_backup = chain
        else:
            cls._grader_chain = chain

        return chain

    @classmethod
    def call_retrieval_grader(cls, question: str, context: str) -> RetrievalGraderOutput:
        try:
            result: RetrievalGraderOutput = cls._get_grader_chain().invoke({
                "question": question,
                "context": context[:6000],
            })

            if isinstance(result, dict):
                result = RetrievalGraderOutput(**result)
            logger.info("[Grader] Success | Score: %s", result.binary_score)
            return result

        except Exception as e:
            try:
                logger.info("Attempting Grader backup chain...")
                backup_chain = cls._get_grader_chain(use_backup=True)
                result: RetrievalGraderOutput = backup_chain.invoke({
                    "question": question,
                    "context": context[:6000],
                })
                if isinstance(result, dict):
                    result = RetrievalGraderOutput(**result)
                logger.info("[Grader Backup] Success | Score: %s", result.binary_score)
                return result
            except Exception as backup_e:
                logger.error(
                    "CRITICAL: Grader backup chain also failed. question=%r | error=%s",
                    question[:80], backup_e,
                )
                raise RuntimeError(f"All Retrieval Grader LLMs failed: {e}") from e

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 4: Academic Answer Drafter
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_drafter_chain(cls, use_backup: bool = False):
        """
        Lazy-initializes the drafter fallback chain once per pool.

        Model strategy (temperature=0.4 — creative but grounded):
        1. PRIMARY:    llama-3.3-70b-versatile    (Groq)
        2. FALLBACK 1: qwen-qwq-32b               (Groq)
        3. FALLBACK 2: llama-3.1-70b-versatile    (Groq)
        4. FALLBACK 3: gemma2-9b-it               (Groq)
        5. FALLBACK 4: Qwen/Qwen2.5-72B-Instruct  (HuggingFace)
        6. FALLBACK 5: llama-3.1-8b-instant       (Groq)  ← last resort
        """
        if use_backup and cls._drafter_chain_backup is not None:
            return cls._drafter_chain_backup
        if not use_backup and cls._drafter_chain is not None:
            return cls._drafter_chain

        primary    = LLMEngine.get_groq_client(
            model_name="llama-3.3-70b-versatile", temperature=0.4, use_backup=use_backup
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="qwen-qwq-32b", temperature=0.4, use_backup=use_backup
        )
        fallback_2 = LLMEngine.get_groq_client(
            model_name="llama-3.1-70b-versatile", temperature=0.4, use_backup=use_backup
        )
        fallback_3 = LLMEngine.get_groq_client(
            model_name="gemma2-9b-it", temperature=0.4, use_backup=use_backup
        )
        fallback_4 = LLMEngine.get_huggingface_chat_client(
            model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.4, use_backup=use_backup
        )
        fallback_5 = LLMEngine.get_groq_client(
            model_name="llama-3.1-8b-instant", temperature=0.4, use_backup=use_backup
        )

        chain = primary.with_fallbacks([
            fallback_1,
            fallback_2,
            fallback_3,
            fallback_4,
            fallback_5,
        ])

        if use_backup:
            cls._drafter_chain_backup = chain
        else:
            cls._drafter_chain = chain

        return chain

    @classmethod
    def call_answer_drafter(
        cls,
        question: str,
        context_chunks: list[dict],
        plan: PlannerOutput,
    ) -> str:

        user_prompt = build_drafter_user_prompt(
            question=question,
            context_chunks=context_chunks,
            plan=plan,
        )

        try:
            result = cls._get_drafter_chain().invoke(user_prompt)
            draft  = result.content if hasattr(result, "content") else str(result)
            logger.info(
                "[Drafter] Success | Category: %s | Words≈%d",
                plan.question_category, len(draft.split())
            )
            return draft

        except Exception as e:
            try:
                logger.info("Attempting Drafter backup chain...")
                backup_chain = cls._get_drafter_chain(use_backup=True)
                result = backup_chain.invoke(user_prompt)
                draft  = result.content if hasattr(result, "content") else str(result)
                logger.info(
                    "[Drafter Backup] Success | Category: %s | Words≈%d",
                    plan.question_category, len(draft.split())
                )
                return draft
            except Exception as backup_e:
                logger.error(
                    "CRITICAL: Both drafter chains failed. question=%r | error=%s",
                    question[:80], backup_e,
                )
                raise RuntimeError(f"All Drafter LLMs failed: {e}") from e

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 5: Question Refiner
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_refiner_chain(cls, use_backup: bool = False):
        """
        Lazy-initializes the question refiner fallback chain once per pool.

        Model strategy (temperature=0.0 — deterministic cleanup):
        1. PRIMARY:    llama-3.3-70b-versatile    (Groq)
        2. FALLBACK 1: qwen-qwq-32b               (Groq)
        3. FALLBACK 2: gemma2-9b-it               (Groq)
        4. FALLBACK 3: llama-3.1-70b-versatile    (Groq)
        5. FALLBACK 4: Qwen/Qwen2.5-72B-Instruct  (HuggingFace)
        6. FALLBACK 5: llama-3.1-8b-instant       (Groq)  ← last resort
        """
        if use_backup and cls._refiner_chain_backup is not None:
            return cls._refiner_chain_backup
        if not use_backup and cls._refiner_chain is not None:
            return cls._refiner_chain

        primary    = LLMEngine.get_groq_client(
            model_name="llama-3.3-70b-versatile", temperature=0.0, use_backup=use_backup
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="qwen-qwq-32b", temperature=0.0, use_backup=use_backup
        )
        fallback_2 = LLMEngine.get_groq_client(
            model_name="gemma2-9b-it", temperature=0.0, use_backup=use_backup
        )
        fallback_3 = LLMEngine.get_groq_client(
            model_name="llama-3.1-70b-versatile", temperature=0.0, use_backup=use_backup
        )
        hf_llm = LLMEngine.get_huggingface_chat_client(
            model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.0, use_backup=use_backup
        )
        fallback_5 = LLMEngine.get_groq_client(
            model_name="llama-3.1-8b-instant", temperature=0.0, use_backup=use_backup
        )

        chain = cls._make_refiner_chain(primary).with_fallbacks([
            cls._make_refiner_chain(fallback_1),
            cls._make_refiner_chain(fallback_2),
            cls._make_refiner_chain(fallback_3),
            REFINE_QUESTIONS_PROMPT | cls._make_hf_structured(hf_llm, RefinedBatch),
            cls._make_refiner_chain(fallback_5),
        ])

        if use_backup:
            cls._refiner_chain_backup = chain
        else:
            cls._refiner_chain = chain

        return chain

    @classmethod
    def call_refine_questions(cls, batch_text: str) -> RefinedBatch:
        try:
            result: RefinedBatch = cls._get_refiner_chain().invoke(
                {"questions": batch_text}
            )
            if isinstance(result, dict):
                result = RefinedBatch(**result)

            if not result or not result.questions:
                logger.warning("[Refiner] LLM returned an empty RefinedBatch.")
                raise ValueError("LLM returned an empty RefinedBatch.")

            logger.info("[Refiner] Successfully refined %d questions.", len(result.questions))
            return result

        except Exception as e:
            try:
                logger.info("Attempting Refiner backup chain...")
                backup_chain = cls._get_refiner_chain(use_backup=True)
                result: RefinedBatch = backup_chain.invoke({"questions": batch_text})
                if isinstance(result, dict):
                    result = RefinedBatch(**result)
                if not result or not result.questions:
                    logger.warning("[Refiner Backup] LLM returned an empty RefinedBatch.")
                    raise ValueError("LLM returned an empty RefinedBatch.")
                logger.info("[Refiner Backup] Successfully refined %d questions.", len(result.questions))
                return result
            except Exception as backup_e:
                logger.error(
                    "CRITICAL: Refiner backup chain also failed. batch_preview=%r | error=%s",
                    batch_text[:80], backup_e,
                )
                raise RuntimeError(f"All Refiner LLMs failed: {e}") from e

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 6: Diagram Query Generator
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_diagram_chain(cls, use_backup: bool = False):
        """
        Lazy-initializes the diagram injector fallback chain once per pool.

        Model strategy (temperature=0.2 — mostly deterministic, slight flexibility):
        1. PRIMARY:    llama-3.3-70b-versatile    (Groq)
        2. FALLBACK 1: qwen-qwq-32b               (Groq)
        3. FALLBACK 2: llama-3.1-70b-versatile    (Groq)
        4. FALLBACK 3: gemma2-9b-it               (Groq)
        5. FALLBACK 4: Qwen/Qwen2.5-72B-Instruct  (HuggingFace)
        6. FALLBACK 5: llama-3.1-8b-instant       (Groq)  ← last resort
        """
        if use_backup and cls._diagram_chain_backup is not None:
            return cls._diagram_chain_backup
        if not use_backup and cls._diagram_chain is not None:
            return cls._diagram_chain

        primary    = LLMEngine.get_groq_client(
            model_name="llama-3.3-70b-versatile", temperature=0.2, use_backup=use_backup
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="qwen-qwq-32b", temperature=0.2, use_backup=use_backup
        )
        fallback_2 = LLMEngine.get_groq_client(
            model_name="llama-3.1-70b-versatile", temperature=0.2, use_backup=use_backup
        )
        fallback_3 = LLMEngine.get_groq_client(
            model_name="gemma2-9b-it", temperature=0.2, use_backup=use_backup
        )
        fallback_4 = LLMEngine.get_huggingface_chat_client(
            model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.2, use_backup=use_backup
        )
        fallback_5 = LLMEngine.get_groq_client(
            model_name="llama-3.1-8b-instant", temperature=0.2, use_backup=use_backup
        )

        # Compose prompt into chain so callers invoke with {"question": ..., "draft": ...}
        chain = DIAGRAM_INJECTOR_PROMPT | primary.with_fallbacks([
            fallback_1,
            fallback_2,
            fallback_3,
            fallback_4,
            fallback_5,
        ])

        if use_backup:
            cls._diagram_chain_backup = chain
        else:
            cls._diagram_chain = chain

        return chain

    @classmethod
    def call_diagram_query_generator(cls, question: str, draft: str) -> str | None:
        """
        Reads the draft and injects [DiagramQuery: ...] + {diagram_N} placeholders
        at contextually appropriate positions.

        Retry strategy:
        - Primary chain  : primary + 5 fallbacks via LangChain .with_fallbacks()
        - Backup chain   : separate model pool via use_backup=True
        - Total failure  : returns None — caller keeps the original draft
        """
        inputs = {"question": question, "draft": draft}

        try:
            result = cls._get_diagram_chain().invoke(inputs)
            output = result.content if hasattr(result, "content") else str(result)

            if not output or not output.strip():
                raise ValueError("Diagram chain returned empty output.")

            if "{diagram_1}" not in output:
                raise ValueError("Diagram chain did not inject any placeholder.")

            logger.info(
                "[DiagramQueryGenerator] Success | Placeholders injected | Words≈%d",
                len(output.split()),
            )
            return output.strip()

        except Exception as e:
            try:
                logger.info("Attempting DiagramQueryGenerator backup chain...")
                backup_chain = cls._get_diagram_chain(use_backup=True)
                result = backup_chain.invoke(inputs)
                output = result.content if hasattr(result, "content") else str(result)

                if not output or not output.strip():
                    raise ValueError("Diagram backup chain returned empty output.")

                if "{diagram_1}" not in output:
                    raise ValueError("Diagram backup chain did not inject any placeholder.")

                logger.info(
                    "[DiagramQueryGenerator Backup] Success | Words≈%d",
                    len(output.split()),
                )
                return output.strip()

            except Exception as backup_e:
                logger.error(
                    "CRITICAL: Both diagram chains failed. question=%r | error=%s",
                    question[:80], backup_e,
                )
                return None

    # ══════════════════════════════════════════════════════════════════════
    # SHARED UTILITY — Vision payload builder
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _build_langchain_payload(cls, prompt_text: str, image_bytes_list: List[bytes]) -> list:
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