import base64
import logging
from typing import List

# Import the neutral Plumbing
from core.utils.llm_engine import LLMEngine
from DocuAgent.schemas.llm_schemas import RefinedBatch
from DocuAgent.prompts.DocuExtractor_Prompts import REFINE_QUESTIONS_PROMPT, PDF_EXTRACTION_PROMPT


logger = logging.getLogger(__name__)

class DocuAgentLLMCalls:
    """
    DocuAgent Specialist: Handles specialized workflows for Document Intelligence.
    Uses LLMEngine to fetch clients and implements resilient fallback logic.

    All LLM chains are lazy-initialized once at the class level and reused
    across all calls — never reconstructed per invocation.
    """
 
    _vision_chain  = None   
    _refiner_chain = None
  

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
    # WORKFLOW 2: Question Refiner
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

        primary = LLMEngine.get_groq_client(
            model_name="llama-3.3-70b-versatile", temperature=0.0
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="deepseek-r1-distill-llama-70b", temperature=0.0
        )
        fallback_2 = LLMEngine.get_groq_client(
            model_name="gemma2-9b-it", temperature=0.0
        )
        fallback_3 = LLMEngine.get_huggingface_chat_client(
            model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.0
        )
        fallback_4 = LLMEngine.get_groq_client(
            model_name="llama-3.1-70b-versatile", temperature=0.0
        )

        cls._refiner_chain = (
            (REFINE_QUESTIONS_PROMPT | primary.with_structured_output(RefinedBatch))
            .with_fallbacks([
                REFINE_QUESTIONS_PROMPT | fallback_1.with_structured_output(RefinedBatch),
                REFINE_QUESTIONS_PROMPT | fallback_2.with_structured_output(RefinedBatch),
                REFINE_QUESTIONS_PROMPT | fallback_3.with_structured_output(RefinedBatch),
                REFINE_QUESTIONS_PROMPT | fallback_4.with_structured_output(RefinedBatch),
            ])
        )

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