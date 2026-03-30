import base64
import logging
import time
from typing import List

# Import the neutral Plumbing
from core.utils.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

class DocuAgentLLMCalls:
    """
    DocuAgent Specialist: Handles specialized workflows for Document Intelligence.
    Uses LLMEngine to fetch clients and implements resilient fallback logic.
    """

    # ==========================================
    # WORKFLOW 1: Scan-Based PDF Extraction
    # ==========================================
    @classmethod
    def VisionExtractorLLM(cls, image_bytes_list: List[bytes]) -> str:
        """
        Takes a list of JPEG image bytes and returns formatted Markdown.
        Resilient 3-tier fallback using purely LangChain models.
        """
        # LAZY LOAD PROMPT
        from DocuAgent.prompts.DocuExtractor_Prompts import PDF_EXTRACTION_PROMPT

        # 1. Build the Universal LangChain Payload once
        lc_payload = cls._build_langchain_payload(PDF_EXTRACTION_PROMPT, image_bytes_list)
        
        # --- ATTEMPT 1: GEMINI (Primary for Vision) ---
        try:
            logger.info(f"Attempting Vision Extraction: Gemini (Pages: {len(image_bytes_list)})")
            gemini = LLMEngine.get_gemini_client(model_name="gemini-2.0-flash", temperature=0.0)
            return gemini.invoke(lc_payload).content
            
        except Exception as e:
            logger.warning(f"Gemini Vision failed: {e}. Trying Groq...")

            # --- ATTEMPT 2: GROQ (Secondary for Vision) ---
            try:
                groq = LLMEngine.get_groq_client(model_name="llama-3.2-90b-vision-preview", temperature=0.0)
                return groq.invoke(lc_payload).content
                
            except Exception as e2:
                logger.warning(f"Groq Vision failed: {e2}. Trying HuggingFace...")

                # --- ATTEMPT 3: HUGGING FACE (Last Resort) ---
                try:
                    hf_client = LLMEngine.get_huggingface_chat_client(
                        model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
                        temperature=0.0 # Engine will safely convert this to 0.01
                    )
                    # Because HF is now a LangChain client, it accepts the exact same payload
                    return hf_client.invoke(lc_payload).content
                    
                except Exception as e3:
                    logger.error(f"FATAL: All Vision LLMs failed. Final Error: {e3}")
                    raise RuntimeError(f"FATAL: All Vision LLMs failed. Final Error: {e3}")

    # ==========================================
    # WORKFLOW 2: Question Refinement (Structured)
    # ==========================================
    @classmethod
    def call_refine_questions(cls, batch_text: str):
        """
        Industry-level 3-Tier Resilient Call.
        """
        # LAZY LOAD PROMPT & SCHEMA
        from DocuAgent.prompts.DocuExtractor_Prompts import REFINE_QUESTIONS_PROMPT
        from DocuAgent.schemas.llm_schemas import RefinedBatch
        
        # 1. Initialize Clients 
        groq_llm = LLMEngine.get_groq_client(temperature=0.0)
        hf_llm = LLMEngine.get_huggingface_chat_client(temperature=0.0)
        gemini_llm = LLMEngine.get_gemini_client(temperature=0.0)

        # 2. Build the Chains
        tier_1 = (REFINE_QUESTIONS_PROMPT | groq_llm.with_structured_output(RefinedBatch)).with_retry(
            stop_after_attempt=3
        )
        tier_2 = REFINE_QUESTIONS_PROMPT | hf_llm.with_structured_output(RefinedBatch)
        tier_3 = REFINE_QUESTIONS_PROMPT | gemini_llm.with_structured_output(RefinedBatch)

        # 3. Create the Waterfall Chain
        resilient_waterfall = tier_1.with_fallbacks([tier_2, tier_3])

        try:
            # Politeness delay for free tier protection
            time.sleep(1.2) 
            logger.info("Initiating 3-tier resilient call for Question Refinement...")
            return resilient_waterfall.invoke({"questions": batch_text})
            
        except Exception as e:
            logger.error(f"CRITICAL: All 3 LLM Tiers failed. Error: {e}")
            raise RuntimeError(f"CRITICAL: All 3 LLM Tiers failed. Error: {e}") from e

    # ==========================================
    # Universal Payload Builder
    # ==========================================
    @classmethod
    def _build_langchain_payload(cls, prompt_text: str, image_bytes_list: List[bytes]) -> list:
        """
        Builds a standard LangChain HumanMessage containing text and multiple Base64 images.
        Universally compatible with Gemini, Groq, and LangChain-HuggingFace Chat models.
        """
        from langchain_core.messages import HumanMessage
        
        content_blocks = [{"type": "text", "text": prompt_text}]
        for img_bytes in image_bytes_list:
            b64_image = base64.b64encode(img_bytes).decode('utf-8')
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}",
                    "detail": "high" 
                }
            })
        return [HumanMessage(content=content_blocks)]