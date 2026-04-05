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
                groq = LLMEngine.get_groq_client(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0)
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