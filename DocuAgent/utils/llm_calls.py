import base64
import logging
from typing import List, Any
from django.conf import settings

# prompts
from DocuAgent.prompts.docu_extractor import PDF_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

class LLMUtility:
    """
    Centralized utility for handling all LLM interactions.
    Uses Lazy Loading (imports inside methods) and Class Methods 
    to preserve server RAM and prevent circular dependency issues in Django.
    """

    # ==========================================
    # Core LLM Callers (Text & Vision)
    # ==========================================

    @classmethod
    def gemini(cls, model_name: str, input_messages: list, temperature: float = 0.5) -> str:
        """Standard method for Gemini calls."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        try:
            gemini_client = ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=temperature, 
                google_api_key=getattr(settings, 'GOOGLE_API_KEY', None)
            )
            response = gemini_client.invoke(input_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Gemini call failed: {str(e)}")
            raise e

    @classmethod
    def groq(cls, model_name: str, input_messages: list, temperature: float = 0.5) -> str:
        """Standard method for Groq calls."""
        from langchain_groq import ChatGroq
        
        try:
            groq_client = ChatGroq(
                model=model_name,
                temperature=temperature,
                groq_api_key=getattr(settings, 'GROQ_API_KEY', None)
            )
            response = groq_client.invoke(input_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Groq call failed: {str(e)}")
            raise e

    @classmethod
    def huggingface(cls, model_name: str, input_messages: list) -> str:
        """Standard method for HuggingFace Serverless Inference calls."""
        from huggingface_hub import InferenceClient
        from huggingface_hub.errors import HfHubHTTPError # Import the error handler
        
        try:
            hf_client = InferenceClient(
                api_key=getattr(settings, 'HUGGINGFACE_API_KEY', None)
            )
            response = hf_client.chat.completions.create(
                model=model_name, 
                messages=input_messages
            )
            return response.choices[0].message.content
            
        except HfHubHTTPError as e:
            # specifically catch rate limits or model loading errors
            logger.error(f"HuggingFace API Error (Might be rate limited or loading): {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"HuggingFace general call failed: {str(e)}")
            raise e

    # ==========================================
    # Specialized Workflows
    # ==========================================

    @classmethod
    def PDFExtractorLLM(cls, image_bytes_list: List[bytes]) -> str:
        """
        Takes a list of JPEG image bytes and returns the formatted Markdown string.
        Executes the resilient 3-tier fallback strategy utilizing the core callers.
        """
        # 1. Build the LangChain Payload (Compatible with Gemini & Groq)
        langchain_payload = cls._build_langchain_payload(PDF_EXTRACTION_PROMPT, image_bytes_list)
        
        # --- Attempt 1: GEMINI ---
        try:
            logger.info(f"Attempting Vision Extraction via Gemini (Batch size: {len(image_bytes_list)})")
            return cls.gemini(
                model_name="gemini-2.0-flash", 
                input_messages=langchain_payload
            )
        except Exception as e1:
            logger.warning(f"Gemini Vision failed: {str(e1)}. Initiating Fallback 1 (Groq)...")
            
            # --- Attempt 2: GROQ ---
            try:
                return cls.groq(
                    model_name="llama-3.2-90b-vision-preview", 
                    input_messages=langchain_payload
                )
            except Exception as e2:
                logger.warning(f"Groq Vision failed: {str(e2)}. Initiating Fallback 2 (HuggingFace)...")
                
                # --- Attempt 3: HUGGING FACE ---
                try:
                    # HuggingFace expects raw dict messages, not Langchain objects
                    hf_payload = cls._build_hf_payload(PDF_EXTRACTION_PROMPT, image_bytes_list)
                    return cls.huggingface(
                        model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", 
                        input_messages=hf_payload
                    )
                except Exception as e3:
                    logger.error(f"FATAL: All Vision LLMs failed. HF Error: {str(e3)}")
                    raise RuntimeError("All Vision AI fallbacks exhausted. Extraction failed.")

    # ==========================================
    # Internal Payload Builders
    # ==========================================

    @classmethod
    def _build_langchain_payload(cls, prompt_text: str, image_bytes_list: List[bytes]) -> list:
        """Constructs the HumanMessage multimodal payload for Langchain."""
        from langchain_core.messages import HumanMessage
        
        content_blocks = [{"type": "text", "text": prompt_text}]
        for img_bytes in image_bytes_list:
            b64_image = base64.b64encode(img_bytes).decode('utf-8')
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            })
            
        return [HumanMessage(content=content_blocks)]

    @classmethod
    def _build_hf_payload(cls, prompt_text: str, image_bytes_list: List[bytes]) -> list:
        """Constructs the raw dictionary multimodal payload for HuggingFace Inference API."""
        content_blocks = [{"type": "text", "text": prompt_text}]
        for img_bytes in image_bytes_list:
            b64_image = base64.b64encode(img_bytes).decode('utf-8')
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            })
            
        return [{"role": "user", "content": content_blocks}]