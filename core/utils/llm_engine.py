from django.conf import settings

class LLMEngine:
    """
    The Neutral Engine: A Factory for LLM Clients.
    Only handles initialization and authentication.
    """

    @staticmethod
    def get_groq_client(model_name: str = "llama-3.1-8b-instant", temperature: float = 0.0):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_retries=3,
            groq_api_key=getattr(settings, 'GROQ_API_KEY', None)
        )

    @staticmethod
    def get_gemini_client(model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=getattr(settings, 'GOOGLE_API_KEY', None)
        )

    @staticmethod
    def get_huggingface_client():
        from huggingface_hub import InferenceClient
        return InferenceClient(
            api_key=getattr(settings, 'HUGGINGFACE_API_KEY', None)
        )
    
    @staticmethod
    def get_huggingface_chat_client(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", temperature: float = 0.0):
        """
        Returns a LangChain-compatible HuggingFace Endpoint.
        Note: Requires 'huggingface_hub' and 'langchain-huggingface' packages.
        """
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=temperature,
            huggingfacehub_api_token=getattr(settings, 'HUGGINGFACE_API_KEY', None),
            timeout=30
        )
        return ChatHuggingFace(llm=llm)