from django.conf import settings

class LLMEngine:
    """
    The Neutral Engine: A Factory for LangChain-compatible LLM Clients.
    Only handles initialization, authentication, and standardizing interfaces.
    """

    @staticmethod
    def get_groq_client(model_name: str = "llama-3.1-8b-instant", temperature: float = 0.0):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_retries=3,
            api_key=getattr(settings, 'GROQ_API_KEY', None)
        )

    @staticmethod
    def get_gemini_client(model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=getattr(settings, 'GOOGLE_API_KEY', None)
        )
    
    @staticmethod
    def get_huggingface_chat_client(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", temperature: float = 0.0):
        """
        Returns a LangChain-compatible HuggingFace Endpoint.
        """
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        
        safe_temp = 0.01 if temperature <= 0.0 else temperature

        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=safe_temp,
            huggingfacehub_api_token=getattr(settings, 'HUGGINGFACE_API_KEY', None),
            max_retries=3
        )
        return ChatHuggingFace(llm=llm)
    
    @staticmethod
    def get_huggingface_embedding_client(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
        
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=getattr(settings, 'HUGGINGFACE_API_KEY', None),
            max_retries=3
        )
        return HuggingFaceEmbeddings(llm=llm)