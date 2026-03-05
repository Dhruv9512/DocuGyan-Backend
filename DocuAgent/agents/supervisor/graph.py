from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate

# Import your extraction/ingestion tools
from ...extraction import DocumentExtractor
from ...ingestion import VectorDBIngestor

# ==========================================
# 1. Define the Graph State
# ==========================================
class AgentState(TypedDict):
    project_id: str
    chosen_strategy: str  

# ==========================================
# 2. The Supervisor Agent Class
# ==========================================
class SupervisorAgent:
    """
    An LLM-driven agent that analyzes incoming documents and dynamically 
    routes them to the correct RAG ingestion pipeline.
    """
    pass