# langgraph and langchain imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Import utility and ingestion classes
from DocuAgent.utils import DocuExtractor, QuestionRefiner
from DocuAgent.ingestion import VectorDBIngestor


# modules import
from DocuAgent.models import DocuProcess

# Import graph states and pydantic models
from DocuAgent.agents.schemas import (
    SupervisorState,
    RAGClassification,
    ExtractedDocument,
)

from DocuAgent.agents.supervisor.prompts import (
    CLASSIFY_RAG_PROMPT,
    REFINE_QUESTIONS_PROMPT,
)


# ==========================================
#  The Supervisor Agent Class
# ==========================================
class SupervisorAgent:
    """
    An LLM-driven agent that analyzes incoming documents and dynamically
    routes them to the correct RAG ingestion pipeline.
    """

    def __init__(self, project_id: str, user_uuid: str):
        self.project_id = project_id
        self.user_uuid = user_uuid
        self.base_instance = DocuProcess.objects.get(
            project_id=project_id, user_uuid=user_uuid
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.graph = self._build_graph()
        self.extractor = DocuExtractor()
        self.question_refiner = QuestionRefiner()

    # ==========================================
    # Graph Construction
    # ==========================================

    def _build_graph(self):
        graph = StateGraph(SupervisorState)

        # Nodes
        graph.add_node("extract", self.extract_documents)
        graph.add_node("classify_rag", self.classify_rag_strategy)
        graph.add_node("vector_rag_ingest", self.vector_rag_ingest)
        graph.add_node("graph_rag_ingest", self.graph_rag_ingest)
        graph.add_node("vectorless_ingest", self.vectorless_ingest)
  

        # Flow: extract → classify → route to ingestor → End
        graph.set_entry_point("extract")
        graph.add_edge("extract", "classify_rag")
        graph.add_conditional_edges("classify_rag", self.route_to_rag)
        graph.add_edge("vector_rag_ingest", END)
        graph.add_edge("graph_rag_ingest", END)
        graph.add_edge("vectorless_ingest", END)


        return graph.compile()

    # ==========================================
    # Nodes
    # ==========================================
    def extract_documents(self, state):
        urls = state["reference_urls"]
        original_questions = state["original_questions"]
        # Send each URL to the worker node
        send_states = [{"url": url} for url in urls]
        # Use LangGraph's Send API to dispatch
        extracted_results = self.graph.send("extract_single_url_worker", send_states)
        # Aggregate results
        extracted_docs = [result["extracted_doc"] for result in extracted_results]
        # Refine questions (add your logic)
        refined_questions = self.question_refiner.refine(original_questions)
        return {
            "extracted_docs": extracted_docs,
            "refined_questions": refined_questions,
        }

    def classify_rag_strategy(self, state: SupervisorState) -> dict:
        """For now, always use vector RAG."""
        return {
            "rag_strategy": "vector",
            "rag_reasoning": "Default strategy for now",
        }

    def route_to_rag(self, state: SupervisorState) -> str:
        """Map strategy name to graph node name."""
        strategy_map = {
            "vector": "vector_rag_ingest",
            "graph": "graph_rag_ingest",
            "vectorless": "vectorless_ingest",
        }
        return strategy_map[state["rag_strategy"]]

    def vector_rag_ingest(self, state: SupervisorState) -> dict:
        """Deterministic node: chunk → embed → store in vector DB."""
        pass
        return {"ingestion_done": True}

    def graph_rag_ingest(self, state: SupervisorState) -> dict:
        """Deterministic node: extract entities/relations → store in graph DB."""
        # TODO: implement GraphDBIngestor
        return {"ingestion_done": True}

    def vectorless_ingest(self, state: SupervisorState) -> dict:
        """Deterministic node: store raw text with keyword index."""
        # TODO: implement VectorlessIngestor
        return {"ingestion_done": True}

    # ==========================================
    # Worker Nodes
    # =========================================
    def extract_single_url_worker(self, state):
        # This is a worker node that can be called by the main graph to extract individual referance documents in parallel if needed.
        url = state["url"]
        content = self.extractor.extract_from_url(url)
        file_type = self.extractor._get_file_extension(url)
        return {"extracted_doc": ExtractedDocument(url=url, content=content, file_type=file_type)}
    
    # ==========================================
    # Entry Point
    # ==========================================

    def run(self):
        """Run the full supervisor pipeline."""
        initial_state = {
            "project_id": self.project_id,
            "user_uuid": self.user_uuid,
            "messages": [],
            "reference_urls": self.base_instance.reference_urls,
            "extracted_docs": [],
            "rag_strategy": "",
            "ingestion_done": False,
            "rag_reasoning": "",
            "original_questions": self.base_instance.question_urls if self.base_instance.question_urls else self.base_instance.text_questions,
            "refined_questions": [],
        }
        return self.graph.invoke(initial_state)


# ==================================================================
#  Builder Function to Initialize and Run the Supervisor Agent
# ==================================================================
def build_supervisor_agent(project_id: str, user_uuid: str):
    """
    Factory function to create and run the SupervisorAgent.
     - Initializes the agent with the given project_id and user_uuid.
     - Executes the agent's run method to start the processing pipeline.
     - Returns the final result of the agent's execution.

    args:
        project_id (str): The unique identifier for the document processing project.
        user_uuid (str): The unique identifier for the user who initiated the process.
    """
    try:
        # Initialize and run the agent
        supervisor = SupervisorAgent(project_id=project_id, user_uuid=user_uuid)
        result = supervisor.run()

        # Mark as completed on success
        docu_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)
        docu_process.status = DocuProcess.StatusChoices.COMPLETED
        docu_process.metadata = {
            "rag_strategy": result.get("rag_strategy"),
            "refined_questions": result.get("refined_questions"),
        }
        docu_process.save(update_fields=['status', 'metadata'])
        
        return result

    except Exception as e:
        # Handle domain-level failures and record them
        docu_process.status = DocuProcess.StatusChoices.PENDING
        docu_process.error_message = str(e)
        docu_process.save(update_fields=['status', 'error_message'])
        
        raise e



    