from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from DocuAgent.websocket.notifier import Notifier
from DocuAgent.models import DocuProcess
from DocuAgent.schemas.agent_schemas import OrchestratorState
from DocuAgent.schemas.llm_schemas import RAGClassification

# Import the Extractor Agent Subgraph
from DocuAgent.agents.extractor.graph import build_docu_extractor_agent

class DocuPipelineOrchestrator:
    """
    A Deterministic Pipeline Orchestrator. 
    Manages a strictly defined, linear ETL workflow for document processing.
    """
    def __init__(self, project_id: str, user_uuid: str):
        self.project_id = project_id
        self.user_uuid = user_uuid
        self.base_instance = DocuProcess.objects.get(
            project_id=project_id, user_uuid=user_uuid
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.notifier = Notifier(project_id)  
        self.graph = self._build_graph()

    # ==========================================
    # Conditional Routing Logic (Deterministic)
    # ==========================================
    def ingestion_router(self, state: OrchestratorState) -> Literal["vector_rag_ingest", "graph_rag_ingest", "vectorless_ingest"]:
        """Routes to the correct ingestor based on the LLM's classification."""
        strategy = state.get("rag_strategy", "vector")
        self.notifier.send_message(f"Orchestrator: Routing to {strategy.capitalize()} Ingestor...")
        
        if strategy == "vector":
            return "vector_rag_ingest"
        elif strategy == "graph":
            return "graph_rag_ingest"
        else:
            return "vectorless_ingest"

    # ==========================================
    # Pipeline Nodes (Workers & Evaluators)
    # ==========================================
    def classify_rag_strategy(self, state: OrchestratorState) -> dict:
        """
        EVALUATOR AGENT: This is the ONLY place the LLM makes a decision in this pipeline.
        It evaluates the extracted text and outputs a structured Pydantic response.
        """
        self.notifier.send_message("Evaluator Agent: Defaulting to Vector Strategy for V1.")
        
        return {
            "rag_strategy": "vector",
            "rag_reasoning": "Vector is the default strategy for MVP.",
        }

    def vector_rag_ingest(self, state: OrchestratorState) -> dict:
        """Worker node: Ingests to Vector DB."""
        self.notifier.send_message("Vector Ingestor: Processing chunks and generating embeddings...")
        # Add actual ingestion logic here in the future
        return {"ingestion_done": True}

    def graph_rag_ingest(self, state: OrchestratorState) -> dict:
        """Worker node: Ingests to Graph DB."""
        self.notifier.send_message("Graph Ingestor: Mapping entities and relationships...")
        # Add actual ingestion logic here in the future
        return {"ingestion_done": True}

    def vectorless_ingest(self, state: OrchestratorState) -> dict:
        """Worker node: Ingests raw text."""
        self.notifier.send_message("Vectorless Ingestor: Storing raw text data...")
        # Add actual ingestion logic here in the future
        return {"ingestion_done": True}

    # ==========================================
    # Network Graph Construction
    # ==========================================
    def _build_graph(self):
        graph = StateGraph(OrchestratorState)

        # 1. Add all nodes
        graph.add_node("extraction_agent", build_docu_extractor_agent(self.project_id))
        graph.add_node("classify_rag", self.classify_rag_strategy)
        graph.add_node("vector_rag_ingest", self.vector_rag_ingest)
        graph.add_node("graph_rag_ingest", self.graph_rag_ingest)
        graph.add_node("vectorless_ingest", self.vectorless_ingest)
  
        # 2. Define the exact linear flow of the pipeline
        
        # Pipeline ALWAYS starts with extraction
        graph.add_edge(START, "extraction_agent")
        
        # Extraction always flows directly into Classification next
        graph.add_edge("extraction_agent", "classify_rag")
        
        # Classification branches out to the specific ingestor
        graph.add_conditional_edges("classify_rag", self.ingestion_router)

        # All ingestors mark the end of the pipeline
        graph.add_edge("vector_rag_ingest", END)
        graph.add_edge("graph_rag_ingest", END)
        graph.add_edge("vectorless_ingest", END)

        return graph.compile()

    def run(self):
        initial_state = {
            "project_id": self.project_id,
            "user_uuid": self.user_uuid,
            "messages": [],
            "reference_urls": self.base_instance.reference_urls,
            "extracted_doc_blob_url": [],
            "rag_strategy": "",
            "ingestion_done": False,
            "rag_reasoning": "",
            "original_questions": self.base_instance.question_urls if self.base_instance.question_urls else self.base_instance.text_questions,
            "refined_questions_blob_url": [],
        }
        return self.graph.invoke(initial_state)

# ==================================================================
#  Builder Function to Initialize and Run the Orchestrator
# ==================================================================
def build_docu_pipeline_orchestrator(project_id: str, user_uuid: str):
    """
    Initializes the Docu Pipeline Orchestrator and runs the entire pipeline.
    This function is the main entry point for the DocuAgent system.

    args:
        project_id (str): The unique identifier for the project.
        user_uuid (str): The unique identifier for the user.
    returns:
        dict: Final results including extracted document URLs and refined question URLs.
    """
    notifier = Notifier(project_id)

    try:
        notifier.send_message("Initializing DocuPipeline Orchestrator...")

        # Initialize and run the orchestrator
        orchestrator = DocuPipelineOrchestrator(project_id=project_id, user_uuid=user_uuid)
        result = orchestrator.run()

        # Mark as completed on success
        docu_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)
        docu_process.status = DocuProcess.StatusChoices.COMPLETED
        docu_process.metadata = {
            "rag_strategy": result.get("rag_strategy"),
            "refined_questions": result.get("refined_questions_blob_url"),
            "extracted_docs": result.get("extracted_doc_blob_url"),
        }
        docu_process.save(update_fields=['status', 'metadata'])
        
        # Notify completion
        notifier.send_completed(final_result={
            "extracted_docs_blob_urls": result.get("extracted_doc_blob_url"), 
            "refined_questions_blob_url": result.get("refined_questions_blob_url"),
        })

        return result

    except Exception as e:
        # Handle failures gracefully
        docu_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)
        docu_process.status = DocuProcess.StatusChoices.PENDING
        docu_process.error_message = str(e)
        docu_process.save(update_fields=['status', 'error_message'])
        
        notifier.send_error(str(e))
        raise e