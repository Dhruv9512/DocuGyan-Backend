from typing import Literal
from langgraph.graph import StateGraph, START, END
from psycopg2 import DatabaseError
from DocuGyan.celery import stop_task

from DocuAgent.websocket.notifier import Notifier
from DocuAgent.models import DocuProcess

# schemas
from DocuAgent.schemas.agent_schemas import OrchestratorState
from DocuAgent.schemas.llm_schemas import RAGClassification

# Import the Extractor Agent Subgraph
from DocuAgent.agents.extractor.graph import build_docu_extractor_agent

from DocuAgent.ingestion.VectorDB_Ingestor import build_vector_db_ingestor

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
        self.notifier = Notifier(project_id)  
        self.graph = self._build_graph()

    # ==========================================
    # Conditional Routing Logic (Deterministic)
    # ==========================================
    def ingestion_router(self, state: OrchestratorState) -> Literal["vector_rag_ingest", "graph_rag_ingest", "vectorless_ingest"]:
        """Routes to the correct ingestor based on the LLM's classification."""
        strategy = state.get("rag_strategy", "vector")
        self.notifier.send_message(f"Orchestrator: Routing to {strategy.capitalize()} Ingestor...")
        
        if not state.get("refined_questions_blob_url") or not state.get("extracted_doc_blob_url"):
            self.notifier.send_error(f"Docuextractor Agent failed to produce extracted urls of refined question url and referenced urls. Halting pipeline. State: {state}")
            stop_task()

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
        try:
            result = build_vector_db_ingestor(project_id=self.project_id, extracted_doc_urls=state.get("extracted_doc_blob_url", []))
        except Exception as e:
            self.notifier.send_error(f"Vector Ingestor failed: {str(e)}")
            raise RuntimeError(f"Vector Ingestor failed: {str(e)}") from e
        
        if result:
            self.notifier.send_message("Vector Ingestor: Successfully ingested documents into the vector database.")
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
        # 1. Fetch the record before doing any heavy lifting.
        docu_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)

        notifier.send_message("Initializing DocuPipeline Orchestrator...")
        
        # 2. Initialize and run the orchestrator
        orchestrator = DocuPipelineOrchestrator(project_id=project_id, user_uuid=user_uuid)
        result = orchestrator.run()

        # 3. COMMIT SUCCESS: Mark as completed
        docu_process.status = DocuProcess.StatusChoices.COMPLETED
        docu_process.extracted_doc_urls = result.get("extracted_doc_blob_url", [])
        docu_process.refined_question_urls = result.get("refined_questions_blob_url", [])
        docu_process.ingestion_strategy = result.get("rag_strategy", "")
        docu_process.save(update_fields=['status', 'extracted_doc_urls', 'refined_question_urls', 'ingestion_strategy'])
        
        # Notify completion
        notifier.send_completed(final_result={
            "extracted_docs_blob_urls": result.get("extracted_doc_blob_url"), 
            "refined_questions_blob_url": result.get("refined_questions_blob_url"),
        })

        return result
    
    except DatabaseError as db_err:
        # Catch initial DB fetch errors or commit errors
        error_message = f"Database error during pipeline execution: {str(db_err)}"
        notifier.send_error(error_message)
        raise RuntimeError(error_message) from db_err

    except Exception as e:
        # Catch LangGraph/LLM/System failures gracefully
        try:
            fallback_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)
            fallback_process.status = DocuProcess.StatusChoices.FAILED  # Changed to FAILED
            fallback_process.error_message = str(e)
            fallback_process.save(update_fields=['status', 'error_message'])
        except Exception as fallback_err:
            notifier.send_error(f"Failed to update process status after error: {str(fallback_err)}")
        
        notifier.send_error(str(e))
        raise e