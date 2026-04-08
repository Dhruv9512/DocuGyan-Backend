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
from DocuAgent.agents.academic.graph import build_academic_agent
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
        
        if not state.get("extracted_questions_blob_url") or not state.get("extracted_doc_blob_url"):
            self.notifier.send_error(f"Docuextractor Agent failed to produce extracted urls of refined question url and referenced urls. Halting pipeline. State: {state}")
            stop_task()

        if strategy == "vector":
            return "vector_rag_ingest"
        elif strategy == "graph":
            return "graph_rag_ingest"
        else:
            return "vectorless_ingest"
    
    def domain_router(self, state: OrchestratorState) -> Literal["academic_agent", "financial_agent", "audit_agent"]:
        """Routes to the correct specialist agent based on domain classification."""
        domain = state.get("domain") or "audit" 
        self.notifier.send_message(f"Orchestrator: Routing to {domain.capitalize()} Specialist Agent...")
        
        if domain == "academic":
            return "academic_agent"
        elif domain == "financial":
            return "financial_agent"
        elif domain == "audit":
            return "audit_agent"
        else:
            return "academic_agent" 

    # =========================================
    # Evaluator Nodes for Classification
    # =========================================
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
    
    def classify_domain(self, state: OrchestratorState) -> dict:
        """
        EVALUATOR AGENT: Reads a snippet of references and questions to determine the domain.
        """
        self.notifier.send_message("Evaluator Agent: Classifying document domain...")
        
        # NOTE: A real LLM structured output call will go here using the DomainClassification schema.
        # For now, we mock the response to allow the graph to run and test the router.
        return {
            "domain": "academic"
        }

    # =================================
    # Pipeline Nodes for Ingestion 
    # =================================
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
    

    # =========================================
    # 
    # =========================================
    def financial_agent(self, state: OrchestratorState) -> dict:
        """Placeholder for Financial Sub-graph"""
        self.notifier.send_message("Financial Agent: Processing financial models and tables...")
        return {}

    def audit_agent(self, state: OrchestratorState) -> dict:
        """Placeholder for Audit Sub-graph"""
        self.notifier.send_message("Audit Agent: Cross-referencing compliance and risk factors...")
        return {}
    
    def academic_agent(self, state: OrchestratorState) -> dict:
        """Placeholder for Academic Sub-graph"""
        self.notifier.send_message("Academic Agent: Analyzing theories and citations...")
        return {}

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
        
        # New Domain & Sub-Agent Nodes
        graph.add_node("classify_domain", self.classify_domain)
        graph.add_node("academic_agent", build_academic_agent(self.project_id))
        graph.add_node("financial_agent", self.financial_agent)
        graph.add_node("audit_agent", self.audit_agent)
  
        # 2. Define the Flow
        
        # Extraction -> RAG Classification
        graph.add_edge(START, "extraction_agent")
        graph.add_edge("extraction_agent", "classify_rag")
        
        # RAG Classification -> Specific Ingestor
        graph.add_conditional_edges("classify_rag", self.ingestion_router)

        # ALL Ingestors -> Domain Classification
        graph.add_edge("vector_rag_ingest", "classify_domain")
        graph.add_edge("graph_rag_ingest", "classify_domain")
        graph.add_edge("vectorless_ingest", "classify_domain")

        # Domain Classification -> Specialist Agent
        graph.add_conditional_edges("classify_domain", self.domain_router)

        # All specialist agents mark the end of the pipeline
        graph.add_edge("academic_agent", END)
        graph.add_edge("financial_agent", END)
        graph.add_edge("audit_agent", END)

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
            "extracted_questions_blob_url": [],
            "domain": "",
            "final_answers_blob_url": [],
        }

        config = {
            "max_concurrency": 10,  
            "recursion_limit": 50,
            "configurable": {
                "thread_id": self.project_id
            }
        }
        return self.graph.invoke(initial_state, config=config)

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
        docu_process.refined_question_urls = result.get("extracted_questions_blob_url", [])
        docu_process.ingestion_strategy = result.get("rag_strategy", "")
        docu_process.results_url = result.get("final_answers_blob_url", [])
        docu_process.save(update_fields=['status', 'extracted_doc_urls', 'refined_question_urls', 'ingestion_strategy', 'results_url'])
        
        # Notify completion
        notifier.send_completed(final_result={
            "extracted_docs_blob_urls": result.get("extracted_doc_blob_url"), 
            "extracted_questions_blob_url": result.get("extracted_questions_blob_url"),
            "final_answers_blob_url": result.get("final_answers_blob_url"),
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
            fallback_process.status = getattr(DocuProcess.StatusChoices, 'FAILED', 'FAILED')
            fallback_process.error_message = str(e)
            fallback_process.save(update_fields=['status', 'error_message'])
        except Exception as fallback_err:
            notifier.send_error(f"Failed to update process status after error: {str(fallback_err)}")
        
        notifier.send_error(str(e))
        raise e