from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_google_genai import ChatGoogleGenerativeAI

from DocuAgent.websocket.notifier import Notifier
from DocuAgent.models import DocuProcess
from DocuAgent.agents.schemas import SupervisorState

# Import the Extractor Agent Subgraph
from DocuAgent.agents import DocuExtractorAgent

class DocuSupervisorAgent:
    """
    The Orchestrator. Evaluates the state and dynamically routes to sub-agents.
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
    # The Brain: Command Routing
    # ==========================================
    def supervisor_router(self, state: SupervisorState) -> Command[Literal[
        "extraction_agent", "classify_rag", "vector_rag_ingest", 
        "graph_rag_ingest", "vectorless_ingest", "__end__"
    ]]:
        """
        Evaluates the current state and explicitly COMMANDS the next agent.
        """
        self.notifier.send_message("Supervisor: Evaluating next steps...")

        # 1. If we have URLs but no extracted documents -> Command the Extractor
        if state.get("reference_urls") and not state.get("extracted_doc_blob_url"):
            self.notifier.send_message("Supervisor: Issuing command to Extractor Agent.")
            return Command(goto="extraction_agent")

        # 2. If docs are extracted but not classified -> Command the Classifier
        if state.get("extracted_doc_blob_url") and not state.get("rag_strategy"):
            self.notifier.send_message("Supervisor: Issuing command to Classifier Agent.")
            return Command(goto="classify_rag")

        # 3. If classified but not ingested -> Command the correct Ingestion Agent
        if state.get("rag_strategy") and not state.get("ingestion_done"):
            strategy = state["rag_strategy"]
            self.notifier.send_message(f"Supervisor: Issuing command to {strategy} Ingestion Agent.")
            
            if strategy == "vector":
                return Command(goto="vector_rag_ingest")
            elif strategy == "graph":
                return Command(goto="graph_rag_ingest")
            else:
                return Command(goto="vectorless_ingest")

        # 4. If everything is done -> End the graph
        self.notifier.send_message("Supervisor: All tasks complete. Shutting down pipeline.")
        return Command(goto="__end__")

    # ==========================================
    # Sub-Agent Nodes (Workers)
    # ==========================================
    def classify_rag_strategy(self, state: SupervisorState) -> dict:
        """Worker node: Classifies the strategy."""
        # Future: Pass excerpts to the LLM to get dynamic strategy
        return {
            "rag_strategy": "vector",
            "rag_reasoning": "Default strategy for now",
        }

    def vector_rag_ingest(self, state: SupervisorState) -> dict:
        """Worker node: Ingests to Vector DB."""
        return {"ingestion_done": True}

    def graph_rag_ingest(self, state: SupervisorState) -> dict:
        """Worker node: Ingests to Graph DB."""
        return {"ingestion_done": True}

    def vectorless_ingest(self, state: SupervisorState) -> dict:
        """Worker node: Ingests raw text."""
        return {"ingestion_done": True}

    # ==========================================
    # Network Graph Construction
    # ==========================================
    def _build_graph(self):
        graph = StateGraph(SupervisorState)

        # 1. Add the Supervisor (Router) node
        graph.add_node("supervisor", self.supervisor_router)

        # 2. Add the Sub-Agents (Workers)
        extractor_subgraph = DocuExtractorAgent(self.project_id).build_graph()
        graph.add_node("extraction_agent", extractor_subgraph)
        graph.add_node("classify_rag", self.classify_rag_strategy)
        graph.add_node("vector_rag_ingest", self.vector_rag_ingest)
        graph.add_node("graph_rag_ingest", self.graph_rag_ingest)
        graph.add_node("vectorless_ingest", self.vectorless_ingest)
  
        # 3. The graph always starts at the Supervisor Brain
        graph.add_edge(START, "supervisor")

        # 4. The Star Pattern: Every worker MUST report back to the Supervisor when finished
        graph.add_edge("extraction_agent", "supervisor")
        graph.add_edge("classify_rag", "supervisor")
        graph.add_edge("vector_rag_ingest", "supervisor")
        graph.add_edge("graph_rag_ingest", "supervisor")
        graph.add_edge("vectorless_ingest", "supervisor")

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
#  Builder Function to Initialize and Run the Supervisor Agent
# ==================================================================
def build_docu_supervisor_agent(project_id: str, user_uuid: str):
    """
    Initializes the Docu Supervisor Agent and runs the entire pipeline.
    This function is the main entry point for the DocuAgent system.

    args:
        project_id (str): The unique identifier for the project.
        user_uuid (str): The unique identifier for the user.
    returns:
        dict: Final results including extracted document URLs and refined question URLs.
    """
    notifier = Notifier(project_id)

    try:
        notifier.send_message("Initializing DocuSupervisor Agent pipeline.")

        # Initialize and run the agent
        supervisor = DocuSupervisorAgent(project_id=project_id, user_uuid=user_uuid)
        result = supervisor.run()

        # Mark as completed on success
        docu_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)
        docu_process.status = DocuProcess.StatusChoices.COMPLETED
        docu_process.metadata = {
            "rag_strategy": result.get("rag_strategy"),
            "refined_questions": result.get("refined_questions_blob_url"),
        }
        docu_process.save(update_fields=['status', 'metadata'])
        
        # Notify completion
        notifier.send_completed(final_result={
            "extracted_docs_blob_urls": result.get("extracted_doc_blob_url"), 
            "refined_questions_blob_url": result.get("refined_questions_blob_url")
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