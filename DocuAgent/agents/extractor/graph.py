from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from DocuAgent.utils.extraction import build_DocuExtractor
from DocuAgent.utils.query_processing import build_QuestionRefiner
from DocuAgent.websocket.notifier import Notifier
from DocuAgent.schemas.agent_schemas import ExtractorState, ExtractionWorkerState

class DocuExtractorAgent:
    """
    A specialized sub-agent that uses Map-Reduce to download, extract, 
    and format documents in parallel, followed by refining questions.
    """
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.notifier = Notifier(project_id)

    # ==========================================
    # Nodes & Routing
    # ==========================================

    def dispatch_extraction(self, state: ExtractorState):
        """
        FAN-OUT: Reads all URLs and maps them to parallel worker nodes.
        """
        urls = state.get("reference_urls", [])
        self.notifier.send_message(
            f"Extractor Agent: Starting parallel document extraction for {len(urls)} document(s)...",
            current_node="extractor",
            status="processing",
        )
        
        if not urls:
            self.notifier.send_error(
                "Extractor Agent: No reference URLs provided. Skipping extraction.",
                current_node="extractor",
            )
            raise ValueError("No reference URLs provided for extraction.")
        
        # Spin up multiple 'extract_single_url_worker' nodes at the same time
        return [
            Send("extract_single_url_worker", {"url": url, "project_id": self.project_id}) 
            for url in urls
        ]

    def extract_single_url_worker(self, state: ExtractionWorkerState):
        """
        WORKER: Extracts a single URL. Runs concurrently for every document.
        """
        url = state["url"]
        project_id = state["project_id"]
        
        try:
            self.notifier.send_message(
                f"Extractor Agent: Extracting document from {url}...",
                current_node="extractor",
                status="processing",
            )
            # Execute the extraction logic
            blob_url = build_DocuExtractor(project_id=project_id, url=url)
            self.notifier.send_message(
                f"Extractor Agent: Extraction complete for {url}.",
                current_node="extractor",
                status="completed",
            )
            # Must return as a list so operator.add can combine them
            return {"extracted_doc_blob_url": [blob_url]}
        except Exception as e:
            self.notifier.send_error(f"Failed to extract: {str(e)}", current_node="extractor")
            # Return empty list on failure so the reducer doesn't break
            raise RuntimeError(f"Failed to extract: {str(e)}") from e

    def refine_questions(self, state: ExtractorState):
        """
        WORKER: Runs in parallel with extraction. 
        Takes the original text questions and refines them using the LLM.
        """
        self.notifier.send_message(
            "Extractor Agent: Starting question refinement in parallel...",
            current_node="extractor",
            status="processing",
        )
        
        original_questions = state.get("original_questions", [])
        refined_urls = []
        
        if original_questions:
            try:
                source_count = len(original_questions) if isinstance(original_questions, list) else 1
                self.notifier.send_message(
                    f"Extractor Agent: Refining questions from {source_count} source(s)...",
                    current_node="extractor",
                    status="processing",
                )
                # If your QuestionRefiner is adapted to take a list of strings directly:
                refined_urls = build_QuestionRefiner(self.project_id, original_questions)
                self.notifier.send_message(
                    "Extractor Agent: Question refinement complete.",
                    current_node="extractor",
                    status="completed",
                )
            except Exception as e:
                self.notifier.send_error(f"Failed to refine questions: {str(e)}", current_node="extractor")
                raise RuntimeError(f"Failed to refine questions: {str(e)}") from e
        else:
            self.notifier.send_error(
                "Extractor Agent: No original questions provided. Skipping refinement.",
                current_node="extractor",
            )
            raise ValueError("No original questions provided for refinement.")
            
        final_url = refined_urls.get("extracted_questions_blob_url")
        self.notifier.send_message(
            f"Extractor Agent: Final refined questions URL: {final_url}",
            current_node="extractor",
            status="completed",
        )
        return {"extracted_questions_blob_url": [final_url] if final_url else []}
    
    
    # ==========================================
    # Graph Construction
    # ==========================================
    def build_graph(self):
        graph = StateGraph(ExtractorState)
        
        # Add the nodes
        graph.add_node("extract_single_url_worker", self.extract_single_url_worker)
        graph.add_node("refine_questions", self.refine_questions)
         
        # Branch 1: Dispatch all URLs to parallel extraction workers
        graph.add_conditional_edges(START, self.dispatch_extraction, ["extract_single_url_worker"])
        
        # Branch 2: Start refining questions immediately
        graph.add_edge(START, "refine_questions")
        
        # LangGraph implicitly waits at the END node for all active parallel branches to complete, so we just need to connect the workers to END
        graph.add_edge("extract_single_url_worker", END)
        graph.add_edge("refine_questions", END)
        
        return graph.compile()
    


# build DocuExtractor and QuestionRefiner
def build_docu_extractor_agent(project_id: str) -> DocuExtractorAgent:
    """
    Factory function to build the DocuExtractorAgent with error handling.

    arges:
        project_id (str): The unique identifier for the project, used for scoping resources and logging.
    """
    try:
        return DocuExtractorAgent(project_id=project_id).build_graph()
    except Exception as e:
        raise RuntimeError(f"Failed to build DocuExtractor agent: {str(e)}") from e
