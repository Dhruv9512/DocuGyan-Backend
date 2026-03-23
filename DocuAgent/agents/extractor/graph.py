from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from DocuAgent.utils.extraction import build_DocuExtractor
from DocuAgent.utils.query_processing import QuestionRefiner
from DocuAgent.websocket.notifier import Notifier
from DocuAgent.agents.schemas import ExtractorState, ExtractionWorkerState


class DocuExtractorAgent:
    """
    A specialized sub-agent that uses Map-Reduce to download, extract, 
    and format documents in parallel, followed by refining questions.
    """
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.notifier = Notifier(project_id)
        # Assuming QuestionRefiner takes project_id or handles it internally
        self.question_refiner = QuestionRefiner(project_id=project_id, file_url="") 

    # ==========================================
    # Nodes & Routing
    # ==========================================

    def dispatch_extraction(self, state: ExtractorState):
        """
        FAN-OUT: Reads all URLs and maps them to parallel worker nodes.
        """
        self.notifier.send_message("Extractor Agent: Starting parallel document extraction...")
        urls = state.get("reference_urls", [])
        
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
            # Execute the extraction logic
            blob_url = build_DocuExtractor(project_id=project_id, url=url)
            # Must return as a list so operator.add can combine them
            return {"extracted_doc_blob_url": [blob_url]}
        except Exception as e:
            self.notifier.send_error(f"Failed to extract {url}: {str(e)}")
            # Return empty list on failure so the reducer doesn't break
            return {"extracted_doc_blob_url": []} 

    def refine_questions(self, state: ExtractorState):
        """
        WORKER: Runs in parallel with extraction. 
        Takes the original text questions and refines them using the LLM.
        """
        self.notifier.send_message("Extractor Agent: Starting question refinement in parallel...")
        
        original_questions = state.get("original_questions", [])
        refined_urls = []
        
        if original_questions:
            try:
                # If your QuestionRefiner is adapted to take a list of strings directly:
                refined_urls = self.question_refiner.refine(original_questions)
                self.notifier.send_message("Extractor Agent: Question refinement complete.")
            except Exception as e:
                self.notifier.send_error(f"Failed to refine questions: {str(e)}")
        else:
            self.notifier.send_message("Extractor Agent: No original questions provided. Skipping refinement.")
            
        return {"refined_questions_blob_url": refined_urls}

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
    

