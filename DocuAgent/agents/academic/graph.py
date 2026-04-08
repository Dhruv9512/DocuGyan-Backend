import logging
import re
from typing import List, Union, Dict, Tuple
import concurrent.futures

from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END

from DocuAgent.websocket.notifier import Notifier
from DocuAgent.schemas.agent_schemas import AcademicAgentState, QuestionState
from DocuAgent.schemas.llm_schemas import PlannerOutput
from DocuAgent.utils.utility import get_collection_name, upload_to_vercel_blob, get_request_session_with_blob_auth
from DocuAgent.utils.llm_calls import DocuAgentLLMCalls
from DocuAgent.agents.academic.tools.CRAG import build_CorrectiveRetriever

logger = logging.getLogger(__name__)

# ============================================================================
#  SUB-GRAPH: The Question Worker
# ============================================================================
class QuestionWorkerGraph:
    """Processes a single question concurrently using deterministic planning."""
    

    # ── Node 1: Plan ────────────────────────────────────────────────────────
    def plan_execution(self, state: QuestionState) -> dict:
        question = state.get("original_question", "No question provided.")
        try:
            plan = DocuAgentLLMCalls.call_question_planner(question)
        except Exception as exc:
            logger.warning(f"[Plan] LLM failed, using fallback for: {question} -> {exc}")
            plan = PlannerOutput(
                steps=["1. Answer question based on context."], 
                question_category="academic", 
                allocated_marks=None, 
                target_word_count=200,
                requires_code=False, 
                requires_diagram=False, 
                is_comparison=False, 
                core_entities=[]
            )
        return {"plan": plan}

    # ── Node 2: Retrieve ────────────────────────────────────────────────────
    def retrieve_context(self, state: QuestionState) -> dict:
        try:
            result = build_CorrectiveRetriever(
                project_id=state["project_id"],
                search_queries=state["original_question"],
            )
            return {
                "retrieved_docs": result.get("retrieved_docs", []),
                "grader_assessment": result.get("grader_assessment"),
            }
        except Exception as exc:
            logger.error(f"[Retrieve] CorrectiveRetriever failed: {exc}")
            return {"retrieved_docs": []}

    def _format_retrieved_context(self, retrieved_docs: List[Document]) -> str:
        if not retrieved_docs:
            return "No relevant context available."

        context_lines = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_type = doc.metadata.get("type", "internal") 
            source = doc.metadata.get("source", "Unknown Source")
            tag = "[WEB]" if doc_type == "web" else "[INTERNAL]"
            
            page = doc.metadata.get("page_number")
            page_str = f" (Page {page})" if page else ""
            
            header = f"--- Source [{i}]: {tag} {source}{page_str} ---"
            context_lines.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n".join(context_lines)

    # ── Node 3: Draft ───────────────────────────────────────────────────────
    def draft_answer(self, state: QuestionState) -> dict:
        plan: PlannerOutput = state.get("plan")
        context: str = self._format_retrieved_context(state.get("retrieved_docs", []))

        try:
            draft = DocuAgentLLMCalls.call_answer_drafter(
                question=state["original_question"],
                context=context,
                plan=plan,
            )
            
            # Validation: Handle empty or purely whitespace returns from LLM
            if not draft or not str(draft).strip():
                raise ValueError("LLM returned an empty response.")
                
        except Exception as exc:
            logger.error(f"[Draft Answer] Failed for '{state['original_question'][:30]}...' | error={exc}")
            draft = "*System Note: Unable to generate a reliable draft for this question due to an internal processing error.*"

        return {"draft_answer": draft}

    # ── Conditional Routing ──────────────────────────────────────────────────
    def should_fetch_diagrams(self, state: QuestionState) -> str:
        """Routes to diagram fetching only if the plan dictates it."""
        plan: PlannerOutput = state.get("plan")
        if plan and plan.requires_diagram:
            return "get_diagrams"
        return "format"

    # ── Node 4: Get Diagrams (Mermaid Generation) ───────────────────────────
    def get_diagrams(self, state: QuestionState) -> dict:
        """
        Scans the draft for {diagram_N} placeholders, extracts the hint sentence
        written by the drafter immediately before each one, then calls the LLM
        diagram generator in parallel for each unique placeholder.
        """
        draft    = state.get("draft_answer", "")
        plan     = state.get("plan")
        category = plan.question_category if plan else "academic"
        original_question = state.get("original_question", "Unknown Question")

        # ── 1. Find all unique placeholders ──────────────────────────────────
        # Uses dict.fromkeys to deduplicate while preserving order
        placeholders = list(dict.fromkeys(re.findall(r"\{diagram_\d+\}", draft)))
        
        if not placeholders:
            return {"diagram_mapping": {}}

        # ── 2. Extract the hint sentence that precedes each placeholder ───────
        def _extract_hint(placeholder: str) -> str:
            """
            Extracts the sentence immediately preceding the placeholder to use
            as highly specific context for the diagram generator.
            """
            # Match the last complete sentence before the placeholder
            pattern = rf'([^.!?\n]+[.!?])\s*{re.escape(placeholder)}'
            match = re.search(pattern, draft)
            
            if match:
                return match.group(1).strip()
                
            # Fallback 1: use first core entity from the plan
            if plan and plan.core_entities:
                return f"Diagram illustrating {plan.core_entities[0]}"
                
            # Fallback 2: use the original question
            return f"Diagram for: {original_question[:80]}"

        # ── 3. Build (placeholder → concept, hint) mapping ───────────────────
        primary_concept = (
            plan.core_entities[0]
            if plan and plan.core_entities
            else original_question[:80]
        )

        tasks = {
            ph: (primary_concept, _extract_hint(ph))
            for ph in placeholders
        }

        # ── 4. Call diagram LLM in parallel ──────────────────────────────────
        diagram_mapping: Dict[str, str] = {}

   
        # Execute the LLM calls concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._generate_one, ph, concept, hint, category): ph
                for ph, (concept, hint) in tasks.items()
            }
            for future in concurrent.futures.as_completed(futures):
                placeholder, rendered_diagram = future.result()
                diagram_mapping[placeholder] = rendered_diagram

        logger.info(f"[GetDiagrams] Resolved {len(diagram_mapping)}/{len(placeholders)} diagrams.")
        
        return {"diagram_mapping": diagram_mapping}

    # ── Node 5: Format ──────────────────────────────────────────────────────
    def format_final(self, state: QuestionState) -> dict:
        """Injects diagrams and finalizes markdown."""
        draft = state.get("draft_answer", "")
        diagram_mapping = state.get("diagram_mapping", {})
        
        # Extract variable safely
        original_question = state.get("original_question", "Unknown Question")

        # Safety check: if draft is an error message, skip diagram processing
        if draft.startswith("*System Note"):
            final_md = f"### {original_question}\n\n{draft}\n\n---"
            return {"completed_answers": [final_md]}

        # Replace placeholders with markdown image syntax
        for placeholder, markdown_image in diagram_mapping.items():
            draft = draft.replace(placeholder, markdown_image)

        # Ensure no leftover, unfetched placeholders remain in the text
        draft = re.sub(r"\{diagram_\d+\}", "", draft)

        final_md = f"### {original_question}\n\n{draft.strip()}\n\n---"
        return {"completed_answers": [final_md]}

    # ── Node 6: Error Fallback ──────────────────────────────────────────────
    def handle_error(self, state: QuestionState) -> dict:
        """Global catch-all for catastrophic node failures."""
        # Extract variable safely
        original_question = state.get("original_question", "Unknown Question")
        err = f"### {original_question}\n\n*Critical error processing this question.*\n\n---"
        return {"completed_answers": [err], "failed_questions": [original_question]}

    def build(self):
        wf = StateGraph(QuestionState)
        
        wf.add_node("plan", self.plan_execution)
        wf.add_node("retrieve", self.retrieve_context)
        wf.add_node("draft", self.draft_answer)
        wf.add_node("get_diagrams", self.get_diagrams)
        wf.add_node("format", self.format_final)
        wf.add_node("error", self.handle_error)

        # Standard Linear Flow
        wf.add_edge(START, "plan")
        wf.add_edge("plan", "retrieve")
        wf.add_edge("retrieve", "draft")
        
        # Conditional Flow for Diagrams
        wf.add_conditional_edges(
            "draft", 
            self.should_fetch_diagrams, 
            {
                "get_diagrams": "get_diagrams", 
                "format": "format"
            }
        )
        
        wf.add_edge("get_diagrams", "format")
        wf.add_edge("format", END)
        
        wf.set_entry_point("plan")

        
        return wf.compile()

    # =======================================================================
    #  Helper methods for Question Worker
    # =======================================================================
    def _generate_one(self,placeholder: str, concept: str, hint: str, category: str) -> Tuple[str, str]:
        try:
            # Call your lazy-loaded chain from DocuAgentLLMCalls
            result = DocuAgentLLMCalls.call_diagram_generator(
                concept=concept,
                hint=hint,
                question_category=category,
            )

            # Check if the LLM actively decided a diagram was impossible/inappropriate
            if getattr(result, "diagram_type", "none") == "none" or not getattr(result, "diagram_code", "").strip():
                logger.info(f"[GetDiagrams] LLM returned diagram_type='none' for {placeholder}")
                fallback = getattr(result, "fallback_text", "Diagram unavailable.")
                return placeholder, f"> *{fallback}*"

            # Wrap in fenced code block + caption for markdown rendering
            rendered = (
                f"```mermaid\n"
                f"{result.diagram_code.strip()}\n"
                f"```\n"
                f"*{result.caption}*"
            )
            return placeholder, rendered

        except Exception as exc:
            logger.warning(f"[GetDiagrams] Failed for {placeholder} | error={exc}")
            # Graceful degradation — remove the placeholder rather than crash
            concept_label = concept[:40]
            return placeholder, f"*[Diagram of {concept_label} unavailable]*"

# ============================================================================
#  MAIN GRAPH: The Academic Orchestrator
# ============================================================================
class AcademicAgent:
    """Dispatches questions to simplified workers and aggregates final document."""
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.notifier = Notifier(project_id)
        self.request_session = get_request_session_with_blob_auth()

    def fetch_and_dispatch(self, state: AcademicAgentState) -> Union[List[Send], dict]:
        self.notifier.send_message("Academic Agent: Extracting questions and initiating workers...")
        
        urls = state.get("extracted_questions_blob_url", [])
        if not urls:
            raise ValueError("No extracted questions URL found.")

        try:
            res = self.request_session.get(urls[0], timeout=20)
            res.raise_for_status()
            questions = self._extract_universal_format(res.text)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch questions: {e}")

        if not questions:
            raise ValueError("No viable questions parsed from the document.")

        return [Send("process_single_question", {
            "project_id": self.project_id,
            "original_question": q
        }) for q in questions]

    def aggregate_and_upload(self, state: AcademicAgentState) -> dict:
        self.notifier.send_message("Academic Agent: Stitching final study guide...")
        
        answers = state.get("completed_answers", [])
        failed_questions = state.get("failed_questions", [])
        
        final_markdown = "# Academic Q&A Final Document\n\n" + "\n\n".join(answers)
        
        # Optionally append a list of questions that catastrophically failed
        if failed_questions:
            final_markdown += "\n\n## Unanswered Questions\n" + "\n".join([f"- {fq}" for fq in failed_questions])

        blob_path = f"{get_collection_name(self.project_id)}/output/final_academic_answers.md"
        final_url = upload_to_vercel_blob(blob_path, final_markdown, "text/markdown")
        
        self.notifier.send_message(f"✅ Academic Agent Complete. {len(answers)} answers processed.")
        return {"final_answers_blob_url": [final_url]}

    def _extract_universal_format(self, raw_text: str) -> List[str]:
        text = re.sub(r'(?m)^## Page \d+\s*(\(Vision Extracted\))?.*?$', '', raw_text)
        text = re.sub(r'(?m)^---\s*$', '', text)
        text = re.sub(r'(?m)^# .*?$', '', text) 
        text = re.sub(r'\n{3,}', '\n\n', text) 
        text = text.strip()

        primary_pattern = r'\n(?=\s*(?:' \
                          r'\d+[\.\)\-]' \
                          r'|Q(?:uestion)?\s*\d*[\.\:\)]?' \
                          r'|[IVXLCDM]+\.' \
                          r')\s)'
        
        raw_blocks = re.split(primary_pattern, text)
        questions = []
        for block in raw_blocks:
            cleaned = block.strip()
            if len(cleaned) < 15: continue
            if len(cleaned) > 1500:
                for p in re.split(r'\n\n+', cleaned):
                    if len(p.strip()) > 15: questions.append(p.strip())
            else:
                questions.append(cleaned)
        return questions
    
    def build_graph(self):
        graph = StateGraph(AcademicAgentState)
        graph.add_node("process_single_question", self._process_single_question_wrapper)
        graph.add_node("aggregate", self.aggregate_and_upload)

        graph.add_conditional_edges(START, self.fetch_and_dispatch, ["process_single_question"])
        graph.add_edge("process_single_question", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()
    
    # ================================================================
    # Helper methods 
    # ================================================================
    def _process_single_question_wrapper(self, state: dict):
        worker_subgraph = build_question_worker()
        # Run the parallel worker
        result = worker_subgraph.invoke(state)
        
        # ONLY return the fields the Parent State is allowed to aggregate!
        # This completely hides 'project_id', 'plan', 'draft', etc. from the parent.
        return {
            "completed_answers": result.get("completed_answers", []),
            "failed_questions": result.get("failed_questions", [])
        }

# ================================================================
# Builder functions
# ================================================================
def build_academic_agent(project_id: str):
    return AcademicAgent(project_id=project_id).build_graph()

def build_question_worker():
    return QuestionWorkerGraph().build()