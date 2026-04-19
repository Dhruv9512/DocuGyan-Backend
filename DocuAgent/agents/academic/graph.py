import base64
import concurrent.futures
import json
import logging
import re
import requests
from django.conf import settings

# Validation imports
from typing import Any, List, Tuple, Union, Dict
from langchain_core.documents import Document

# Import the Graph state
from DocuAgent.schemas.agent_schemas import (
    AcademicAgentState,
    QuestionState,
)

# Import Schema state
from DocuAgent.schemas.llm_schemas import (
    PlannerOutput
)

# Import the helper utility functions
from DocuAgent.utils.utility import (
    get_request_session_with_blob_auth,
    upload_to_vercel_blob,
    get_collection_name
)

# Import CRAG utility
from DocuAgent.agents.academic.tools.CRAG import build_CorrectiveRetriever

# Import LLM calls
from DocuAgent.utils.llm_calls import DocuAgentLLMCalls

# Import Graph framework
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Import websocket notifier to send real-time updates to FE
from DocuAgent.websocket.notifier import Notifier

# Import vector DB ingestor for final step
from DocuAgent.ingestion.VectorDB_Ingestor import build_vector_db_ingestor

logging.basicConfig(level=logging.INFO)

log_format = '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
logger = logging.getLogger("AcademicAgent")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("AcademicAgent.log")
handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(handler)


# ── Module-level constants ────────────────────────────────────────────────────
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "cross-site",
}

ALLOWED_MIME_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif',
    'image/webp', 'image/svg+xml', 'image/bmp', 'image/tiff',
}


# ===========================================================================
# QuestionWorkerGraph
# ===========================================================================

class QuestionWorkerGraph:
    """Processes a single question concurrently using deterministic planning."""

    # ── Node 1: Retrieve ─────────────────────────────────────────────────────
    def _retrieve_context(self, state: QuestionState) -> dict:
        try:
            result = build_CorrectiveRetriever(
                project_id=state["project_id"],
                search_queries=state["original_question"],
            )

            references_seen = set()
            for doc in result.get("retrieved_docs", []):
                source_url = doc.metadata.get("source_url", "")
                if source_url and source_url not in references_seen:
                    references_seen.add(source_url)

            return {
                "retrieved_docs": result.get("retrieved_docs", []),
                "grader_assessment": result.get("grader_assessment"),
                "retrieved_references": list(references_seen),
            }
        except Exception as exc:
            logger.error(f"[Retrieve] CorrectiveRetriever failed: {exc}")
            return {
                "retrieved_docs": [],
                "retrieved_references": [],
                "grader_assessment": None,
            }

    # ── Node 2: Plan ─────────────────────────────────────────────────────────
    def _plan_execution(self, state: QuestionState) -> dict:
        question = state.get("original_question", "No question provided.")
        KB = self._format_retrieved_context(state.get("retrieved_docs", []))
        context  = [v.get("page_content", "") for v in KB]

        try:
            plan = DocuAgentLLMCalls.call_question_planner(question, context)
        except Exception as exc:
            logger.warning(f"[Plan] LLM failed, using fallback for: {question} -> {exc}")
            plan = PlannerOutput(
                question=question,         
                question_category="academic",
                allocated_marks=5,
                requires_code=False,
                requires_diagram=False,
                is_comparison=False,
                core_entities=[],
            )
        return {"plan": plan}

    # ── Helper: Format retrieved docs ─────────────────────────────────────────
    def _format_retrieved_context(self, retrieved_docs: List[Document]) -> List[Dict[str, Any]]:
        """Formats LangChain documents into dicts compatible with build_drafter_user_prompt."""
        if not retrieved_docs:
            return []

        return [
            {
                "source_url": doc.metadata.get("source_url", ""),
                "page_number": doc.metadata.get("page_number", ""),
                "page_content": doc.page_content.strip() if doc.page_content else "",
            }
            for doc in retrieved_docs
        ]

    # ── Node 3: Draft ─────────────────────────────────────────────────────────
    def _draft_answer(self, state: QuestionState) -> dict:
        plan: PlannerOutput      = state.get("plan")
        context: List[Dict[str, Any]] = self._format_retrieved_context(
            state.get("retrieved_docs", [])
        )

        try:
            draft = DocuAgentLLMCalls.call_answer_drafter(
                question=state["original_question"],
                context_chunks=context,
                plan=plan,
            )
            if not draft or not str(draft).strip():
                raise ValueError("LLM returned an empty response.")
        except Exception as exc:
            logger.error(
                f"[Draft] Failed for '{state['original_question'][:30]}...' | error={exc}"
            )
            draft = (
                "*System Note: Unable to generate a reliable draft for this question "
                "due to an internal processing error.*"
            )

        return {"draft_answer": draft}

    # ── Conditional Routing ───────────────────────────────────────────────────
    def should_fetch_diagrams(self, state: QuestionState) -> str:
        plan: PlannerOutput = state.get("plan")
        if plan and plan.requires_diagram:
            return "add_diagram_query"
        return "format"

    # ── Node 4a: Add Diagram Query (if needed) ────────────────────────────────
    def _add_diagram_query(self, state: QuestionState) -> dict:
        drafter = state.get("draft_answer", "")
        question = state.get("original_question", "Unknown Question")

        try:
            new_drafter = DocuAgentLLMCalls.call_diagram_query_generator(question, drafter)
            if new_drafter:
                return {"draft_answer": new_drafter}
            logger.warning(
                f"[AddDiagramQuery] LLM returned empty response for: '{question[:60]}...'"
            )
            return {"draft_answer": drafter}

        except Exception as exc:
            logger.error(f"[AddDiagramQuery] Failed: {exc}")
            return {"draft_answer": drafter}      # ← safe fallback, never returns None

    # ── Node 4b: Get Diagrams ─────────────────────────────────────────────────
    def _get_diagrams(self, state: QuestionState) -> dict:
        """
        Scans the draft for {diagram_N} placeholders, extracts [Query: ...] hints,
        then fetches images in parallel.
        """
        draft = state.get("draft_answer", "")
        plan = state.get("plan")
        original_question = state.get("original_question", "Unknown Question")

        placeholders = list(dict.fromkeys(re.findall(r"\{diagram_\d+\}", draft)))
        if not placeholders:
            return {"diagram_mapping": {}}

        def _extract_hint(placeholder: str) -> str:
            pattern = rf'\[Query:\s*(.*?)\]\s*{re.escape(placeholder)}'
            match = re.search(pattern, draft)
            if match:
                return match.group(1).strip()
            if plan and plan.core_entities:
                return f"{plan.core_entities[0]} diagram"
            return f"{original_question[:80]} diagram"

        primary_concept = (
            plan.core_entities[0]
            if plan and plan.core_entities
            else original_question[:80]
        )

        tasks = {ph: (primary_concept, _extract_hint(ph)) for ph in placeholders}

        diagram_mapping: Dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    self._generate_one,
                    placeholder=ph,
                    concept=concept,
                    extracted_query=hint,
                ): ph
                for ph, (concept, hint) in tasks.items()
            }
            for future in concurrent.futures.as_completed(futures):
                ph = futures[future]
                try:
                    placeholder, rendered = future.result()
                    diagram_mapping[placeholder] = rendered
                except Exception as exc:
                    logger.error(f"[GetDiagrams] Future failure for {ph}: {exc}")
                    diagram_mapping[ph] = "*[Diagram unavailable]*"

        logger.info(
            f"[GetDiagrams] Resolved {len(diagram_mapping)}/{len(placeholders)} diagrams."
        )
        return {"diagram_mapping": diagram_mapping}

    # ── Node 5: Format ────────────────────────────────────────────────────────
    def _format_final(self, state: QuestionState) -> dict:
        draft = state.get("draft_answer", "")
        diagram_mapping = state.get("diagram_mapping", {})
        original_question = state.get("original_question", "Unknown Question")

        clean_question = re.sub(r'^\d+[\.\)\-]?\s*', '', original_question)

        if draft.startswith("*System Note"):
            return {"completed_answers": [f"### {clean_question}\n\n{draft}\n\n---"]}

        # Inject diagrams
        for placeholder, markdown_image in diagram_mapping.items():
            draft = draft.replace(placeholder, markdown_image)

        # Clean up any leftover placeholders / query hints
        draft = re.sub(r"\{diagram_\d+\}", "", draft)
        draft = re.sub(r"\[Query:.*?\]", "", draft)

        # References section
        retrieved_references = state.get("retrieved_references", [])
        if retrieved_references:
            draft += "\n\n---\n\n### References\n"
            draft += "\n".join(f"- {url}" for url in retrieved_references)

        return {"completed_answers": [f"### {clean_question}\n\n{draft.strip()}\n\n---"]}

    # ── Image Pipeline ────────────────────────────────────────────────────────
    def _generate_one(
        self, placeholder: str, concept: str, extracted_query: str
    ) -> Tuple[str, str]:
        """
        Image search pipeline: Serper.dev → SerpApi → Google Custom Search.
        Downloads the best result and returns it as a base64 data URI.
        """
        query = extracted_query if len(extracted_query) > 3 else f"{concept} architecture diagram"
        logger.info(f"[GetDiagrams] Image search query: '{query}'")

        img_url = None
        caption = None

        # PRIMARY: Serper.dev
        if not img_url:
            try:
                serper_key = getattr(settings, 'SERPER_API_KEY', None)
                if serper_key:
                    logger.debug("[GetDiagrams] Trying Serper.dev...")
                    res = requests.post(
                        "https://google.serper.dev/images",
                        headers={
                            'X-API-KEY': serper_key,
                            'Content-Type': 'application/json',
                        },
                        data=json.dumps({"q": query}),
                        timeout=10,
                    )
                    res.raise_for_status()
                    images = res.json().get('images', [])
                    if images:
                        img_url, caption = self._pick_best_url(
                            images, 'imageUrl', 'title', concept
                        )
            except Exception as e:
                logger.warning(f"[GetDiagrams] Serper.dev failed: {e}")

        # FALLBACK 1: SerpApi
        if not img_url:
            try:
                serpapi_key = getattr(settings, 'SERPAPI_API_KEY', None)
                if serpapi_key:
                    logger.debug("[GetDiagrams] Trying SerpApi...")
                    res = requests.get(
                        "https://serpapi.com/search",
                        params={
                            "engine": "google_images",
                            "q": query,
                            "api_key": serpapi_key,
                        },
                        timeout=10,
                    )
                    res.raise_for_status()
                    images = res.json().get('images_results', [])
                    if images:
                        img_url, caption = self._pick_best_url(
                            images, 'original', 'title', concept
                        )
            except Exception as e:
                logger.warning(f"[GetDiagrams] SerpApi failed: {e}")

        # FALLBACK 2: Google Custom Search
        if not img_url:
            try:
                google_key = getattr(settings, 'GOOGLE_API_KEY', None)
                google_cx = getattr(settings, 'GOOGLE_CX', None)
                if google_key and google_cx:
                    logger.debug("[GetDiagrams] Trying Google Custom Search...")
                    res = requests.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params={
                            "q": query,
                            "searchType": "image",
                            "key": google_key,
                            "cx": google_cx,
                            "num": 3,
                        },
                        timeout=10,
                    )
                    res.raise_for_status()
                    images = res.json().get('items', [])
                    if images:
                        img_url, caption = self._pick_best_url(
                            images, 'link', 'title', concept
                        )
            except Exception as e:
                logger.warning(f"[GetDiagrams] Google Custom Search failed: {e}")

        if not img_url:
            logger.error(f"[GetDiagrams] All providers failed for: '{query}'")
            return placeholder, f"*[Diagram of {concept[:40]} unavailable]*"

        # Download + Base64 encode
        try:
            logger.info(f"[GetDiagrams] Downloading from: {img_url}")
            img_res = requests.get(img_url, headers=DOWNLOAD_HEADERS, timeout=15)
            img_res.raise_for_status()

            mime_type = (
                img_res.headers.get('Content-Type', 'image/png')
                .split(';')[0].strip().lower()
            )
            if mime_type not in ALLOWED_MIME_TYPES:
                logger.warning(
                    f"[GetDiagrams] Unexpected MIME '{mime_type}', defaulting to image/png"
                )
                mime_type = 'image/png'

            if not caption:
                caption = f"Diagram illustrating {concept}"

            data_uri = (
                f"data:{mime_type};base64,"
                f"{base64.b64encode(img_res.content).decode('utf-8')}"
            )
            rendered = f"![{caption}]({data_uri})\n*{caption}*"

            logger.info(f"[GetDiagrams] Encoded {mime_type} image for '{placeholder}'")
            return placeholder, rendered

        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"[GetDiagrams] HTTP {e.response.status_code} "
                f"for '{placeholder}': {img_url}"
            )
        except requests.exceptions.Timeout:
            logger.warning(f"[GetDiagrams] Timeout for '{placeholder}': {img_url}")
        except Exception as e:
            logger.warning(f"[GetDiagrams] Download failed for '{placeholder}': {e}")

        return placeholder, f"*[Diagram of {concept[:40]} unavailable]*"

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _pick_best_url(
        self,
        candidates: list,
        url_key: str,
        title_key: str,
        concept: str,
    ) -> Tuple[str | None, str | None]:
        """Try top-3 candidates; return first accessible URL, or first URL as fallback."""
        first_url, first_caption = None, None
        for item in candidates[:3]:
            url = item.get(url_key)
            caption = item.get(title_key, f"Diagram illustrating {concept}")
            if not url:
                continue
            if first_url is None:
                first_url, first_caption = url, caption
            if self._is_url_accessible(url):
                return url, caption
        return first_url, first_caption

    def _is_url_accessible(self, url: str) -> bool:
        """HEAD check — fast way to detect 403/404 before a full download."""
        try:
            resp = requests.head(
                url, headers=DOWNLOAD_HEADERS, timeout=5, allow_redirects=True
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── Graph Builder ─────────────────────────────────────────────────────────
    def build(self):
        wf = StateGraph(QuestionState)

        wf.add_node("retrieve", self._retrieve_context)
        wf.add_node("plan", self._plan_execution)
        wf.add_node("draft", self._draft_answer)
        wf.add_node("add_diagram_query", self._add_diagram_query)
        wf.add_node("get_diagrams", self._get_diagrams)
        wf.add_node("format", self._format_final)

        wf.add_edge(START,"retrieve")
        wf.add_edge("retrieve", "plan")
        wf.add_edge("plan", "draft")

        wf.add_conditional_edges(
            "draft",
            self.should_fetch_diagrams,
            {"add_diagram_query": "add_diagram_query", "format": "format"},
        )

        wf.add_edge("add_diagram_query", "get_diagrams")
        wf.add_edge("get_diagrams", "format")
        wf.add_edge("format", END)

        return wf.compile()


# ===========================================================================
# AcademicAgent
# ===========================================================================

class AcademicAgent:
    """Dispatches questions to worker graphs and aggregates the final document."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.notifier = Notifier(project_id)
        self.request_session = get_request_session_with_blob_auth()

    def fetch_and_dispatch(
        self, state: AcademicAgentState
    ) -> Union[List[Send], dict]:
        self.notifier.send_message(
            "Academic Agent: Extracting questions and initiating workers..."
        )

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

        return [
            Send("process_single_question", {
                "project_id":        self.project_id,
                "original_question": q,
            })
            for q in questions
        ]

    def aggregate_and_upload(self, state: AcademicAgentState) -> dict:
        self.notifier.send_message("Academic Agent: Stitching final study guide...")

        answers = state.get("completed_answers", [])
        failed_questions  = state.get("failed_questions",  [])

        final_markdown = "# Academic Q&A Final Document\n\n" + "\n\n".join(answers)

        if failed_questions:
            final_markdown += "\n\n## Unanswered Questions\n" + "\n".join(
                f"- {fq}" for fq in failed_questions
            )

        blob_path = (
            f"{get_collection_name(self.project_id)}/output/final_academic_answers.md"
        )
        final_url = upload_to_vercel_blob(blob_path, final_markdown, "text/markdown")

        self.notifier.send_message(
            f"Academic Agent Complete. {len(answers)} answers processed."
        )

        try:
            success = build_vector_db_ingestor(
                project_id=self.project_id,
                extracted_doc_urls=[final_url],
                is_final_answer=True,
            )
            if success:
                self.notifier.send_message("Final Q&A successfully indexed in Milvus.")
            else:
                self.notifier.send_error("Vector ingestion process returned False.")
        except Exception as e:
            logger.error(f"Failed to ingest Q&A: {e}")
            self.notifier.send_error(f"Failed to index Q&A: {e}")

        return {"final_answers_blob_url": [final_url]}

    def _extract_universal_format(self, raw_text: str) -> List[str]:
        text = re.sub(r'(?m)^## Page \d+\s*(\(Vision Extracted\))?.*?$', '', raw_text)
        text = re.sub(r'(?m)^---\s*$', '', text)
        text = re.sub(r'(?m)^# .*?$', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        primary_pattern = (
            r'\n(?=\s*(?:'
            r'\d+[\.\)\-]'
            r'|Q(?:uestion)?\s*\d*[\.\:\)]?'
            r'|[IVXLCDM]+\.'
            r')\s)'
        )
        raw_blocks = re.split(primary_pattern, text)

        questions = []
        for block in raw_blocks:
            cleaned = block.strip()
            if len(cleaned) < 15:
                continue
            if len(cleaned) > 1500:
                for p in re.split(r'\n\n+', cleaned):
                    if len(p.strip()) > 15:
                        questions.append(p.strip())
            else:
                questions.append(cleaned)

        return questions

    def build_graph(self):
        graph = StateGraph(AcademicAgentState)
        graph.add_node("process_single_question", self._process_single_question_wrapper)
        graph.add_node("aggregate", self.aggregate_and_upload)

        graph.add_conditional_edges(
            START, self.fetch_and_dispatch, ["process_single_question"]
        )
        graph.add_edge("process_single_question", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    def _process_single_question_wrapper(self, state: dict) -> dict:
        """
        Runs QuestionWorkerGraph for one question.
        On ANY failure: logs the error, records the question as failed,
        and returns gracefully so the parent graph keeps running.
        """
        question = state.get("original_question", "Unknown")
        try:
            worker_subgraph = build_question_worker()
            result = worker_subgraph.invoke(state)
            return {
                "completed_answers": result.get("completed_answers", []),
                "failed_questions": result.get("failed_questions",  []),
            }
        except Exception as exc:
            logger.error(
                f"[Worker] QuestionWorkerGraph FAILED for "
                f"'{question[:60]}...' | error={exc}",
                exc_info=True,
            )
            return {
                "completed_answers": [],
                "failed_questions": [question],
            }


# ===========================================================================
# Builder functions
# ===========================================================================

def build_academic_agent(project_id: str):
    return AcademicAgent(project_id=project_id).build_graph()

def build_question_worker():
    return QuestionWorkerGraph().build()