import logging
import re


# Validation imports
from typing import List, Union


# Import the Graph state
from DocuAgent.schemas.agent_schemas import (
    AcademicAgentState,
)


# Import the helper utility functions
from DocuAgent.utils.utility import (
    get_request_session_with_blob_auth,
    upload_to_vercel_blob,
    get_collection_name
)

# Import Graph framework
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Import websocket notifier to send real-time updates to FE
from DocuAgent.websocket.notifier import Notifier

# Import vector DB ingestor for final step
from DocuAgent.ingestion.VectorDB_Ingestor import build_vector_db_ingestor

from DocuAgent.agents.academic.question_worker import build_question_worker

logging.basicConfig(level=logging.INFO)

log_format = '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
logger = logging.getLogger("AcademicAgent")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("AcademicAgent.log")
handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(handler)


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
            "Academic Agent: Extracting questions and initiating workers...",
            current_node="academic",
            status="processing",
        )

        urls = state.get("extracted_questions_blob_url", [])
        if not urls:
            self.notifier.send_error(
                "Academic Agent: No extracted questions URL found.",
                current_node="academic",
            )
            raise ValueError("No extracted questions URL found.")

        try:
            res = self.request_session.get(urls[0], timeout=20)
            res.raise_for_status()
            questions = self._extract_universal_format(res.text)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch questions: {e}")

        if not questions:
            self.notifier.send_error(
                "Academic Agent: No viable questions parsed from the document.",
                current_node="academic",
            )
            raise ValueError("No viable questions parsed from the document.")

        self.notifier.send_message(
            f"Academic Agent: Parsed {len(questions)} question(s). Dispatching workers.",
            current_node="academic",
            status="processing",
        )

        return [
            Send("process_single_question", {
                "project_id":        self.project_id,
                "original_question": q,
            })
            for q in questions
        ]

    def aggregate_and_upload(self, state: AcademicAgentState) -> dict:
        self.notifier.send_message(
            "Academic Agent: Stitching final study guide...",
            current_node="academic",
            status="processing",
        )

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
            f"Academic Agent Complete. {len(answers)} answers processed.",
            current_node="academic",
            status="completed",
        )

        try:
            success = build_vector_db_ingestor(
                project_id=self.project_id,
                extracted_doc_urls=[final_url],
                is_final_answer=True,
            )
            if success:
                self.notifier.send_message(
                    "Final Q&A successfully indexed in Milvus.",
                    current_node="academic",
                    status="completed",
                )
            else:
                self.notifier.send_error(
                    "Vector ingestion process returned False.",
                    current_node="academic",
                )
        except Exception as e:
            logger.error(f"Failed to ingest Q&A: {e}")
            self.notifier.send_error(f"Failed to index Q&A: {e}", current_node="academic")

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
            self.notifier.send_error(
                f"Academic Agent: Worker failed for question '{question[:60]}...': {exc}",
                current_node="academic",
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
