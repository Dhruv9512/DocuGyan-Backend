import re
import time
import logging
import concurrent.futures
import requests


# Import your Utilities!
from DocuAgent.utils.llm_calls import DocuAgentLLMCalls
from DocuAgent.utils.extraction import build_DocuExtractor 
from DocuAgent.utils.utility import upload_to_vercel_blob, get_collection_name, get_request_session_with_blob_auth, sanitize_blob_filename

from DocuAgent.schemas.llm_schemas import RefinedBatch
from DocuAgent.websocket.notifier import Notifier

logger = logging.getLogger(__name__)



# ==========================================
# Main Class
# ==========================================
class QuestionRefiner:
    def __init__(self, project_id: str, file_url: str):
        self.project_id = project_id
        self.file_url = file_url
        self.session = get_request_session_with_blob_auth()
        self.extracted_md_url = None
        self.refined_md_url = None
        self.blob_collection = get_collection_name(self.project_id)
        self.notifier = Notifier(project_id)
        
    def run(self) -> dict:
        logger.info(f"Building QuestionRefiner pipeline for project {self.project_id}...")
        self.notifier.send_message(
            f"Question Extractor: Starting question refinement for project {self.project_id}...",
            current_node="extractor",
            status="processing",
        )
        self._extract()
        self.notifier.send_message(
            "Question Extractor: Extraction complete. Starting refinement phase...",
            current_node="extractor",
            status="processing",
        )
        
        self.notifier.send_message(
            f"Question Refining: Starting refinement for project {self.project_id}...",
            current_node="extractor",
            status="processing",
        )
        self._refine()
        self.notifier.send_message(
            "Question Refining: Refinement phase complete.",
            current_node="extractor",
            status="completed",
        )
        return {
            "project_id": self.project_id,
            "raw_questions_blob_url": self.extracted_md_url,
            "extracted_questions_blob_url": self.refined_md_url,
        }

    def _extract(self) -> str:
        logger.info("Triggering Extraction Sub-process...")
        self.extracted_md_url = build_DocuExtractor(self.project_id, self.file_url)
        return self.extracted_md_url

    def _refine(self):
        logger.info("Starting format-based refinement phase...")

        # 1. Safely handle the download
        try:
            response = self.session.get(self.extracted_md_url, timeout=30)
            response.raise_for_status()
            raw_text = response.text
        except requests.exceptions.RequestException as e:
            self.notifier.send_error(
                f"Question Refining: Failed to fetch extracted markdown: {str(e)}",
                current_node="extractor",
            )
            logger.error(f"Failed to download raw markdown from {self.extracted_md_url}")
            raise RuntimeError(f"Failed to fetch extracted text for refinement: {str(e)}") from e

        # 2. Extract and Batch
        raw_questions = self._extract_universal_format(raw_text)
        logger.info(f"Successfully parsed {len(raw_questions)} raw question blocks.")

        if not raw_questions:
            raise ValueError("No questions were found in the extracted text to refine.")

        batch_size = 30
        batches = [raw_questions[i:i + batch_size] for i in range(0, len(raw_questions), batch_size)]
        logger.info(f"Split into {len(batches)} exact batches.")
        self.notifier.send_message(
            f"Question Refining: Parsed {len(raw_questions)} question block(s) into {len(batches)} batch(es).",
            current_node="extractor",
            status="processing",
        )

        all_refined_questions = []
        failed_batches = 0
        completed_batches = 0
        total_batches = len(batches)
        
        # 3. PROTECT GROQ FREE TIER: Max workers reduced to 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Create a dictionary of futures so we can track them independently
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }

            # Process them as they complete (out of order is fine, we just want the data)
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    # If the worker thread permanently failed, the exception is raised HERE
                    batch_result = future.result()
                    if batch_result and batch_result.questions:
                        all_refined_questions.extend(batch_result.questions)
                    completed_batches += 1
                    self.notifier.send_message(
                        f"Question Refining: Completed batch {completed_batches}/{total_batches}.",
                        current_node="extractor",
                        status="processing",
                    )
                except Exception as e:
                    self.notifier.send_error(
                        f"Question Refining: A batch failed to refine: {str(e)}",
                        current_node="extractor",
                    )
                    # Catch the failure so the OTHER threads keep running
                    logger.error(f"A batch permanently failed after retries: {str(e)}")
                    failed_batches += 1

        # 4. Post-processing Safety Checks
        if failed_batches == len(batches):
             # Total failure state
             raise RuntimeError("Critical Failure: All LLM refinement batches failed.")
        elif failed_batches > 0:
             # Partial success state
             logger.warning(f"Refinement finished with {failed_batches} failed batches. Saving partial data ({len(all_refined_questions)} questions).")
             self.notifier.send_message(
                 f"Question Refining: Continuing with partial success ({failed_batches} failed batch(es)).",
                 current_node="extractor",
                 status="processing",
             )

        if not all_refined_questions:
             raise ValueError("Refinement completed but zero valid questions were generated.")

        # 5. Format and Upload
        final_markdown = f"# Refined Questions for Project: {self.project_id}\n\n"
        for idx, rq in enumerate(all_refined_questions, 1):
            final_markdown += f"{idx}. {rq.refined_question}\n"

        blob_path = f"{self.blob_collection}/output/refined_questions.md"
        self.refined_md_url = upload_to_vercel_blob(
            blob_path=blob_path, 
            content=final_markdown,
            content_type="text/markdown"
        )
        self.notifier.send_message(
            f"Question Refining: Uploaded refined questions to {self.refined_md_url}",
            current_node="extractor",
            status="completed",
        )
        logger.info(f"Refinement complete. Refined Markdown URL: {self.refined_md_url}")

    # ==========================================
    # Industry-Level Adaptive Parser
    # ==========================================
    def _extract_universal_format(self, raw_text: str) -> list[str]:
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
            if len(cleaned) < 15:
                continue
            if len(cleaned) > 1500:
                paragraphs = re.split(r'\n\n+', cleaned)
                for p in paragraphs:
                    if len(p.strip()) > 15:
                        questions.append(p.strip())
            else:
                questions.append(cleaned)
                
        return questions

    # ==========================================
    # LLM Batch Worker
    # ==========================================
    def _process_batch(self, batch: list[str]) -> RefinedBatch:
        """
        Processes a batch of raw questions and returns a structured RefinedBatch.
        """
        # Format the list into a single string for the LLM prompt
        batch_text = "\n\n".join(f"- {q}" for q in batch)
        
        try:
            refined = DocuAgentLLMCalls.call_refine_questions(batch_text)
            if refined and refined.questions:
                logger.info(f"Batch of {len(batch)} questions refined successfully.")
                return refined
            else:
                logger.warning("LLM returned no refined questions for the batch.")
                raise ValueError("LLM returned no refined questions.")
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise RuntimeError(f"Failed to process batch: {str(e)}") from e

# Builder function to run QuestionRefiner in one step
def build_QuestionRefiner(project_id: str, file_url: str) -> dict:
    """
    Factory function to build and execute the QuestionRefiner pipeline.
    and return the final URLs for both raw and refined questions.

    args:
        project_id: Unique identifier for the project.
        file_url: URL of the original document to process.
    """
    refiner = None
    try:
        refiner = QuestionRefiner(project_id, file_url)
        return refiner.run()
    except Exception as e:
        notifier = refiner.notifier if refiner else Notifier(project_id)
        notifier.send_error(
            f"Failed to build or run QuestionRefiner: {str(e)}",
            current_node="extractor",
        )
        logger.error(f"Failed to build or run QuestionRefiner: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to build or run QuestionRefiner: {str(e)}") from e