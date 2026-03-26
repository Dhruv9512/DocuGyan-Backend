import re
import time
import logging
import concurrent.futures
from pydantic import BaseModel, Field


# Import your Utilities!
from DocuAgent.utils.llm_calls import DocuAgentLLMCalls
from DocuAgent.utils.extraction import build_DocuExtractor 
from DocuAgent.utils.utility import upload_to_vercel_blob

from DocuAgent.schemas.llm_schemas import RefinedBatch

logger = logging.getLogger(__name__)



# ==========================================
# Main Class
# ==========================================
class QuestionRefiner:
    def __init__(self, project_id: str, file_url: str):
        self.project_id = project_id
        self.file_url = file_url
        self.extracted_md_url = None
        self.refined_md_url = None
        
    def build(self) -> dict:
        logger.info(f"Building QuestionRefiner pipeline for project {self.project_id}...")
        self._extract()
        self._refine()
        return {
            "project_id": self.project_id,
            "raw_questions_blob_url": self.extracted_md_url,
            "refined_questions_blob_url": self.refined_md_url,
        }

    def _extract(self) -> str:
        logger.info("Triggering Extraction Sub-process...")
        self.extracted_md_url = build_DocuExtractor(self.project_id, self.file_url)
        return self.extracted_md_url

    def _refine(self):
        import requests
        logger.info("Starting format-based refinement phase...")

        response = requests.get(self.extracted_md_url, timeout=30)
        response.raise_for_status()
        raw_text = response.text

        raw_questions = self._extract_universal_format(raw_text)
        logger.info(f"Successfully parsed {len(raw_questions)} raw question blocks.")

        batch_size = 30
        batches = [raw_questions[i:i + batch_size] for i in range(0, len(raw_questions), batch_size)]
        logger.info(f"Split into {len(batches)} exact batches.")

        all_refined_questions = []
        
        # PROTECT GROQ FREE TIER: Max workers reduced to 2 to prevent hitting the 30k TPM limit instantly
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(self._process_batch, batches)
            for batch_result in results:
                if batch_result and batch_result.questions:
                    all_refined_questions.extend(batch_result.questions)

        final_markdown = f"# Refined Questions for Project: {self.project_id}\n\n"
        for idx, rq in enumerate(all_refined_questions, 1):
            final_markdown += f"{idx}. {rq.refined_question}\n"

        blob_path = f"{self.project_id}/output/refined_questions.md"
        self.refined_md_url = upload_to_vercel_blob(
            blob_path=blob_path, 
            content=final_markdown,
            content_type="text/markdown"
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
                return RefinedBatch(questions=[])
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return RefinedBatch(questions=[])
        

# Builder function to run QuestionRefiner in one step
def build_QuestionRefiner(project_id: str, file_url: str) -> dict:
    """
    Factory function to build and execute the QuestionRefiner pipeline.
    and return the final URLs for both raw and refined questions.

    args:
        project_id: Unique identifier for the project.
        file_url: URL of the original document to process.
    """
    refiner = QuestionRefiner(project_id, file_url)
    return refiner.build()