import re
import logging
import requests
import concurrent.futures
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

# Import your factory function
from .extraction import build_DocuExtractor 

logger = logging.getLogger(__name__)

# ==========================================
# Pydantic Schemas
# ==========================================
class RefinedQuestion(BaseModel):
    refined_question: str = Field(description="The improved, clear, and professional version of the extracted question.")

class RefinedBatch(BaseModel):
    questions: list[RefinedQuestion]

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
        self.extracted_md_url = build_DocuExtractor(self.project_id, self.file_url)
        return self.extracted_md_url

    def _refine(self):
        logger.info("Starting format-based refinement phase...")

        # 1. Fetch the universally formatted markdown content
        response = requests.get(self.extracted_md_url, timeout=30)
        response.raise_for_status()
        raw_text = response.text

        # 2. Extract strictly using the Adaptive Parser
        raw_questions = self._extract_universal_format(raw_text)
        logger.info(f"Successfully parsed {len(raw_questions)} raw question blocks.")

        # 3. Create exact batches of 30 questions
        batch_size = 30
        batches = [raw_questions[i:i + batch_size] for i in range(0, len(raw_questions), batch_size)]
        logger.info(f"Split into {len(batches)} exact batches of up to {batch_size} questions.")

        # 4. Process batches concurrently (Protects your 500MB server RAM)
        all_refined_questions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(self._process_batch, batches)
            for batch_result in results:
                if batch_result and batch_result.questions:
                    all_refined_questions.extend(batch_result.questions)

        # 5. Format the final Markdown output cleanly
        final_markdown = ""
        for idx, rq in enumerate(all_refined_questions, 1):
            final_markdown += f"{idx}. {rq.refined_question}\n"

        # 6. Upload final Markdown to Blob Storage
        self.refined_md_url = self._upload_refined_to_blob(final_markdown)
        logger.info(f"Refinement complete. Refined Markdown URL: {self.refined_md_url}")

    # ==========================================
    # Industry-Level Adaptive Parser
    # ==========================================
    def _extract_universal_format(self, raw_text: str) -> list[str]:
        """
        Isolates questions by detecting Primary question markers ONLY.
        Keeps sub-questions (a, b, i, ii) and multiple choice options attached to their parent.
        """
        # Step A: Sanitize the DocuGyan structural tags
        text = re.sub(r'(?m)^## Page \d+\s*$', '', raw_text)
        text = re.sub(r'(?m)^---\s*$', '', text)
        text = re.sub(r'(?m)^# .*?$', '', text) 
        text = text.strip()

        # Step B: The "Parent-Only" Splitter
        # Matches a newline, immediately followed by PRIMARY markers only:
        # 1. Numbers (1., 1), 1-)
        # 2. Q/Question prefixes (Q1, Question 2:, Q. 3)
        # 3. Capital Roman Numerals (I., II., III.)
        # NOTICE: Lowercase letters and bullet points are removed from the split criteria!
        primary_pattern = r'\n(?=\s*(?:' \
                          r'\d+[\.\)\-]' \
                          r'|Q(?:uestion)?\s*\d*[\.\:\)]?' \
                          r'|[IVXLCDM]+\.' \
                          r')\s)'
        
        raw_blocks = re.split(primary_pattern, text)
        
        # Step C: Validation and Fallback
        questions = []
        for block in raw_blocks:
            cleaned = block.strip()
            
            if len(cleaned) < 15:
                continue
                
            # FALLBACK: If a document uses bullets for main questions, it won't split above.
            # It will end up here as a massive block. We dynamically slice it by paragraphs 
            # to protect the LLM and your 500MB server RAM.
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
        structured_llm = self.llm.with_structured_output(RefinedBatch)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert educational AI. Read the following list of raw extracted text blocks. "
             "Your job is to identify the questions, extract them, and refine them into clear, professional, "
             "and grammatically correct questions. "
             "Ignore pure instructional text (e.g., 'Answer the following'). "
             "Do not include answers or multiple-choice options."
            ),
            ("user", "Raw text blocks:\n\n{questions}")
        ])
        
        batch_text = "\n\n".join(f"- {q}" for q in batch)
        chain = prompt | structured_llm
        
        try:
            return chain.invoke({"questions": batch_text})
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            return None

    def _upload_refined_to_blob(self, content: str) -> str:
        # TODO: Implement Vercel Blob PUT request
        return "https://blob.vercel-storage.com/dummy-url/output/refined_questions.md"