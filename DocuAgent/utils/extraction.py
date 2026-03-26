# DocuAgent/utils/extraction.py
import os
import tempfile
import logging
from urllib.parse import urlparse, unquote
import re

import fitz
import requests  # PyMuPDF

# Import LLM Utility for Vision Calls
from DocuAgent.utils.llm_calls import DocuAgentLLMCalls
from DocuAgent.utils.utility import upload_to_vercel_blob



# Initialize industry-standard logger
logger = logging.getLogger(__name__)

class DocuExtractor:
    """
    Responsible for securely streaming files, executing intelligent hybrid extraction 
    (Local PyMuPDF vs Vision LLM), and uploading to Vercel Blob.
    """
    def __init__(self, project_id: str, file_url: str):
        self.project_id = project_id
        self.file_url = file_url
        self.session = requests.Session()

  

    def extract_from_url(self) -> str:
        """
        Main pipeline: Determines file type, routes extraction, and uploads to Blob.
        """
        extension = self._get_file_extension(self.file_url)
        file_name = self._get_file_name(self.file_url)
        
        logger.info(f"Starting pipeline for: {file_name} (Project: {self.project_id})")

        try:
            if extension == '.pdf':
                extracted_text = self._extract_pdf()
            elif extension in ['.txt', '.md']:
                extracted_text = self._extract_text()
            elif extension in ['.doc', '.docx']:
                extracted_text = self._extract_word()
            elif extension in ['.ppt', '.pptx']:
                extracted_text = self._extract_ppt()
            else:
                raise ValueError(f"Unsupported format: {extension}")
        except Exception as e:
            logger.error(f"Extraction failed for {self.file_url}: {e}", exc_info=True)
            raise

        # Upload final Markdown to Vercel Blob
        base_name = os.path.splitext(file_name)[0]
        md_filename = f"{base_name}.md"
        blob_path = f"{self.project_id}/temp/{md_filename}"

        return upload_to_vercel_blob(blob_path=blob_path, content=extracted_text,content_type="text/markdown")

    # ==========================================
    # The Intelligent PDF Router
    # ==========================================
    def _extract_pdf(self) -> str:
        pdf_name = self._get_file_name(self.file_url)
        logger.info(f"Streaming {pdf_name} to disk (RAM protection active)...")
        
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            with self.session.get(self.file_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        temp_pdf.write(chunk)
            temp_pdf.flush()
            
            with fitz.open(temp_pdf.name) as doc:
                total_pages = len(doc)
                
                # 1. THE PLACEHOLDER ARRAY: Guarantees perfect order
                # Creates an array like: [None, None, None, None]
                extracted_pages = [None] * total_pages 
                
                vision_batch_images = []
                vision_batch_indices = []
                BATCH_SIZE = 3 
                
                for page_num, page in enumerate(doc):
                    
                    if self._needs_vision_llm(page):
                        # Add scan to the batch, remembering its EXACT page number (index)
                        pix = page.get_pixmap(dpi=150)
                        vision_batch_images.append(pix.tobytes("jpeg"))
                        vision_batch_indices.append(page_num)
                        
                        # Flush batch if it's full
                        if len(vision_batch_images) >= BATCH_SIZE:
                            self._process_and_place_pdf_batch(
                                vision_batch_images, vision_batch_indices, extracted_pages
                            )
                            vision_batch_images.clear()
                            vision_batch_indices.clear()
                            
                    else:
                        # 2. FLUSH ON TRANSITION: 
                        # If we were batching scans, but just hit a digital page, 
                        # we must flush the scans immediately to prevent non-consecutive 
                        # pages (like 1, 5, 9) from being merged into one block.
                        if vision_batch_images:
                            self._process_and_place_pdf_batch(
                                vision_batch_images, vision_batch_indices, extracted_pages
                            )
                            vision_batch_images.clear()
                            vision_batch_indices.clear()

                        # 3. Process digital page instantly and put in exact slot
                        logger.info(f"Page {page_num + 1}: Digital Text.")
                        extracted_pages[page_num] = (
                            f"## Page {page_num + 1}\n"
                            f"{self._extract_page_local(page)}\n"
                            f"---\n"
                        )
                
                # 4. END OF DOCUMENT: Flush any remaining scans in the queue
                if vision_batch_images:
                    self._process_and_place_pdf_batch(
                        vision_batch_images, vision_batch_indices, extracted_pages
                    )
                    
            # 5. Build the final perfectly-ordered document
            final_markdown = f"# {pdf_name}\n\n"
            for page_content in extracted_pages:
                if page_content: # Ignore empty slots from batch merging
                    final_markdown += page_content
                    
            return final_markdown

    def _process_and_place_pdf_batch(self, images: list, indices: list, extracted_pages: list):
        """
        Processes a batch of images via LLM, parses the page-by-page output, 
        and places each page's result exactly where it belongs in the array.
        """
        logger.info(f"Processing Vision Batch for pages: {[i+1 for i in indices]}")
        
        try:
            # 1. Call your LLM API (Gemini/HF) with the list of images
            extracted_text_block = DocuAgentLLMCalls.PDFExtractorLLM(images)
            
            # 2. Split the LLM's response by the "## Page X" headers
            # This regex splits the string every time it sees "## Page " followed by a number at the start of a line.
            raw_blocks = re.split(r'(?im)^##\s*Page\s*\d+', extracted_text_block)
            
            # Clean up empty strings (like any intro text the LLM added before the first header)
            parsed_blocks = [block.strip() for block in raw_blocks if block.strip()]
            
            # 3. Validation: Did the LLM give us the correct number of pages back?
            if len(parsed_blocks) == len(indices):
                # SUCCESS: The LLM perfectly separated the pages.
                logger.info("LLM perfectly separated the batch pages. Mapping to array...")
                for i, idx in enumerate(indices):
                    page_num = idx + 1
                    # Place the exact content into the exact array slot
                    extracted_pages[idx] = f"## Page {page_num} (Vision Extracted)\n{parsed_blocks[i]}\n---\n"
                    
            else:
                # FALLBACK: The LLM messed up the formatting (merged pages, missed a header, etc.)
                # To prevent data loss, we dump the whole raw response into the first slot.
                logger.warning(f"LLM format mismatch. Expected {len(indices)} blocks, got {len(parsed_blocks)}. Using fallback.")
                
                first_slot = indices[0]
                if len(indices) > 1:
                    header = f"## Pages {indices[0]+1} to {indices[-1]+1} (Vision Extracted - Format Fallback)\n"
                else:
                    header = f"## Page {indices[0]+1} (Vision Extracted)\n"
                    
                extracted_pages[first_slot] = f"{header}{extracted_text_block}\n---\n"
                
                # Empty the other slots so we don't get duplicate data
                for idx in indices[1:]:
                    extracted_pages[idx] = ""
                
        except Exception as e:
            logger.error(f"Vision Batch failed for pages {indices}: {e}")
            # If the API fails completely, insert error messages into the correct slots
            for idx in indices:
                extracted_pages[idx] = f"## Page {idx+1}\n[Extraction Failed: Vision API Error]\n---\n"

    def _needs_vision_llm(self, page) -> bool:
        """
        Evaluates a page to see if it requires LLM Vision analysis to catch edge cases.
        """
        text = page.get_text("text").strip()
        images = page.get_images()
        
        # 1. THE BLANK PAGE CHECK
        if len(text) < 50 and len(images) == 0:
            return False # It's just a blank page, skip it.

        # 2. THE DIGITAL FOOTER TRAP (Raised threshold)
        if len(text) < 200 and len(images) > 0:
            return True

        # 3. THE GIANT DIAGRAM TRAP
        if len(images) > 0:
            page_area = page.rect.width * page.rect.height
            if page_area == 0: return False
            
            image_area = 0
            for img in page.get_image_info():
                bbox = img["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                image_area += (width * height)
                
            if (image_area / page_area) > 0.30:
                return True

        # 4. DEFAULT: It's a normal, text-heavy digital document.
        return False

    def _extract_page_local(self, page) -> str:
        lines = []
        
        # EXTRACT TABLES FIRST
        tables = page.find_tables()
        if tables:
            for table in tables:
                lines.append(table.to_markdown())
                lines.append("\n")
        
        # EXTRACT TEXT WITH LAYOUT AWARENESS
        try:
            text = page.get_text("markdown")
            if text:
                lines.append(text)
        except Exception:
            # INDUSTRY FALLBACK: Sort text blocks by their physical Y/X coordinates
            # This handles two-column layouts beautifully if "markdown" mode fails
            blocks = page.get_text("blocks")
            # Sort by Y coordinate first (top to bottom), then X (left to right)
            blocks.sort(key=lambda b: (b[1], b[0])) 
            for b in blocks:
                lines.append(b[4]) # b[4] contains the actual text string
            
        content = "\n".join(lines).strip()
        return content if content else "[Blank Page]"

    
    # ==========================================
    # Standard Helpers
    # ==========================================
    def _get_file_extension(self, url: str) -> str:
        parsed_url = urlparse(url)
        _, ext = os.path.splitext(unquote(parsed_url.path))
        return ext.lower()

    def _get_file_name(self, url: str) -> str:
        parsed_url = urlparse(url)
        return os.path.basename(unquote(parsed_url.path))

    def _extract_text(self) -> str:
        file_name = self._get_file_name(self.file_url)
        logger.info(f"Extracting text file: {file_name}...")

        response = self.session.get(self.file_url, timeout=15)
        response.raise_for_status()
       
        raw_text = response.content.decode('utf-8', errors='replace')

        output_lines = [
            f"# {file_name}\n",
            "## Page 1\n",
            raw_text.strip() if raw_text else "[No text content found]",
            "\n---\n"
        ]
        return "\n".join(output_lines)

    def _extract_word(self) -> str:
        return "Word extraction not yet implemented... (Use python-docx here)"

    def _extract_ppt(self) -> str:
        return "PPT extraction not yet implemented... (Use python-pptx here)"

# ==========================================
# Factory Function
# ==========================================
def build_DocuExtractor(project_id: str, url: str) -> DocuExtractor:
    app = DocuExtractor(project_id=project_id, file_url=url)
    return app.extract_from_url()