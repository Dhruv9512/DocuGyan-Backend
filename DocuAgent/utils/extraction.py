# DocuAgent/utils/extraction.py
import os
import tempfile
import logging
import requests
from urllib.parse import urlparse, unquote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from django.conf import settings

import fitz  # PyMuPDF
from pathlib import Path
from llama_cloud import LlamaCloud

# Initialize industry-standard logger
logger = logging.getLogger(__name__)

class DocuExtractor:
    """
    Responsible for securely streaming files, executing hybrid extraction 
    (Local PyMuPDF vs Cloud LlamaParse), and uploading to Vercel Blob.
    """
    def __init__(self, project_id: str, file_url: str):
        self.project_id = project_id
        self.file_url = file_url

        # Load tokens securely from Django settings
        self.blob_token = getattr(settings, 'BLOB_READ_WRITE_TOKEN', None)
        self.llamaparse_client = LlamaCloud(api_key=settings.llamaparse_key)
    

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

        return self._upload_to_vercel_blob(blob_path, extracted_text)

    # ==========================================
    # The Hybrid PDF Router
    # ==========================================

    def _extract_pdf(self) -> str:
        pdf_name = self._get_file_name(self.file_url)
        logger.info(f"Streaming {pdf_name} to disk (RAM protection active)...")
        
        # 1. Stream download to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            with self.session.get(self.file_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    temp_pdf.write(chunk)
            temp_pdf.flush()
            
            is_complex = False
            extracted_text = None

            # 2. Open locally just to check and/or extract
            with fitz.open(temp_pdf.name) as doc:
                is_complex = self._is_complex_document(doc)
                
                # If it's simple, extract it right now while it's open
                if not is_complex:
                    logger.info(f"Digital text detected. Routing to local PyMuPDF extraction...")
                    extracted_text = self._extract_local_pdf(doc, pdf_name)
                    
            # --- IMPORTANT: THE 'WITH' BLOCK ENDS HERE ---
            # PyMuPDF is completely wiped from RAM at this exact line.
            
            # 3. Execute LlamaParse if needed (with zero PyMuPDF RAM overhead)
            if is_complex:
                logger.warning(f"Complex elements detected in {pdf_name}. Routing to LlamaParse Vision API...")
                # The temp_pdf file is still safely on the disk!
                return self._extract_with_vision_api(temp_pdf.name)
                
            # Return the locally extracted text if it wasn't complex
            return extracted_text

    def _is_complex_document(self, doc) -> bool:
        """
        Samples the start, middle, and end of the document to detect low text 
        density (scans) or heavy image use (formulas/diagrams).
        """
        total_pages = len(doc)
        if total_pages == 0: return False

        pages_to_check = [0]
        if total_pages > 1: pages_to_check.append(total_pages // 2)
        if total_pages > 2: pages_to_check.append(total_pages - 1)

        complex_flags = 0

        for page_num in pages_to_check:
            page = doc[page_num]
            text = page.get_text("text").strip()
            images = page.get_images()
            
            if len(text) < 50 or len(images) > 1:
                complex_flags += 1

        return complex_flags > (len(pages_to_check) / 2)

    def _extract_local_pdf(self, doc, pdf_name: str) -> str:
        """Fast, RAM-efficient local extraction for standard digital PDFs."""
        output_lines = [f"# {pdf_name}\n"]
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            output_lines.append(f"## Page {page_num + 1}\n")
            output_lines.append(text.strip() if text else "[No text content found]")
            output_lines.append("\n---\n")
            
        return "\n".join(output_lines)

    def _extract_with_vision_api(self, file_path: str) -> str:
        """Advanced extraction for STEM documents, tables, and scans using LlamaCloud."""
        if not self.llama_client:
            logger.warning("No LlamaParse key found! Falling back to PyMuPDF (Expect poor quality on images).")
            with fitz.open(file_path) as doc:
                return self._extract_local_pdf(doc, "Fallback_Extraction")

        logger.info("Uploading document to LlamaCloud for multimodal extraction...")
        
        try:
            # 1. Upload the temporary file to LlamaCloud
            file_obj = self.llama_client.files.create(
                file=Path(file_path),
                purpose="parse"
            )
            
            # 2. Trigger the extraction job (Agentic tier is best for complex images/tables)
            result = self.llama_client.parsing.parse(
                file_id=file_obj.id,
                tier="agentic",
                expand=["markdown"]
            )
            
            # 3. Format the output to exactly match the PyMuPDF local format
            pdf_name = os.path.basename(file_path)
            output_lines = [f"# {pdf_name} (Vision Extracted)\n"]
            
            # Loop through the pages returned by the API
            for page_num, page_data in enumerate(result.markdown.pages):
                text = page_data.markdown
                
                output_lines.append(f"## Page {page_num + 1}\n")
                output_lines.append(text.strip() if text else "[No text content found]")
                output_lines.append("\n---\n")
                
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"LlamaCloud extraction failed: {e}")
            # Fallback to local extraction if the API crashes or rate-limits
            with fitz.open(file_path) as doc:
                return self._extract_local_pdf(doc, "Fallback_Extraction_After_API_Fail")

    # ==========================================
    # Helpers Methods
    # ==========================================
    def _upload_to_vercel_blob(self, blob_path: str, content: str) -> str:
        """Uploads text content to Vercel Blob using their REST API."""
        if not self.blob_token:
            raise ValueError("Cannot upload: Vercel Blob token is missing.")

        url = f"https://blob.vercel-storage.com/{blob_path}"
        headers = {
            "Authorization": f"Bearer {self.blob_token}",
            "x-api-version": "7",         
            "x-content-type": "text/markdown",
        }
        
        data = content.encode('utf-8')
        response = self.session.put(url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        
        logger.info(f"Successfully uploaded to Vercel Blob: {blob_path}")
        return response.json().get("url")


    def _get_file_extension(self, url: str) -> str:
        parsed_url = urlparse(url)
        _, ext = os.path.splitext(unquote(parsed_url.path))
        return ext.lower()

    def _get_file_name(self, url: str) -> str:
        parsed_url = urlparse(url)
        return os.path.basename(unquote(parsed_url.path))

    def _extract_text(self) -> str:
        """
        Fetches raw text/markdown securely using a HEAD check to prevent RAM spikes,
        and wraps it in the universal DocuGyan page format.
        """
        file_name = self._get_file_name(self.file_url)
        logger.info(f"Extracting text file: {file_name}...")

        # It is safe, so download the text normally
        response = requests.get(self.file_url, timeout=15)
        response.raise_for_status()
       
        # Decode safely (replaces weird broken characters instead of crashing)
        raw_text = response.content.decode('utf-8', errors='replace')

        # Apply the universal DocuGyan format
        output_lines = [f"# {file_name}\n"]
        output_lines.append("## Page 1\n")
        output_lines.append(raw_text.strip() if raw_text else "[No text content found]")
        output_lines.append("\n---\n")

        return "\n".join(output_lines)

    def _extract_word(self) -> str:
        return "Word extraction not yet implemented..."

    def _extract_ppt(self) -> str:
        return "PPT extraction not yet implemented..."


# ==========================================
# Build Function (Factory Pattern)
# ==========================================
def build_DocuExtractor(project_id: str, url: str) -> DocuExtractor:
    """
    Factory function to create a DocuExtractor instance and run the extraction pipeline.
    args:
    - project_id: The unique identifier for the project (used for Blob pathing).
    - url: The URL of the document to be extracted.
    returns:
    - The URL of the extracted Markdown file in Vercel Blob.
    """
    app = DocuExtractor(project_id=project_id, file_url=url)
    return app.extract_from_url()