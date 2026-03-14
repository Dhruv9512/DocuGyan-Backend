# DocuAgent/utils/extraction.py
import os
import io
import requests
from urllib.parse import urlparse, unquote
import PyPDF2  

class DocuExtractor:
    """
    Responsible solely for downloading files and extracting raw text/markdown.
    """
    def __init__(self):
        # You can initialize specific API clients here if needed (e.g., OCR services)
        pass

    def extract_from_url(self, file_url: str) -> str:
        """Determines file type and routes to the correct extraction method."""
        extension = self._get_file_extension(file_url)
        
        if extension == '.pdf':
            return self._extract_pdf(file_url)
        elif extension in ['.txt', '.md']:
            return self._extract_text(file_url)
        elif extension in ['.doc', '.docx']:
            return self._extract_word(file_url)
        elif extension in ['.ppt', '.pptx']:
            return self._extract_ppt(file_url)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _get_file_extension(self, url: str) -> str:
        """Safely extracts the file extension from a URL, ignoring query parameters."""
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)
        _, ext = os.path.splitext(path)
        return ext.lower()

    def _get_file_name(self, url: str) -> str:
        """Extracts just the file name from a URL."""
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)
        return os.path.basename(path)

    def _extract_pdf(self, url: str) -> str:
        """
        Downloads the PDF into memory and extracts text page-by-page.
        Format matches:
        
        pdf name
        -----
        page 1
        
        <content>
        """
        pdf_name = self._get_file_name(url)
        
        # Download the file directly into memory
        response = requests.get(url)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        output_lines = [pdf_name]
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            output_lines.append("")
            output_lines.append("-----")
            output_lines.append(f"page {page_num + 1}")
            output_lines.append("")
            output_lines.append(text.strip() if text else "")
            
        return "\n".join(output_lines)

    def _extract_text(self, url: str) -> str:
        """Downloads and decodes raw text/markdown files."""
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def _extract_word(self, url: str) -> str:
        # Placeholder for future Word document extraction logic
        return "extracted word content..."

    def _extract_ppt(self, url: str) -> str:
        # Placeholder for future PowerPoint extraction logic
        return "extracted ppt content..."