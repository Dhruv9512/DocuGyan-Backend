import os

class DocumentExtractor:
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
        # Future expansions:
        # elif extension == '.docx':
        #     return self._extract_docx(file_url)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _get_file_extension(self, url: str) -> str:
        # Basic logic to extract '.pdf' from 'https://vercel.blob.../file.pdf'
        _, ext = os.path.splitext(url.split('?')[0]) 
        return ext.lower()

    def _extract_pdf(self, url: str) -> str:
        # Logic to download and parse PDF
        return "extracted pdf content..."

    def _extract_text(self, url: str) -> str:
        # Logic to download and parse raw text
        return "extracted text content..."