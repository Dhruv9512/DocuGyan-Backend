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
from DocuAgent.utils.utility import upload_to_vercel_blob, get_collection_name, get_request_session_with_blob_auth, sanitize_blob_filename
from django.conf import settings



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
        self.session = get_request_session_with_blob_auth()
        self.image_cache = {}  
        self.blob_collection = get_collection_name(self.project_id)

  

    def extract_from_url(self) -> str:
        extension = self._get_file_extension(self.file_url)
        file_name = self._get_file_name(self.file_url)
        
        logger.info(f"Starting pipeline for: {file_name} (Project: {self.project_id})")

        try:
            if extension == '.pdf':
                extracted_text = self._extract_pdf()
            elif extension in ['.txt', '.md']:
                extracted_text = self._extract_text()
            elif extension in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                extracted_text = self._extract_image()
            elif extension in ['.doc', '.docx']:
                extracted_text = self._extract_word()
            elif extension in ['.ppt', '.pptx']:
                extracted_text = self._extract_ppt()
            else:
                raise ValueError(f"Unsupported format: {extension}")

            md_filename = sanitize_blob_filename(f"{os.path.splitext(file_name)[0].strip()}.md")
            blob_path = f"{self.blob_collection}/temp/{md_filename}"

            logger.info("Uploading extracted Markdown to Vercel Blob...")
            return upload_to_vercel_blob(blob_path=blob_path, content=extracted_text, content_type="text/markdown")

        except Exception as e:
            logger.error(f"Pipeline failed for {self.file_url}: {e}", exc_info=True)
            raise RuntimeError(f"Extraction pipeline failed: {str(e)}") from e
        
        finally:
            self.session.close()
        
    # ==========================================
    # The Intelligent PDF Router
    # ==========================================
    def _extract_pdf(self) -> str:
        pdf_name = self._get_file_name(self.file_url)
        logger.info(f"Streaming {pdf_name} to disk (RAM protection active)...")

        # 1. Capture the name outside the try block so the finally block always knows what to delete
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf_name = temp_pdf.name 
        
        try:
            # 2. Safely stream to disk
            with self.session.get(self.file_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        temp_pdf.write(chunk)
            # Flush and close before PyMuPDF tries to open it
            temp_pdf.flush()
            temp_pdf.close()
            
            # 3. Process the file
            with fitz.open(temp_pdf_name) as doc:
                total_pages = len(doc)
                extracted_pages = [None] * total_pages 
                
                vision_batch_images = []
                vision_batch_indices = []
                BATCH_SIZE = 3 
                
                for page_num, page in enumerate(doc):
                    
                    if self._needs_vision_llm(page):
                        pix = page.get_pixmap(dpi=150)
                        vision_batch_images.append(pix.tobytes("jpeg"))
                        vision_batch_indices.append(page_num)
                        
                        if len(vision_batch_images) >= BATCH_SIZE:
                            self._process_and_place_pdf_batch(
                                vision_batch_images, vision_batch_indices, extracted_pages
                            )
                            vision_batch_images.clear()
                            vision_batch_indices.clear()
                            
                    else:
                        if vision_batch_images:
                            self._process_and_place_pdf_batch(
                                vision_batch_images, vision_batch_indices, extracted_pages
                            )
                            vision_batch_images.clear()
                            vision_batch_indices.clear()

                        logger.info(f"Page {page_num + 1}: Digital Text.")
                        extracted_pages[page_num] = (
                            f"## Page {page_num + 1}\n"
                            f"{self._extract_page_local(page)}\n"
                            f"---\n"
                        )
                
                if vision_batch_images:
                    self._process_and_place_pdf_batch(
                        vision_batch_images, vision_batch_indices, extracted_pages
                    )
                    
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process PDF: {str(e)}") from e
            
        finally:
            # 4. GUARANTEED CLEANUP: This runs even if the download times out or the LLM fails
            if not temp_pdf.closed:
                temp_pdf.close()
            if os.path.exists(temp_pdf_name):
                os.remove(temp_pdf_name)
                
        # 5. Build the final perfectly-ordered document
        final_markdown = f"# {pdf_name}\n\n"
        for page_content in extracted_pages:
            if page_content: 
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
            extracted_text_block = DocuAgentLLMCalls.VisionExtractorLLM(images)
            
            # 2. Split the LLM's response by the "## Page X" headers
            # This regex splits the string every time it sees "## Page " followed by a number at the start of a line.
            raw_blocks = re.split(r'(?im)^##\s*Pages?\s*[\d\s,to-]+', extracted_text_block)
            
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
            raise RuntimeError(f"Vision API failed during batch extraction: {str(e)}") from e

    def _needs_vision_llm(self, page) -> bool:
        """
        Evaluates a page to determine if Vision LLM is needed.
        Handles: scanned pages, OCR ghost layers, giant diagrams,
        vector drawings, tracking pixels, and watermarks.
        """
        text = page.get_text("text").strip()
        text_len = len(text)
        
        # 1. NOISE FILTER — Find meaningful images (>50x50)
        # Prevents microscopic tracking pixels from falsely triggering the Vision LLM
        meaningful_images = []
        for img in page.get_image_info():
            bbox = img["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w >= 50 and h >= 50:
                meaningful_images.append(img)
                
        meaningful_image_count = len(meaningful_images)
        drawings = page.get_drawings()

        # 2. TRULY BLANK PAGE
        # Skip entirely if there is no text, no real images, and no vector shapes
        if text_len < 50 and meaningful_image_count == 0 and len(drawings) < 5:
            return False

        # 3. OCR GHOST LAYER DETECTION
        # Catches bad invisible OCR text baked into scanned PDFs (garbled text)
        if meaningful_image_count >= 1 and text_len > 0:
            words = text.split()
            if len(words) > 10: 
                avg_word_len = sum(len(w) for w in words) / len(words)
                # Real text averages 3-12 chars. OCR garbage is often 1-2 or 20+
                if avg_word_len < 2.5 or avg_word_len > 15:
                    return True

        # 4. SCANNED PAGE / SPARSE TEXT TRAP
        if text_len < 200 and (meaningful_image_count > 0 or len(drawings) > 10):
            return True

        # 5. GIANT DIAGRAM TRAP (WITH WATERMARK PROTECTION)
        if meaningful_image_count > 0:
            page_area = page.rect.width * page.rect.height
            if page_area > 0:
                image_area = sum(
                    (img["bbox"][2] - img["bbox"][0]) * (img["bbox"][3] - img["bbox"][1])
                    for img in meaningful_images
                )
                
                if (image_area / page_area) > 0.30:
                    # WATERMARK PROTECTION: If there's massive amounts of readable text (> 1000 chars), 
                    # the giant image is just a background watermark/border. Don't use Vision API.
                    if text_len > 1000:
                        return False 
                    return True

        # 6. COMPLEX VECTOR GRAPHIC TRAP
        # Flowcharts/CAD drawings can have dozens of vector lines AND lots of text.
        # This routes them to the Vision LLM so visual relationships are preserved.
        if len(drawings) > 30:
            return True

        # 7. DEFAULT: Normal digital text page
        return False

    def _extract_page_local(self, page) -> str:
        lines = []
        doc = page.parent

        # 1. ROTATION GUARD (Claude's excellent addition)
        if page.rotation != 0:
            page.set_rotation(0)

        # 2. EXTRACT FORM FIELDS (Claude's addition)
        try:
            widgets = page.widgets()
            if widgets:
                lines.append("### Form Fields\n")
                for widget in widgets:
                    field_name = widget.field_name or "Unknown Field"
                    field_value = widget.field_value or "[Empty]"
                    lines.append(f"**{field_name}:** {field_value}")
                lines.append("\n")
        except Exception:
            pass 

        # 3. PRE-PROCESS TABLES (Spatial mapping)
        tables = page.find_tables()
        table_data = []

        if tables and tables.tables:  
            for table in tables.tables:  
                md = None  
                try:
                    md = table.to_markdown()
                    if not md or not md.strip():  
                        raise ValueError("Empty markdown from table")
                except Exception as te:
                    logger.warning(f"table.to_markdown() failed: {te}")
                    try:
                        rows = table.extract()
                        md = "\n".join(
                            "| " + " | ".join(str(c or "") for c in row) + " |"
                            for row in rows if any(row)
                        )
                    except Exception:
                        md = "[Table extraction failed]"

                if md: 
                    table_data.append({
                        "bbox": fitz.Rect(table.bbox),
                        "markdown": md,
                        "extracted": False
                    })

        # 4. PRE-PROCESS IMAGES (Spatial mapping)
        images = page.get_image_info(xrefs=True)
        image_data = []
        for img in images:
            image_data.append({
                "bbox": fitz.Rect(img["bbox"]),
                "xref": img["xref"],
                "extracted": False
            })

        # 5. NATURAL READING FLOW (The Enterprise Engine)
        # sort=True natively handles 2-col, 3-col, and unequal columns perfectly.
        blocks = page.get_text("blocks", sort=True)
        
        for b in blocks:
            block_rect = fitz.Rect(b[:4])
            block_type = b[6]

            # --- TEXT BLOCKS ---
            if block_type == 0:
                block_text = b[4].strip()
                if not block_text:
                    continue

                in_table = False
                for t in table_data:
                    if block_rect.intersects(t["bbox"]):
                        in_table = True
                        # Inject table exactly where it belongs in the text flow
                        if not t["extracted"]:
                            lines.append("\n" + t["markdown"] + "\n")
                            t["extracted"] = True
                        break 
                
                if not in_table:
                    lines.append(block_text)

            # --- IMAGE BLOCKS ---
            elif block_type == 1:
                for img in image_data:
                    if block_rect.intersects(img["bbox"]):
                        if not img["extracted"]:
                            image_url = self._extract_and_upload_image(doc, img["xref"])
                            if image_url:
                                lines.append(f"\n![Embedded Image]({image_url})\n")
                            img["extracted"] = True
                        break

        # 6. ORPHAN CATCHER (Safety Net)
        for t in table_data:
            if not t["extracted"]:
                lines.append("\n" + t["markdown"] + "\n")
        
        for img in image_data:
            if not img["extracted"]:
                image_url = self._extract_and_upload_image(doc, img["xref"])
                if image_url:
                    lines.append(f"\n![Embedded Image]({image_url})\n")

        # 7. HYPERLINKS (Claude's addition)
        try:
            links = page.get_links()
            url_links = [l for l in links if l.get("kind") == 2] 
            if url_links:
                lines.append("\n### Document Links\n")
                for link in url_links:
                    uri = link.get("uri", "")
                    if uri:
                        lines.append(f"- {uri}")
        except Exception:
            pass

        content = "\n\n".join(lines).strip()
        return content if content else "[Blank Page]"
    
    def _extract_image(self) -> str:
        """
        Processes a standalone image file using the Vision LLM pipeline and 
        formats it to match the standard DocuGyan Document schema.
        """
        file_name = self._get_file_name(self.file_url)
        logger.info(f"Streaming single image file: {file_name}...")

        try:
            # 1. Download the raw image bytes securely
            response = self.session.get(self.file_url, timeout=30)
            response.raise_for_status()
            
            with fitz.open(stream=response.content, filetype="image") as img_doc:
                page = img_doc[0]
                pix = page.get_pixmap(dpi=150)
                normalized_image_bytes = pix.tobytes("jpeg")

            # 2. Call the Vision LLM (reusing your resilient 3-tier pipeline)
            # We wrap image_bytes in a list because PDFExtractorLLM expects a batch
            logger.info("Sending normalized image to Vision LLM for extraction...")
            extracted_text = DocuAgentLLMCalls.VisionExtractorLLM([normalized_image_bytes])

            # 3. Clean and Validate the Output
            extracted_text = extracted_text.strip()
            
            # Fallback: Just in case the LLM forgot the mandated '## Page 1' header
            if not extracted_text.startswith("## Page"):
                extracted_text = f"## Page 1 (Vision Extracted)\n{extracted_text}"

            # 4. Construct the Standardized Markdown Document
            final_markdown = (
                f"# {file_name}\n\n"
                f"{extracted_text}\n"
                f"---\n"
            )
            
            return final_markdown

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image file {file_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to download image file: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to process image {file_name} via Vision LLM: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract text from image: {str(e)}") from e
        

    def _extract_and_upload_image(self, doc, xref: int) -> str:
        """
        Extracts an embedded image by its PDF XREF, deduplicates it, 
        filters out microscopic noise, and uploads it to Vercel Blob.
        """
        # 1. Deduplication: If we already uploaded this exact image, reuse the URL
        if xref in self.image_cache:
            return self.image_cache[xref]
            
        try:
            # 2. Extract raw image data from the PDF
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            width = base_image["width"]
            height = base_image["height"]
            
            # 3. Noise Filtering: Ignore microscopic 1x1 tracking pixels or tiny icons
            if width < 50 or height < 50:
                return ""
                
            # 4. Generate a unique path for the blob storage
            blob_path = f"{get_collection_name(self.project_id)}/temp/img_{xref}.{ext}"
            
            # 5. Upload the raw bytes directly to Vercel
            image_url = upload_to_vercel_blob(
                blob_path=blob_path, 
                content=image_bytes, 
                content_type=f"image/{ext}"
            )
            
            # 6. Cache the result to prevent re-uploading the same image (like headers)
            self.image_cache[xref] = image_url
            return image_url
            
        except Exception as e:
            logger.warning(f"Failed to extract or upload embedded image xref {xref}: {e}")
            return ""
        
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

        try:
            response = self.session.get(self.file_url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download text file from {file_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to download text file from {self.file_url}: {e}") from e
       
        raw_text = response.content.decode('utf-8', errors='replace')

        output_lines = [
            f"# {file_name}\n\n",
            "## Page 1\n",
            raw_text.strip() if raw_text else "[No text content found]",
            "\n---\n"
        ]
        return "\n".join(output_lines)

    def _extract_word(self) -> str:
        raise NotImplementedError("Word (.doc/.docx) extraction is not yet supported in this pipeline.")

    def _extract_ppt(self) -> str:
        raise NotImplementedError("PowerPoint (.ppt/.pptx) extraction is not yet supported in this pipeline.")

# ==========================================
# Factory Function
# ==========================================
def build_DocuExtractor(project_id: str, url: str) -> str:
    
    try:
        app = DocuExtractor(project_id=project_id, file_url=url)
        extracted_md_url = app.extract_from_url() 
        return extracted_md_url
    except Exception as e:
        logger.error(f"DocuExtractor failed: {e}", exc_info=True)
        raise RuntimeError(f"DocuExtractor failed: {str(e)}") from e
