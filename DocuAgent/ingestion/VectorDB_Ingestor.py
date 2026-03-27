import logging

import requests
import re
from typing import List
from django.conf import settings

# Import the industry-standard Document format
from langchain_core.documents import Document
from langchain_milvus import Zilliz

# Adjust import based on your actual project structure
from DocuAgent.utils.utility import get_collection_name, create_zilliz_collection
from core.utils.llm_engine import LLMEngine
from langchain_text_splitters import RecursiveCharacterTextSplitter

from DocuAgent.models import DocuProcess

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
BATCH_SIZE = 100

class VectorDBIngestor:
    """
    Responsible for downloading extracted markdown files, chunking them page-by-page,
    converting them into standard LangChain Documents with rich metadata, and 
    inserting them into the Vector DB.
    """
    def __init__(self, project_id: str, extracted_doc_urls: List[str]):
        self.project_id = project_id
        self.collection_name = get_collection_name(project_id)
        self.extracted_doc_urls = extracted_doc_urls
        self.embedding_model = LLMEngine.get_huggingface_embedding_client()
        self.vectorstore = None
    
    def run(self) -> bool:
        try:
            logger.info(f"Starting Vector Ingestion for project {self.project_id}. Target Collection: {self.collection_name}")

            # 1. Download and parse into LangChain Documents
            documents = self._process_documents(self.extracted_doc_urls)
            if not documents:
                logger.warning("No valid text found in the provided documents to ingest.")
                raise ValueError("No valid text found in the provided documents to ingest.")

            # 2. Chunk the documents
            chunked_documents = self._document_chunking(documents)
            if not chunked_documents:
                logger.warning("Document chunking resulted in no valid chunks to ingest.")
                raise ValueError("Document chunking resulted in no valid chunks to ingest.")

            # 3. Insert into Zilliz vector DB
            if not self._insert_into_vector_db(chunked_documents):
                logger.error("Failed to insert documents into the vector database.")
                raise ValueError("Failed to insert documents into the vector database.")

            # 4. Update DB entry
            try:
                docu_instance = DocuProcess.objects.get(project_id=self.project_id)
                docu_instance.collection_name = self.collection_name
                docu_instance.save(update_fields=['collection_name'])
            except DocuProcess.DoesNotExist:
                logger.error(f"DocuProcess not found for project_id: {self.project_id}")
                raise ValueError(f"DocuProcess not found for the given project_id: {self.project_id}")

            logger.info(f"Successfully processed {len(chunked_documents)} page chunks.")
            return True

        except Exception as e:
            logger.error(f"FATAL: VectorDBIngestor failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"VectorDBIngestor failed: {str(e)}") from e

    def _process_documents(self, doc_urls: List[str]) -> List[Document]:
        """
        Downloads documents from URLs, parses them strictly by their markdown 
        '## Page X' headers, and wraps them in LangChain Document objects with metadata.
        """
        all_processed_docs = []
        
        for url in doc_urls:
            try:
                logger.info(f"Downloading extracted document: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                content = response.text

                # Parse the markdown into standard Documents
                parsed_pages = self._parse_markdown_to_documents(content, url)
                all_processed_docs.extend(parsed_pages)

            except requests.exceptions.RequestException as e:
                # Log the error but continue processing other URLs
                logger.error(f"Failed to download {url}. Skipping. Error: {e}")
                continue
        
        if not all_processed_docs:
            logger.warning("No valid documents were processed from the provided URLs.")
            raise ValueError("No valid documents were processed from the provided URLs.")

        return all_processed_docs

    def _parse_markdown_to_documents(self, raw_md: str, source_url: str) -> List[Document]:
        """
        Intelligently splits the DocuExtractor markdown format into separate pages.
        Extracts metadata like Page Number, Document Title, and Vision constraints.
        
        Expected Format:
        # Document Title
        
        ## Page 1
        Content...
        ---
        """
        documents = []
        
        # 1. Split the raw markdown by the "## Page " prefix.
        # This regex strictly splits lines starting with "## Page ", keeping the number/tags attached to the body block.
        raw_blocks = re.split(r'(?im)^##\s*Page\s*', raw_md)
        
        # 2. Parse the Introduction Block (Contains the Title)
        # raw_blocks[0] contains everything before the first "## Page "
        intro_block = raw_blocks[0]
        title_match = re.search(r'(?m)^#\s*(.+)', intro_block)
        document_title = title_match.group(1).strip() if title_match else "Unknown Document"

        # 3. Iterate through the actual page blocks
        for block in raw_blocks[1:]:
            
            # Separate the first line (e.g., "1" or "2 (Vision Extracted)") from the rest of the text
            parts = block.split('\n', 1)
            
            # Safety check: If block is empty or malformed
            if len(parts) < 2:
                continue
                
            header_remainder = parts[0].strip()  
            page_body = parts[1]

            # 4. Clean up the body content
            # Remove the trailing horizontal rules '---' and strip extra newlines
            clean_content = re.sub(r'(?m)^---\s*$', '', page_body).strip()

            # Skip truly empty pages or explicitly marked blank pages
            if not clean_content or clean_content == "[Blank Page]":
                continue

            # 5. Extract Rich Metadata
            # Extract the integer page number from the header remainder
            page_num_match = re.search(r'^(\d+)', header_remainder)
            page_num = int(page_num_match.group(1)) if page_num_match else 0
            
            # Check if this specific page required the Vision LLM fallback
            is_vision_extracted = "(Vision Extracted)" in header_remainder

            # 6. Construct the Industry-Standard LangChain Document
            doc = Document(
                page_content=clean_content,
                metadata={
                    "project_id": self.project_id,
                    "collection_name": self.collection_name,
                    "source_title": document_title,
                    "source_url": source_url,
                    "page_number": page_num,
                    "is_vision_extracted": is_vision_extracted,
                }
            )
            documents.append(doc)
            
        return documents
    
    def _document_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Intelligently chunks documents while perfectly preserving 
        the unique metadata (like page_number) of each original page.
        """
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

        # The splitter will create new chunks but we need to ensure that the metadata from the original documents is preserved in each chunk.
        chunked_docs = text_splitter.split_documents(documents)

        return chunked_docs

    def _insert_into_vector_db(self, chunked_documents: List[Document]) -> bool:
        try:
            if not chunked_documents:
                logger.warning("No chunked documents to insert into the vector database.")
                raise ValueError("No chunked documents to insert into the vector database.")

            logger.info("Connecting to Zilliz....")
            create_zilliz_collection(
                collection_name=self.collection_name,
                dim=384,
                zilliz_uri=getattr(settings, 'ZILLIZ_URI', None),
                zilliz_token=getattr(settings, 'ZILLIZ_TOKEN', None),
                alias=getattr(settings, 'ZILLIZ_ALIAS', 'default')
            )

            for i in range(0, len(chunked_documents), BATCH_SIZE):
                docs = chunked_documents[i:i + BATCH_SIZE]
                success = self._insert_batch_to_zilliz(docs)
                if not success:
                    logger.error(f"Failed on batch starting at index {i}")
                    raise ValueError(f"Failed to insert batch starting at index {i}")

            return True  

        except Exception as e:
            logger.error(f"Error: {e}")
            raise ValueError(f"Error during vector DB insertion: {e}") from e

    def _insert_batch_to_zilliz(self, batch: List[Document]) -> bool:
        if not batch:
            logger.warning("No documents provided for preparation.")
            raise ValueError("No documents provided for preparation.")

        try:
            if not self.vectorstore: 
                self.vectorstore = Zilliz(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                    connection_args={
                        "uri": getattr(settings, 'ZILLIZ_URI', None),
                        "token": getattr(settings, 'ZILLIZ_TOKEN', None),
                        "alias": getattr(settings, 'ZILLIZ_ALIAS', 'default')
                    }
                )
            self.vectorstore.add_documents(batch)
            logger.info(f"Inserted batch of {len(batch)} documents into Zilliz.")
            return True

        except Exception as e:
            logger.error(f"Error during batch insertion: {e}")
            raise ValueError(f"Error during batch insertion: {e}") from e   

            


# =========================================================
# Builder Function
# =========================================================
def build_vector_db_ingestor(project_id: str, extracted_doc_urls: list) -> bool:
    """
    Factory function to initialize and execute the VectorDBIngestor.
    """
    ingestor = VectorDBIngestor(project_id=project_id, extracted_doc_urls=extracted_doc_urls)
    return ingestor.run()