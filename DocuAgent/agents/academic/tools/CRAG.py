import logging
from typing import List, Union
from django.conf import settings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document

from DocuAgent.utils.utility import get_collection_name
from core.utils.llm_engine import LLMEngine
from DocuAgent.utils.llm_calls import DocuAgentLLMCalls
from DocuAgent.schemas.llm_schemas import RetrievalGraderOutput

logger = logging.getLogger(__name__)

class CorrectiveRetriever:
    """
    Industry-level C-RAG utility for DocuGyan.
    Executes secure retrieval, grades context, and falls back to Web Search.
    """

    def __init__(self, project_id: str, search_queries: Union[str, List[str]]) -> None:
        # CRITICAL FIX: Save project_id to the instance so the filter_expr works
        self.project_id = project_id
        self.embedding_model = LLMEngine.get_huggingface_embedding_client()
        
        # SAFETY CHECK: Handle String, List[str], or an accidental empty list from the LLM
        if isinstance(search_queries, str):
            self.search_queries = [search_queries]
        elif isinstance(search_queries, list) and len(search_queries) > 0:
            self.search_queries = search_queries
        else:
            logger.warning("[CorrectiveRetriever] Received empty search queries. Using fallback.")
            self.search_queries = ["general context"] 
            
        self.collection_name = get_collection_name(project_id)
        self.embeddings = LLMEngine.get_huggingface_embedding_client()
        self._vectorstore = None

    def _get_vectorstore(self) -> Milvus:
        if self._vectorstore is not None:
            return self._vectorstore

        self._vectorstore = Milvus(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            connection_args={
                "uri": getattr(settings, 'ZILLIZ_URI', None),
                "token": getattr(settings, 'ZILLIZ_TOKEN', None),
                "secure": True,
                "timeout": 60
            },
            primary_field="pk",          
            text_field="page_content",   
            vector_field="vector",  
        )
        return self._vectorstore
    
    def _execute_retrieval(self, top_k: int = 10) -> list[Document]:
        """Multi-Query Retrieval with Cosine Fusion & Metadata Filtering."""
        vectorstore = self._get_vectorstore()
        unique_chunks = {}
        
        # Uses the safely stored project_id
        filter_expr = f'project_id == "{self.project_id}"'

        for query in self.search_queries:
            try:
                hits = vectorstore.similarity_search_with_score(
                    query, 
                    k=top_k,
                    expr=filter_expr
                )
                for doc, score in hits:
                    page_num = doc.metadata.get('page_number', 'unknown')
                    chunk_hash = hash(f"{doc.page_content}_{page_num}")
                    
                    if chunk_hash not in unique_chunks:
                        unique_chunks[chunk_hash] = {"doc": doc, "best_score": score}
                    else:
                        if score > unique_chunks[chunk_hash]["best_score"]:
                            unique_chunks[chunk_hash]["best_score"] = score
            except Exception as exc:
                logger.warning("[CorrectiveRetriever] Query failed: query=%r | error=%s", query, exc)

        sorted_fused_results = sorted(unique_chunks.values(), key=lambda x: x["best_score"], reverse=True)
        return [item["doc"] for item in sorted_fused_results[:top_k]]

    def run(self, top_k: int = 10) -> dict:
        """
        Executes the full C-RAG workflow.
        Returns the raw LangChain Documents and the Grader Assessment.
        """
        all_docs = self._execute_retrieval(top_k=top_k)
        primary_query = self.search_queries[0]
        
        # ── Step 1: Zero-hit fast path ──
        if not all_docs:
            logger.warning("[CorrectiveRetriever] Zero hits from Milvus — forcing web search.")
            web_docs = self._web_search_as_documents(primary_query)
            fallback_grade = RetrievalGraderOutput(
                binary_score="not_found",
                reasoning="No documents matched the query in the database. Triggered web search."
            )
            return {
                "retrieved_docs": web_docs,
                "grader_assessment": fallback_grade
            }

        # ── Step 2: Grade ──
        grading_context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(all_docs)])
        grade: RetrievalGraderOutput = DocuAgentLLMCalls.call_retrieval_grader(primary_query, grading_context)

        # ── Step 3: Decide & Route ──
        if grade.binary_score == "not_found":
            logger.warning("[CorrectiveRetriever] Grade=not_found — replacing with web docs.")
            return {
                "retrieved_docs": self._web_search_as_documents(primary_query),
                "grader_assessment": grade
            }

        elif grade.binary_score == "ambiguous":
            logger.warning("[CorrectiveRetriever] Grade=ambiguous — augmenting with web docs.")
            web_docs = self._web_search_as_documents(primary_query)
            return {
                "retrieved_docs": all_docs + web_docs,
                "grader_assessment": grade
            }

        else:  # "accurate"
            logger.info("[CorrectiveRetriever] Grade=accurate — keeping internal docs.")
            return {
                "retrieved_docs": all_docs,
                "grader_assessment": grade
            }

    def _web_search_as_documents(self, query: str) -> list[Document]:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            
            raw_results: list[dict] = TavilySearchResults(k=10, api_key=getattr(settings, 'TAVILY_API_KEY', None)).run(query)
            return [
                Document(
                    page_content=r.get("content", ""),
                    metadata={"source": r.get("url", "web"), "title": r.get("title", ""), "type": "web"}
                ) for r in raw_results if r.get("content")
            ]
        except Exception as exc:
            logger.error("[CorrectiveRetriever] Web search failed: %s", exc)
            return [Document(page_content="Web search unavailable.", metadata={"source": "fallback", "type": "error"})]

# ================================================================
# Builder Function
# ================================================================
def build_CorrectiveRetriever(project_id: str, search_queries: Union[str, List[str]]) -> dict:
    try:
        retriever = CorrectiveRetriever(project_id, search_queries)
        return retriever.run()
    except Exception as exc:
        logger.error("[CorrectiveRetriever] Failed to build corrective retriever: %s", exc)
        return {
            "retrieved_docs": [],
            "grader_assessment": RetrievalGraderOutput(
                binary_score="not_found",
                reasoning=f"Internal retrieval error occurred: {exc}"
            )
        }