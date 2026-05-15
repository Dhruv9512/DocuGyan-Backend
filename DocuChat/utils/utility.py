from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

from django.conf import settings

def rerank_docs(query: str, docs:list):
        try:
            reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")

            # Re-rank using FlashRank
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(docs)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked = reranker.rerank(rerank_request)
            reranked.sort(key=lambda x: x["score"], reverse=True)
            reranked = reranked[:settings.RERANK_K]  

            # Rebuild Document list in reranked order
            reranked_docs = [
                Document(
                    page_content=r["text"],
                    metadata=r["meta"]
                )
                for r in reranked
            ]

            return reranked_docs
        except Exception as e:
            print(f"Error during reranking: {e}")
            return docs 
        

def is_token_limit_error(e: Exception) -> bool:
    error_str = str(e).lower()
    return any(k in error_str for k in [
        "token", "context length", "context_length_exceeded",
        "maximum context", "too long", "input is too long",
        "reduce the length"
    ])