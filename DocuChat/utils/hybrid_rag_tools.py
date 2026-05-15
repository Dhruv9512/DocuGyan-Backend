import json
import logging

from django.conf import settings
from core.utils.llm_engine import LLMEngine
from core.utils.utility import get_collection_name
from DocuChat.utils.utility import rerank_docs

logger = logging.getLogger(__name__)




class HybridRAGTools:
    def __init__(self, project_id, sources):
        self.project_id = project_id
        self.sources = sources
        self.collection_name = get_collection_name(project_id)
        self.embedding_model = LLMEngine.get_huggingface_embedding_client()

        self.search_expr = f"source_url in {json.dumps(self.sources)}" if self.sources else None
        self.docs = None

        result = self._load_retrievers()
        if result:
            from langchain_classic.retrievers import EnsembleRetriever
            self.bm25_retriever, self.vectorstore_retriever = result
            self.ensembled_hybrid_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vectorstore_retriever],
                weights=settings.ENSEMBLE_WEIGHTS
            )
        else:
            self.bm25_retriever = None
            self.vectorstore_retriever = None
            self.ensembled_hybrid_retriever = None

    def _load_retrievers(self):
        from pymilvus import connections, Collection, utility
        from pymilvus.exceptions import MilvusException
        from langchain_core.documents import Document
        from langchain_community.vectorstores import Milvus
        from langchain_community.retrievers import BM25Retriever
        try:
            connections.connect(
                alias="default",
                uri=getattr(settings, 'ZILLIZ_URI', None),
                token=getattr(settings, 'ZILLIZ_TOKEN', None),
                secure=True
            )
            logger.info(f"Connected to Milvus collection: {self.collection_name}")

            if not utility.has_collection(self.collection_name):
                logger.warning(f"Collection not found: {self.collection_name}")
                return None

            collection = Collection(name=self.collection_name)
            if utility.load_state(self.collection_name).name != "Loaded":
                collection.load()
                collection.wait_for_loading_complete()

        except MilvusException as e:
            logger.error(f"Error connecting to Milvus collection: {e}")
            return None
        except Exception as e:
            logger.error(f"General error loading Milvus collection: {e}")
            return None

        docs = []
        output_fields = ["page_content", "source_title", "page_number", "source_url", "chunk_type"]

        if self.search_expr:
            results = collection.query(expr=self.search_expr, output_fields=output_fields)
            for result in results:
                docs.append(Document(
                    page_content=result["page_content"],
                    metadata={
                        "source_title": result["source_title"],
                        "page_number": result["page_number"],
                        "source_url": result["source_url"],
                        "chunk_type": result["chunk_type"]
                    }
                ))

        # Fallback: load all docs if no expr or filtered query returned nothing
        if not docs:
            results = collection.query(expr="pk >= 0",output_fields=output_fields)
            for result in results:
                docs.append(Document(
                    page_content=result["page_content"],
                    metadata={
                        "source_title": result["source_title"],
                        "page_number": result["page_number"],
                        "source_url": result["source_url"],
                        "chunk_type": result["chunk_type"]
                    }
                ))

        if not docs:
            logger.warning(f"No documents found in Milvus collection: {self.collection_name}")
            return None

        self.docs = docs
        k = min(1500, max(100, int(0.002 * len(self.docs))))
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k

        vectorstore = Milvus(
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

        if self.search_expr:
            vectorstore_retriever = vectorstore.as_retriever(
                search_kwargs={"k": settings.TOP_K, "expr": self.search_expr}
            )
        else:
            vectorstore_retriever = vectorstore.as_retriever(
                search_kwargs={"k": settings.TOP_K}
            )
        logger.info(f"Initialized BM25 and vector retrievers for project {self.project_id} with {len(self.docs)} documents")
        return bm25_retriever, vectorstore_retriever


    def retrieve_docs(self, query: str) -> str:
        """
        Retrieves and reranks relevant documents for the given query.
        Returns a formatted string for the LLM to consume.
        """
        if not self.ensembled_hybrid_retriever:
            logger.error("Ensemble retriever not initialized for project %s", self.project_id)
            return "Retriever not initialized. No documents available."
        try:
            docs = self.ensembled_hybrid_retriever.invoke(query)

            logger.info(f"Retrieved {len(docs)} documents before reranking for query: {query}")
            reranked_docs = rerank_docs(query, docs)

            logger.info(f"{len(reranked_docs)} documents after reranking for query: {query}")
            if not reranked_docs:
                logger.warning(f"No relevant documents found for query: {query}")
                return "No relevant documents found."

            formatted = []
            for i, doc in enumerate(reranked_docs, 1):
                formatted.append(
                    f"[{i}] Source: {doc.metadata.get('source_title', 'Unknown')} "
                    f"| Page: {doc.metadata.get('page_number', 'N/A')}\n"
                    f"{doc.page_content}"
                )
            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            logger.error(f"Error during retrieval/reranking: {e}")
            return "Error occurred during document retrieval."


    def get_tools(self) -> list:
        """Returns list of LangChain-compatible tools."""
        from langchain_core.tools import StructuredTool
        retrieve_tool = StructuredTool.from_function(
            func=self.retrieve_docs,
            name="retrieve_documents",
            description=(
                "Retrieves the most relevant documents from the knowledge base "
                "for a given query using hybrid BM25 + vector search with re-ranking. "
                "Always call this before answering questions about documents."
            ),
        )
        return [retrieve_tool]