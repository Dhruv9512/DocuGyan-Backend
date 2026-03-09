CLASSIFY_RAG_PROMPT = """
    You are a document analysis expert. Analyze the following document excerpts and determine the best RAG (Retrieval Augmented Generation) strategy.

    **Strategies:**
    - `vector_rag`: Best for long-form unstructured text (articles, reports, research papers, books). Uses semantic embedding search.
    - `graph_rag`: Best for highly structured/relational data (org charts, legal references, citation networks, entity-heavy docs). Uses knowledge graph traversal.
    - `vectorless`: Best for short, simple documents (FAQs, glossaries, small Q&A sets, lists). Uses keyword/BM25 search.

    **Document excerpts:**
    {document_excerpts}

    Classify the appropriate RAG strategy.
"""


REFINE_QUESTIONS_PROMPT = """
    You are a question refinement expert. Given the original user questions and document context, refine the questions to be more specific, clear, and answerable from the provided documents.

    **Original questions:**
    {questions}

    **Document context:**
    {context}

    Return a list of refined questions. Keep the same number of questions. Make them specific and answerable.
"""