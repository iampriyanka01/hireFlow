"""
HireFlow core package.

Modules (in pipeline order):
    ingestion      - PDFs -> LangChain Documents
    parsing        - LLM structured extraction (resume / job description)
    vector_store   - Pinecone embeddings + semantic search
    hybrid_indexer - BM25 keyword search combined with vector search
    search_router  - shallow vs deep search routing (+ LLM re-ranking)
    re_ranker      - LLM candidate-vs-job evaluation
    evaluator      - RAGAS search-quality metrics
    memory_rag     - session search history
"""
