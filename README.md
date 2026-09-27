# HireFlow

Intelligent candidate search and evaluation — a modular RAG pipeline over
resumes and job descriptions, with hybrid retrieval, LLM-based reranking,
and RAG quality evaluation.

Start with **[HOMEWORK.md](./HOMEWORK.md)** for setup and a guided build order.

## Project structure

```
HireFlow/
├── core/
│   ├── ingestion.py        # Load resume and job description PDFs as Documents
│   ├── parsing.py          # LLM-based resume / job description parsing
│   ├── vector_store.py     # Pinecone/FAISS indexing and semantic search
│   ├── hybrid_indexer.py   # BM25 keyword search + vector search, combined
│   ├── search_router.py    # Routes queries to shallow or deep search
│   ├── re_ranker.py        # AI evaluation of candidate fit
│   ├── evaluator.py        # RAGAS quality evaluation
│   └── memory_rag.py       # Search history for the session
├── utils/
│   ├── config.py           # Configuration (reads .env)
│   ├── schemas.py          # Pydantic data models
│   ├── multi_query.py      # Rewrites a query several ways
│   └── utils.py            # PDF loading, embeddings, rate limiter, filters
├── streamlit/
│   └── app.py              # Web interface
├── data/
│   ├── resumes/            # Sample resume PDFs go here
│   └── jds/                # Sample job description PDFs go here
├── main.py                 # Command-line demo
└── HOMEWORK.md
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY

python main.py --jd data/jds/example.pdf
# or
streamlit run streamlit/app.py
```

Every module is currently a stub — see `HOMEWORK.md` for the recommended
implementation order.
