#streamlit/app.py
"""HireFlow - Clean Architecture Implementation"""

import streamlit as st
from pathlib import Path
import sys
from typing import Any
import re
import json
from langchain_core.documents import Document

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import ui_helpers` works
import ui_helpers as uh   # pictures, resume/JD parsing and HTML card builders (see ui_helpers.py)

from core.hybrid_indexer import HybridIndexer
from core.vector_store import VectorStore, RESUME_NAMESPACE
from core.re_ranker import ReRanker
from core.ingestion import load_resumes, DocumentProcessor
from core.parsing import ResumeParser, JobParser
from utils.schemas import JobDescription, Resume
from utils.utils import get_rate_limiter
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from core.memory_rag import MemoryRAG
from core.search_router import SearchRouter
from core.evaluator import RAGEvaluator

# Data directory
DATA_RESUMES_DIR = project_root / "data" / "resumes"
DATA_JDS_DIR = project_root / "data" / "jds"

# Page config
st.set_page_config(page_title="HireFlow", page_icon="🎯", layout="wide")
st.markdown(uh.CSS, unsafe_allow_html=True)   # cards, chips, avatars, timeline styles

# ============================================================================
# DISPLAY DATA (rich details for the UI, read straight from the PDFs)
# ============================================================================
# The search engines return thin results (name + text; skills / location are
# placeholders - see HW1 / HW9). To still show real details and pictures, the UI
# parses every resume PDF itself with simple rules (no LLM, no API cost) and looks
# each search result up in that table. Cached, and refreshed when a PDF is added.

def _pdf_signature(directory: Path) -> tuple:
    """(filename, modified-time) of every PDF: the cache key that detects uploads."""
    if not directory.exists():
        return ()
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(directory.glob("*.pdf")))


@st.cache_data(show_spinner="Reading resumes...")
def _cached_profiles(signature: tuple) -> list:
    return uh.load_profiles(DATA_RESUMES_DIR)


@st.cache_data(show_spinner=False)
def _cached_jobs(signature: tuple) -> list:
    return uh.load_jobs(DATA_JDS_DIR)


def get_profiles() -> list:
    """Parsed resume details (name, title, location, years, skills, education, ...)."""
    return _cached_profiles(_pdf_signature(DATA_RESUMES_DIR))


def get_jobs() -> list:
    """Parsed job descriptions from data/jds (used for the sample-job picker)."""
    return _cached_jobs(_pdf_signature(DATA_JDS_DIR))

# ============================================================================
# LOCAL PERSISTENT BM25 CACHE  (data/bm25_index.json)
# ============================================================================
# BM25 itself (core/hybrid_indexer.py) only ever lives in RAM - it is rebuilt from
# whatever Documents are passed to index_resumes(). Re-scanning and re-reading every
# PDF in data/resumes on every app start is wasted work once the pool is large, so the
# extracted text is cached here as JSON. On startup this file - not the PDFs - is what
# gets turned back into Documents and fed to BM25. The cache only grows when the
# "Upload Resumes" button (or the sidebar's one-time indexing button) adds to it, and it
# never re-embeds anything: it just remembers text, keeping app restarts instant.

BM25_CACHE_PATH = project_root / "data" / "bm25_index.json"


def _load_bm25_cache() -> list:
    """Read the cached (filename, candidate_id, name, text) entries, oldest first."""
    if not BM25_CACHE_PATH.exists():
        return []
    try:
        return json.loads(BM25_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []   # a corrupt cache should not crash the app - it will be rebuilt


def _save_bm25_cache(entries: list) -> None:
    BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BM25_CACHE_PATH.write_text(json.dumps(entries), encoding="utf-8")


def _cache_entries_to_documents(entries: list) -> list:
    """Cache rows -> Document objects, in the SAME order they were cached.
    Order matters: HybridIndexer's BM25 result ids are positional ('c_<position>'), so
    resumes must always be appended, never re-sorted, or old positions would shift."""
    return [Document(page_content=e["text"],
                     metadata={"source": e.get("source", ""), "filename": e["filename"],
                               "candidate_id": e["candidate_id"], "name": e["name"]})
            for e in entries]


# ============================================================================
# SYSTEM INITIALIZATION (Clean, minimal)
# ============================================================================

class SystemManager:
    """Manages system components without business logic"""
    
    def __init__(self):
        self._components = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize core system components"""
        if self._initialized:
            return True
            
        try:
            # Initialize VectorStore
            vector_store = VectorStore()
            vector_store_ready = vector_store.initialize()
            
            # Create LLM
            llm = None
            try:
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    rate_limiter=get_rate_limiter(),  # max 5 requests/minute on the free tier
                )
            except Exception as e:
                st.warning(f"LLM initialization failed: {e}")
            
            # Create core components
            # Share the vector store (HybridIndexer() alone opens a second connection)
            # Fix for HW30: HybridIndexer and SearchRouter are now ALWAYS created, even
            # when Pinecone is unavailable. `vector_store` itself always exists (only its
            # readiness varies) and both HybridIndexer and VectorStore already check
            # is_ready() internally before doing anything Pinecone-specific - so BM25
            # (which needs no Pinecone at all) keeps working, and the app degrades to
            # keyword-only search instead of refusing to search at all.
            hybrid_indexer = HybridIndexer(vector_store=vector_store)
            search_router = SearchRouter(hybrid_indexer=hybrid_indexer)
            if not vector_store_ready:
                st.info("Pinecone is not available right now, so HireFlow is running in "
                        "**keyword-only (BM25) search** mode. Results will still work, but "
                        "semantic/meaning-based matching and AI re-ranking of a wider "
                        "candidate pool are unavailable until Pinecone reconnects.")
            document_processor = DocumentProcessor()
            resume_parser = ResumeParser()
            job_parser = JobParser()
            reranker = ReRanker()
            memory_rag = MemoryRAG()
            
            # Nothing is embedded to Pinecone here. Pinecone is written to ONLY when the
            # user clicks "Process & Index Resumes" (new files) or the sidebar's one-time
            # "Index existing resumes" button (files already in data/resumes) - see below.
            
            # Store components
            self._components = {
                'vector_store': vector_store,
                'hybrid_indexer': hybrid_indexer,
                'llm': llm,
                'document_processor': document_processor,
                'resume_parser': resume_parser,
                'job_parser': job_parser,
                'reranker': reranker,
                'search_router': search_router,
                'memory_rag': memory_rag,
                'vector_store_ready': vector_store_ready
            }
            
            self._initialized = True
            
            # Build the (free, local) keyword index from the persistent cache - or, the very
            # first time the app ever runs, from data/resumes - so BM25 works with zero API
            # calls and without waiting on Pinecone.
            if hybrid_indexer:
                try:
                    self.load_or_build_bm25()
                except Exception as e:
                    st.warning(f"Keyword index build failed: {e}")
            return True
            
        except Exception as e:
            st.error(f"System initialization failed: {e}")
            return False

    @staticmethod
    def _indexed_candidate_ids(vector_store):
        """candidate_ids that Pinecone already holds, or None if that cannot be read.

        Updated for the HW6/HW4 fixes in core/vector_store.py: a vector's id is now just
        'resume_<candidate_id>' (no position number - see HW6), and resumes live in their
        own namespace (see HW4), so listing must pass that namespace explicitly.
        index.list() only lists ids - no embeddings, no LLM."""
        try:
            found = set()
            for page in vector_store.index.list(prefix="resume_", namespace=RESUME_NAMESPACE):
                for vector_id in page:
                    match = re.match(r"resume_(.+)$", vector_id)
                    if match:
                        found.add(match.group(1))
            return found
        except Exception:
            return None

    @staticmethod
    def _index_keyword_only(hybrid_indexer, documents: list) -> None:
        """Run HybridIndexer.index_resumes() with Pinecone swapped out, so ONLY the free,
        in-memory BM25 model is (re)built - no embedding call is made."""
        class _KeywordOnly:
            def is_ready(self):
                return False
        real_store = hybrid_indexer.vector_store
        hybrid_indexer.vector_store = _KeywordOnly()
        try:
            hybrid_indexer.index_resumes(documents)
        finally:
            hybrid_indexer.vector_store = real_store

    def load_or_build_bm25(self) -> int:
        """Build the keyword index from the local cache; touches Pinecone never.

        - Cache file exists  -> rebuild BM25 from the cached text (instant, no PDF reads).
        - No cache yet (first run) -> read data/resumes once, build BM25, and WRITE the cache
          so every later start loads instead of re-reading the PDFs.

        Returns the number of resumes now in the keyword index.
        """
        hybrid_indexer = self._components['hybrid_indexer']
        entries = _load_bm25_cache()
        if entries:
            self._index_keyword_only(hybrid_indexer, _cache_entries_to_documents(entries))
            return len(entries)

        resumes = load_resumes(str(DATA_RESUMES_DIR))   # first run only
        if not resumes:
            return 0
        self._index_keyword_only(hybrid_indexer, resumes)
        _save_bm25_cache([{"source": d.metadata.get("source", ""), "filename": d.metadata["filename"],
                           "candidate_id": d.metadata["candidate_id"], "name": d.metadata["name"],
                           "text": d.page_content} for d in resumes])
        return len(resumes)

    def add_resumes_to_bm25_cache(self, documents: list) -> int:
        """Append new resumes to the persistent keyword cache and rebuild BM25 over all of it.

        Existing entries are never reordered or re-embedded - only appended - so a resume
        already in the cache keeps the same BM25 position. Re-uploading the same filename
        replaces its cached text instead of creating a second entry.

        Returns the new total number of resumes in the keyword index.
        """
        hybrid_indexer = self._components['hybrid_indexer']
        entries = _load_bm25_cache()
        by_id = {e["candidate_id"]: e for e in entries}
        for d in documents:
            by_id[d.metadata["candidate_id"]] = {
                "source": d.metadata.get("source", ""), "filename": d.metadata["filename"],
                "candidate_id": d.metadata["candidate_id"], "name": d.metadata["name"], "text": d.page_content}
        entries = list(by_id.values())
        _save_bm25_cache(entries)
        self._index_keyword_only(hybrid_indexer, _cache_entries_to_documents(entries))
        return len(entries)

    def index_new_resumes_to_pinecone(self, documents: list) -> dict:
        """Embed and upsert ONLY the given resumes - called once, from the Upload button.

        A resume's Pinecone id is derived from its candidate_id (stable per file name), so
        upserting is naturally idempotent - but this is only ever called with resumes that
        were just uploaded, so nothing here re-embeds a resume already in Pinecone.

        Returns: {'uploaded', 'failed'}
        """
        vector_store = self._components['vector_store']
        if not documents or not self._components.get('vector_store_ready'):
            return {"uploaded": 0, "failed": len(documents)}
        ok = vector_store.add_resumes(documents)   # one embedding call per resume, one upsert call total
        return {"uploaded": len(documents) if ok else 0, "failed": 0 if ok else len(documents)}

    def index_existing_resumes_to_pinecone(self, batch_size: int = 10) -> dict:
        """One-time bulk indexing for resumes that are already in data/resumes (e.g. the
        starter pool) but were never embedded - for the sidebar button, not run automatically.

        Compares candidate_ids on disk with the ids Pinecone already holds and embeds +
        upserts ONLY what is missing, in batches so a quota error keeps partial progress.

        Returns: {'total', 'already', 'uploaded', 'pending'}
        """
        vector_store = self._components['vector_store']
        resumes = load_resumes(str(DATA_RESUMES_DIR))
        summary = {"total": len(resumes), "already": 0, "uploaded": 0, "pending": 0}
        if not resumes:
            return summary

        already = self._indexed_candidate_ids(vector_store)
        if already is None:
            # Listing not supported: never guess by re-embedding everything. Treat a
            # non-empty index as complete and only fill an EMPTY one.
            stats = vector_store.get_stats() or {}
            everyone = {d.metadata['candidate_id'] for d in resumes}
            already = set() if not stats.get('total_vector_count') else everyone
        new_docs = [d for d in resumes if d.metadata['candidate_id'] not in already]
        summary["already"] = len(resumes) - len(new_docs)

        for start in range(0, len(new_docs), batch_size):
            batch = new_docs[start:start + batch_size]
            if not vector_store.add_resumes(batch):   # quota / network error: stop, retry later
                break
            summary["uploaded"] += len(batch)
        summary["pending"] = len(new_docs) - summary["uploaded"]

        # Anything found on disk but missing from the BM25 cache (e.g. dropped in by hand)
        # should still be searchable by keyword, so fold it into the cache too.
        self.add_resumes_to_bm25_cache(resumes)
        return summary

    def is_initialized(self) -> bool:
        return self._initialized

    def get_component(self, name: str) -> Any:
        """Get component by name"""
        if not self._initialized:
            raise RuntimeError("System not initialized")
        return self._components.get(name)
    
    def is_ready(self) -> bool:
        """Check if system is ready"""
        return self._initialized and self._components.get('vector_store_ready', False)

# ============================================================================
# UI LAYER (Only UI logic, no business logic)
# ============================================================================

class HireFlowUI:
    """Clean UI layer that delegates to core modules"""
    
    def __init__(self, system_manager: SystemManager):
        self.system_manager = system_manager
    
    def render_upload_section(self):
        """Render resume upload section"""
        st.header("Add Resumes")
        st.info("Upload PDF resumes to add them to your searchable pool. Each one gets added to "
                "**both** search indexes when you click the button below:\n"
                "- **Keyword index (BM25)** - free, local, rebuilt every time you upload.\n"
                "- **Vector index (Pinecone)** - costs one embedding call per resume, and only "
                "happens here - re-running the app later will not re-embed it.")
        
        resume_files = st.file_uploader("Select PDF Resumes", type="pdf", accept_multiple_files=True)
        
        if resume_files:
            st.success(f"Selected {len(resume_files)} resume(s)")
            if st.button("Process & Index Resumes", type="primary"):
                self.handle_resume_upload(resume_files)
    
    @staticmethod
    def _job_label(job: dict) -> str:
        return f"{job['title']} · {job['company']}" if job["company"] else job["title"]

    @staticmethod
    def _fill_form(job: dict):
        """Copy a parsed job description into the three search-form fields."""
        skills = job["tech"] or job["required"][:8]
        description = (job["intro"] + " " + " ".join(job["responsibilities"][:4])).strip() or job["text"][:1500]
        st.session_state["job_title"] = job["title"]
        st.session_state["job_description"] = description
        st.session_state["required_skills"] = "\n".join(skills)

    @staticmethod
    def _fill_from_sample():
        """on_change callback: the user picked a job from the list."""
        chosen = st.session_state.get("sample_job")
        job = next((j for j in get_jobs() if HireFlowUI._job_label(j) == chosen), None)
        if job:
            HireFlowUI._fill_form(job)

    @staticmethod
    def _fill_from_upload():
        """on_change callback: the user uploaded a job-description PDF.
        The file is saved to data/jds (so it shows up in the list from now on) and parsed."""
        file = st.session_state.get("jd_upload")
        if file is None:
            return
        try:
            DATA_JDS_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_JDS_DIR / file.name
            path.write_bytes(file.getbuffer())
            job = uh.parse_job(uh.read_pdf(path), path)
            HireFlowUI._fill_form(job)
            st.session_state["sample_job"] = HireFlowUI._job_label(job)
            st.session_state["jd_upload_error"] = ""
        except Exception as e:
            st.session_state["jd_upload_error"] = f"Could not read that PDF: {e}"

    def render_sample_job_picker(self):
        """Fill the form from a job in data/jds, or from an uploaded job-description PDF."""
        jobs = get_jobs()
        labels = ["- Write my own -"] + [self._job_label(j) for j in jobs]
        st.selectbox("Select a job description (fills the form below)", labels, key="sample_job",
                     on_change=self._fill_from_sample)
        st.file_uploader("...or upload a job description (PDF)", type="pdf", key="jd_upload",
                         on_change=self._fill_from_upload)
        if st.session_state.get("jd_upload_error"):
            st.warning(st.session_state["jd_upload_error"])
        chosen = next((j for j in jobs if self._job_label(j) == st.session_state.get("sample_job")), None)
        if chosen:
            with st.container(border=True):
                st.markdown(uh.job_card_html(chosen), unsafe_allow_html=True)

    def render_search_section(self):
        """Render candidate search section"""
        st.header("Search Candidates")
        st.info("Enter job details to find matching candidates")
        self.render_sample_job_picker()
        
        with st.form("search_form"):
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Accountant", key="job_title")
            job_description = st.text_area("Job Description", placeholder="Enter detailed job requirements...", height=100, key="job_description")
            required_skills = st.text_area("Required Skills (one per line)", placeholder="Python\nJavaScript\nReact", key="required_skills")
            debug_trace = st.checkbox(
                "🔍 Show step-by-step search trace (BM25 matches, vector matches, the merge, "
                "and the AI re-rank step)", value=False, key="debug_trace",
                help="Also turns on AI re-ranking for this search, so you can see that step "
                     "happen - this costs one extra LLM call and only runs when checked.")
            top_k = st.slider("Number of Results", 3, 10, 5)
            
            submitted = st.form_submit_button("Find Candidates", type="primary")
        
        if submitted and job_description:
            self.handle_search(job_title, job_description, required_skills, top_k, debug_trace)
    
    def render_status_sidebar(self):
        """Render system status in sidebar"""
        st.sidebar.markdown(
            f'<div class="card" style="align-items:center"><img class="logo" style="width:48px;height:48px" '
            f'src="{uh.company_uri("HireFlow")}"><div><div class="name">HireFlow</div>'
            f'<div class="title">AI candidate search</div></div></div>', unsafe_allow_html=True)
        st.sidebar.header("System Status")
        st.sidebar.write(f"**Resumes in pool:** {len(get_profiles())}")
        
        if self.system_manager.is_ready():
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            if hybrid_indexer:
                st.sidebar.write(f"**Resumes Indexed:** {len(hybrid_indexer.resume_texts)}")
            st.sidebar.write("**System:** Ready")
            # A read-only stats call (no embedding, no upsert) - safe to show on every load.
            vector_store = self.system_manager.get_component('vector_store')
            stats = vector_store.get_stats() if vector_store else {}
            if stats.get('status') == 'ready':
                st.sidebar.caption(f"Pinecone vectors: {stats.get('total_vector_count', '?')} "
                                   "(resumes + job descriptions)")
            st.sidebar.caption("Pinecone is only written to when you click an indexing button below "
                               "or upload resumes - never automatically on startup.")
            if st.sidebar.button("📥 Index existing resumes into Pinecone", use_container_width=True,
                                 help="One-time: embeds every resume in data/resumes that Pinecone doesn't have yet."):
                with st.spinner("Embedding resumes that are not yet in Pinecone..."):
                    result = self.system_manager.index_existing_resumes_to_pinecone()
                st.sidebar.success(f"{result['uploaded']} resume(s) embedded, {result['already']} were already in Pinecone.")
                if result['pending']:
                    st.sidebar.warning(f"{result['pending']} still pending (API quota) - click again later.")
        else:
            st.sidebar.write("**System:** Initializing...")
        
        # Add Memory & Evaluation section
        st.sidebar.markdown("---")
        st.sidebar.write("**Memory & Evaluation:**")
        
        memory_rag = self.system_manager.get_component('memory_rag')
        if memory_rag:
            stats = memory_rag.get_memory_stats()
            st.sidebar.write(f"• Total Interactions: {stats['total_messages']}")
            st.sidebar.write(f"• Searches: {stats['search_count']}")
            st.sidebar.write(f"• Candidate Views: {stats['candidate_views']}")
        
        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.write("**Navigation:**")
        if st.sidebar.button("📊 Memory & Evaluation", use_container_width=True):
            st.session_state.page = "memory_eval"
        if st.sidebar.button("🏠 Main Page", use_container_width=True):
            st.session_state.page = "main"
        if st.sidebar.button("👥 Talent Pool", use_container_width=True):
            st.session_state.page = "pool"
    
    def render_memory_evaluation_page(self):
        """Render Memory & Evaluation page"""
        st.header("🧠 Memory & Evaluation Dashboard")
        
        # Memory RAG Section
        st.subheader("📝 Search Memory")
        memory_rag = self.system_manager.get_component('memory_rag')
        
        if memory_rag:
            # Show memory stats
            col1, col2, col3 = st.columns(3)
            stats = memory_rag.get_memory_stats()
            
            with col1:
                st.metric("Total Interactions", stats['total_messages'])
            with col2:
                st.metric("Searches", stats['search_count'])
            with col3:
                st.metric("Candidate Views", stats['candidate_views'])
            
            # Show recent search history
            st.subheader("🔍 Recent Search History")
            search_history = memory_rag.get_search_history()
            if search_history:
                for i, query in enumerate(search_history, 1):
                    st.write(f"{i}. **{query}**")
            else:
                st.info("No search history yet. Perform some searches to see them here!")
            
            # Show recent interactions
            st.subheader("👥 Recent Interactions")
            # Fix for HW23 (core/memory_rag.py): MemoryRAG now keeps a plain list at
            # `.messages` instead of `.memory.chat_memory.messages` (the deprecated
            # ConversationBufferMemory it used to wrap).
            messages = memory_rag.messages[-10:]  # Last 10 messages
            for msg in messages:
                if hasattr(msg, 'content'):
                    if "Search:" in msg.content:
                        st.write(f"🔍 **{msg.content}**")
                    elif "Viewed candidate:" in msg.content:
                        st.write(f"👀 **{msg.content}**")
                    else:
                        st.write(f"💬 {msg.content}")
        else:
            st.warning("Memory RAG not available")
        
        # RAG Evaluation Section
        st.subheader("📊 RAG Quality Evaluation")
        st.info("Evaluate the quality of your search results using RAGAS metrics")
        
        # Simple evaluation form
        with st.form("evaluation_form"):
            eval_query = st.text_input("Query to Evaluate", placeholder="e.g., Senior Accountant with QuickBooks")
            expected_skills = st.text_area("Expected Skills (one per line)", placeholder="QuickBooks\nAccounting\nExcel")
            eval_top_k = st.slider("Number of Results to Evaluate", 3, 10, 5)
            # Fix for HW22: show the cost BEFORE running, not after - RAGAS plus the
            # ReRanker call per candidate (see core/evaluator.py's estimate_cost()).
            estimate = RAGEvaluator.estimate_cost(eval_top_k)
            st.caption(f"⏱️ Estimated cost: ~{estimate['estimated_llm_calls']} LLM calls, "
                      f"~{estimate['estimated_minutes']} min on the free tier (5 requests/minute).")
            
            if st.form_submit_button("Evaluate Search Quality"):
                if eval_query and expected_skills:
                    self.run_evaluation(eval_query, expected_skills.split('\n'), eval_top_k)
                else:
                    st.warning("Please provide both query and expected skills")
    
    def run_evaluation(self, query: str, expected_skills: list, top_k: int):
        """Run RAG evaluation"""
        try:
            # Use the app's router - a new one would search an empty index
            evaluator = RAGEvaluator(search_router=self.system_manager.get_component('search_router'))
            st.info("Running evaluation... This may take a moment.")
            
            # Clean up skills list
            skills = [s.strip() for s in expected_skills if s.strip()]
            
            # Run evaluation
            metrics = evaluator.evaluate_search_quality(query, skills, "deep", top_k)
            
            if metrics:
                st.success("Evaluation completed!")
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Answer Relevancy", f"{metrics.answer_relevancy:.2f}")
                    st.metric("Context Precision", f"{metrics.context_precision:.2f}")
                
                with col2:
                    st.metric("Faithfulness", f"{metrics.faithfulness:.2f}")
                    st.metric("Answer Correctness", f"{metrics.answer_correctness:.2f}")
                
                # Fix for HW27: overall_score is a weighted average of four 0-1 metrics,
                # so it lives in [0, 1] - not out of 10. The delta now compares against a
                # 0.5 (coin-flip) baseline, and the thresholds below are on the same scale.
                st.metric("Overall Score", f"{metrics.overall_score:.2f}", delta=f"{metrics.overall_score - 0.5:.2f}")
                
                # Fix for HW21 (core/evaluator.py): judge_failed means RAGAS could not
                # score at least one metric at all (NaN/missing) - a low number here is
                # "we don't know", not "the search is bad". Surface that distinction
                # instead of giving a confident-sounding verdict on a broken run.
                if metrics.judge_failed:
                    st.warning("⚠️ The AI judge could not score every metric (API quota, timeout, "
                              "or an unparseable reply). The numbers above may be incomplete - "
                              "this is not necessarily a search-quality problem. Try again, or "
                              "with a smaller top_k.")
                elif metrics.overall_score >= 0.8:
                    st.success("Excellent search quality! 🎉")
                elif metrics.overall_score >= 0.6:
                    st.info("Good search quality. Room for improvement.")
                else:
                    st.warning("Search quality needs improvement. Consider refining your query or adding more relevant resumes.")
            else:
                st.warning("Evaluation returned no results. Check your query and available data.")
                
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.info("This might be due to missing dependencies or configuration issues.")
    
    def handle_resume_upload(self, resume_files):
        """Save uploaded resumes, add them to the local keyword cache, and embed them in
        Pinecone - each resume touches Pinecone exactly once, right here, right now."""
        try:
            document_processor = self.system_manager.get_component('document_processor')
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            
            if not all([document_processor, hybrid_indexer]):
                st.error("System not ready for resume processing")
                return
            
            st.info(f"Processing {len(resume_files)} resume(s)...")
            
            new_documents = []
            for file in resume_files:
                try:
                    import os
                    os.makedirs(DATA_RESUMES_DIR, exist_ok=True)
                    temp_path = DATA_RESUMES_DIR / file.name
                    temp_path.write_bytes(file.getbuffer())
                    
                    # Use core DocumentProcessor to get text
                    text = document_processor.load_pdf(str(temp_path))
                    if text:
                        # Same id/name convention as core/ingestion.py, so a resume dropped
                        # into data/resumes later resolves to the exact same candidate_id.
                        new_documents.append(Document(page_content=text, metadata={
                            "source": str(temp_path), "filename": file.name,
                            "candidate_id": f"c_{temp_path.stem}",
                            "name": temp_path.stem.replace("_", " ").title()}))
                    else:
                        st.error(f"{file.name} - Could not extract text")
                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")
            
            if new_documents:
                # 1. Free, local, instant: add to the persistent keyword cache and rebuild BM25.
                with st.spinner("Updating the keyword (BM25) index..."):
                    total_keyword = self.system_manager.add_resumes_to_bm25_cache(new_documents)
                # 2. Costs an embedding call per resume: send ONLY these new resumes to Pinecone,
                #    exactly once, right now - never repeated on a later app restart.
                with st.spinner(f"Embedding {len(new_documents)} resume(s) in Pinecone..."):
                    vec = self.system_manager.index_new_resumes_to_pinecone(new_documents)
                st.session_state.flash = (
                    f"Saved {len(new_documents)} resume(s). "
                    f"✅ Keyword index (BM25): now covers {total_keyword} resumes. "
                    + (f"✅ Vector index (Pinecone): {vec['uploaded']} newly embedded." if vec['uploaded'] == len(new_documents)
                       else f"⚠️ Vector index (Pinecone): {vec['uploaded']} embedded, {vec['failed']} failed (API quota?) - "
                            "use the sidebar's 'Index existing resumes' button to retry."))
                st.rerun()
                
        except Exception as e:
            st.error(f"Resume upload failed: {e}")
    
    def handle_search(self, job_title: str, job_description: str, required_skills: str, top_k: int,
                      debug_trace: bool = False):
        """Handle candidate search using core modules"""
        try:
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            reranker = self.system_manager.get_component('reranker')
            
            if not hybrid_indexer:
                st.error("Search service not available")
                return
            
            st.session_state.search_results = None   # clear the previous search
            with st.spinner("Searching candidates..." if not debug_trace else "Searching candidates (tracing every step)..."):
                # Use core HybridIndexer for search
                skills_list = [s.strip() for s in required_skills.split('\n') if s.strip()] if required_skills else []
                search_query = f"{job_title} {' '.join(skills_list)} {job_description}"
                
                memory_rag = self.system_manager.get_component('memory_rag')
                
                # Fix for HW29: required skills used to only get pasted into the free-text
                # query (a soft hint a search engine can ignore). They are now ALSO passed
                # as a hard `filters` dict, which HybridIndexer.search_resumes() applies
                # on a pool larger than top_k so filtering never starves an already-
                # truncated result list (same approach as search_router.py's HW15 fix).
                # This needs real skills metadata on each resume - added by HW1.
                search_filters = {"skills": skills_list} if skills_list else None

                trace = None
                if debug_trace:
                    # The debug toggle routes through SearchRouter in "deep" mode instead of
                    # calling hybrid_indexer directly, so the AI re-rank step genuinely runs
                    # and can be traced - not just simulated for display. This costs one
                    # extra LLM call (the re-rank), only when the user asks to see it.
                    search_router = self.system_manager.get_component('search_router')
                    trace = {}
                    search_result = search_router.search(search_query, top_k=top_k, filters=search_filters,
                                                         search_mode="deep", trace=trace)
                    candidates_data = search_result.get('results', [])
                else:
                    candidates_data = hybrid_indexer.search_resumes(search_query, top_k=top_k, filters=search_filters)
                
                # Fix for HW28: record the search exactly ONCE, with the real result
                # count - not once with a placeholder 0 and again with the true count.
                if memory_rag:
                    memory_rag.record_search(search_query, len(candidates_data))
                
                if candidates_data:
                    st.success(f"Found {len(candidates_data)} candidates!")
                    # Fix for HW28 (continued): the old code recorded the top 3 candidates
                    # as "viewed" on every search, whether or not anyone actually looked at
                    # them. A view is now recorded in display_search_results() only when a
                    # candidate's own "Full profile..." expander is actually opened.

                    # Keep the results in session_state: every click (expanders, downloads)
                    # reruns this script, and results shown only right after submit would vanish.
                    # main() draws them full-width via render_search_results().
                    st.session_state.search_results = {
                        "candidates": candidates_data, "job_title": job_title or "Position",
                        "job_description": job_description, "skills": required_skills or "", "trace": trace}
                else:
                    st.warning("No matching candidates found")
                    
        except Exception as e:
            st.error(f"Search failed: {e}")
    
    def render_search_results(self):
        """Draw the stored search results (full width, below the form)."""
        stored = st.session_state.get("search_results")
        if stored:
            self.display_search_results(stored["candidates"], stored["job_title"],
                                        stored["job_description"], self.system_manager.get_component('reranker'),
                                        stored.get("trace"))

    def render_scoring_help(self):
        """Plain-language summary of how candidates are selected and scored."""
        with st.expander("ℹ️ How candidates are selected and scored"):
            st.markdown("""
**1. Selection - who is shown**
- Your job title and description are joined into one free-text search query.
- Required skills do double duty: they're folded into that query text, AND used as a **hard filter** - a resume must have at least one of them to appear at all.
- *Keyword search (BM25)* scores every resume by how well it matches your words. *Meaning search (vector)* asks Pinecone for resumes whose content is similar in meaning.
- Both lists are merged with **Reciprocal Rank Fusion**: each engine contributes based on a resume's *rank position* in its own results, not the raw score - so an unbounded keyword score can no longer drown out a 0-1 vector score, and a resume found by both engines is merged into one entry instead of showing up twice.
- The merged, filtered list is sorted by **Final Score**, and the top N are shown.

**2. Scores on each card**
- **Final Score** - the combined rank-fusion score described above. Higher is better, but the number itself isn't a percentage.
- **Search relevance** bar - Final Score compared with the best result on this page (best = 100%).
- **Skills found in job post** - how many of the candidate's own skills, tools and certifications (read from the PDF) appear in your job text. Green chips are the matches. This is for display only and does not change the ranking.

**3. AI Evaluation (top 3 only)**
- Gemini reads the job and the candidate's real resume text and returns a structured assessment: a fit score plus strengths, gaps, risks and a summary - the score comes directly from the model's judgment, not a formula based on counting bullet points.
- **AI Score is 0-100.** It is a second opinion: it does **not** change the ranking above.
- If the API is unavailable, a simpler rule-based fallback compares skills and years of experience instead.
""")

    def render_search_trace(self, trace: dict):
        """Render the step-by-step 'show your work' debug panel for one search.

        `trace` is filled in-place by SearchRouter.search() -> HybridIndexer.search_resumes()
        -> combine_results() / llm_rerank() (see the `trace` parameter added to each of
        those for this feature). Each key is optional, since which steps ran depends on
        the route chosen and whether an LLM/Pinecone was available.
        """
        with st.expander("🔍 Search debug trace - what happened, step by step", expanded=True):
            route = trace.get('route')
            if route:
                st.markdown(f"**1. Route chosen: `{route['chosen']}`** - {route['reason']}")

            bm25 = trace.get('bm25')
            if bm25:
                if bm25.get('ready'):
                    st.markdown(f"**2. BM25 (keyword) search** - tokenized query: "
                               f"`{', '.join(bm25['query_tokens'])}` → "
                               f"{bm25['resumes_matched']} of {bm25['resumes_searched']} resumes matched at least one word")
                    if bm25['top_matches']:
                        st.dataframe([{"Candidate": m['name'], "BM25 score": m['bm25_score']} for m in bm25['top_matches']],
                                    hide_index=True, use_container_width=True)
                else:
                    st.markdown(f"**2. BM25 (keyword) search** - skipped: {bm25.get('note', 'not ready')}")

            vector = trace.get('vector')
            if vector:
                if vector.get('ready'):
                    st.markdown("**3. Vector (meaning) search** - Pinecone matches by similarity:")
                    if vector['top_matches']:
                        st.dataframe([{"Candidate": m['name'], "Similarity (0-1)": m['similarity_score']} for m in vector['top_matches']],
                                    hide_index=True, use_container_width=True)
                    else:
                        st.caption("No vector matches returned.")
                else:
                    st.markdown(f"**3. Vector (meaning) search** - skipped: {vector.get('note', 'Pinecone not ready')}")

            merge = trace.get('merge')
            if merge:
                st.markdown("**4. Merging BM25 + vector with Reciprocal Rank Fusion**")
                st.caption(merge['method'])
                st.dataframe([{"Candidate": c['name'], "BM25 score": c['bm25_score'],
                              "Vector score": c['vector_score'], "Combined (RRF) score": c['combined_score']}
                             for c in merge['candidates']], hide_index=True, use_container_width=True)

            filters_info = trace.get('filters')
            if filters_info:
                st.markdown(f"**5. Applying required-skills filter** `{filters_info['applied']}`")
                st.caption(f"{filters_info['kept_after_filter']} of {filters_info['pool_before_filter']} candidates passed the filter.")
                if filters_info['removed']:
                    st.caption("Removed: " + ", ".join(filters_info['removed']))

            rerank = trace.get('llm_rerank')
            if rerank:
                st.markdown("**6. AI re-ranking (Gemini)**")
                if not rerank.get('attempted'):
                    st.caption(f"Skipped: {rerank.get('reason_skipped', 'not attempted')}")
                else:
                    st.caption(f"Sent {len(rerank.get('candidates_sent_to_llm', []))} candidates to the LLM, "
                              "each with a resume excerpt:")
                    for c in rerank.get('candidates_sent_to_llm', []):
                        st.caption(f"　{c['position']}. {c['name']} — *\"{c['excerpt_sent_to_llm']}\"*")
                    st.caption(f"Raw LLM reply: `{rerank.get('raw_llm_response', '')}`")
                    if rerank.get('accepted'):
                        st.success("✅ Accepted - new order: " + " → ".join(rerank.get('order_after_ai_rerank', [])))
                    else:
                        st.warning(f"⚠️ Rejected - kept the original order. Reason: {rerank.get('reason_rejected', 'unknown')}")

            final = trace.get('final') or trace.get('hybrid_final')
            if final:
                st.markdown("**7. Final order shown below**")
                st.dataframe(final, hide_index=True, use_container_width=True)

            if trace.get('error'):
                st.error(f"An error occurred during search: {trace['error']}")

    def display_search_results(self, candidates_data: list, job_title: str, job_description: str,
                               reranker: ReRanker, trace: dict = None):
        """Display search results as picture cards with full candidate details"""
        st.header(f"Top Matches for: {job_title}")
        self.render_scoring_help()
        if trace:
            self.render_search_trace(trace)
        
        index = uh.ProfileIndex(get_profiles())
        skills_text = (st.session_state.get("search_results") or {}).get("skills", "")
        job_text = f"{job_title} {job_description} {skills_text}"
        # hybrid_indexer.search_resumes() uses 'combined_score'; going through
        # SearchRouter (the debug-trace path) renames it to 'score' - accept either,
        # same defensive pattern main.py already uses for this exact inconsistency.
        get_score = lambda c: c.get('combined_score', c.get('score', 0)) or 0
        top_score = max((get_score(c) for c in candidates_data), default=0) or 1
        
        for i, candidate in enumerate(candidates_data):
            # Real details come from the parsed PDF; fall back to what the search returned
            profile = index.resolve(candidate) or uh.blank_profile(candidate)
            match = uh.skill_match(profile, job_text)
            score = get_score(candidate)
            
            with st.container(border=True):
                st.markdown(uh.card_html(i + 1, profile, highlight=match["matched"]), unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.markdown(uh.bar(100 * score / top_score, "Search relevance (vs. best result)"), unsafe_allow_html=True)
                c2.markdown(uh.bar(match["pct"], f"Skills found in job post ({len(match['matched'])}/{len(match['matched']) + len(match['other'])})"), unsafe_allow_html=True)
                if score > 0:
                    c3.metric("Final Score", f"{score:.2f}")
                
                # Fix for HW28: record a candidate view only when this candidate's own
                # expander is actually open, not automatically for the top 3 on every
                # search. The expander's `key` persists its open/closed state in
                # session_state, so we can tell whether THIS candidate was opened.
                expander_key = f"expander_{i}_{profile['id']}"
                with st.expander("Full profile, resume & AI evaluation", key=expander_key):
                    tab_overview, tab_exp, tab_resume, tab_ai = st.tabs(["Overview", "Experience", "Resume", "AI Evaluation"])
                    
                    with tab_overview:
                        left, right = st.columns([2, 1])
                        with left:
                            st.markdown("**Summary**")
                            st.write(profile["summary"] or "No summary available.")
                            st.markdown("**Matched with this job**")
                            st.markdown(uh.chips(match["matched"], "match") or "_No overlap found_", unsafe_allow_html=True)
                            st.markdown("**All technical skills**")
                            st.markdown(uh.chips(profile["skills"], "skill", highlight=match["matched"]), unsafe_allow_html=True)
                            st.markdown("**Software**")
                            st.markdown(uh.chips(profile["software"], "sw", highlight=match["matched"]), unsafe_allow_html=True)
                            if profile["certs"]:
                                st.markdown("**Certifications**")
                                st.markdown(uh.chips(profile["certs"], "cert", highlight=match["matched"]), unsafe_allow_html=True)
                        with right:
                            st.markdown("**Contact**")
                            for label, value in (("Email", profile["email"]), ("Phone", profile["phone"]), ("LinkedIn", profile["linkedin"])):
                                if value:
                                    st.write(f"{label}: {value}")
                            edu = profile["education"]
                            if edu["degree"]:
                                st.markdown("**Education**")
                                st.write(f"{edu['degree']}")
                                st.caption(f"{edu['school']} · {edu['year']}" + (f" · GPA {edu['gpa']}" if edu['gpa'] else ""))
                            st.markdown("**Experience fit**")
                            st.write(f"{profile['years']}+ years · {profile['level']} level")
                    
                    with tab_exp:
                        st.markdown(uh.timeline_html(profile["jobs"]), unsafe_allow_html=True)
                    
                    with tab_resume:
                        pdf_path = Path(profile["file"]) if profile["file"] else None
                        if pdf_path and pdf_path.exists():
                            st.download_button("Download resume (PDF)", pdf_path.read_bytes(), file_name=pdf_path.name,
                                               mime="application/pdf", key=f"dl_{i}_{profile['id']}")
                        text = profile["text"] or candidate.get('text', '')
                        if text:
                            st.text(text[:1500] + "..." if len(text) > 1500 else text)
                    
                    with tab_ai:
                        # AI evaluation for top candidates using core ReRanker
                        if i < 3 and reranker:
                            self.display_ai_evaluation(candidate, job_title, job_description, reranker)
                        else:
                            st.caption("AI evaluation runs for the top 3 candidates.")

                # Fix for HW28: this expander's own widget state tells us whether the user
                # actually opened THIS candidate - record the view once, the first time
                # it's seen open (a `viewed` set guards against re-recording on every
                # rerun while it stays open).
                if st.session_state.get(expander_key):
                    viewed = st.session_state.setdefault("viewed_candidates", set())
                    if profile["id"] not in viewed:
                        viewed.add(profile["id"])
                        memory_rag = self.system_manager.get_component('memory_rag')
                        if memory_rag:
                            memory_rag.record_candidate_view(profile["name"])
    
    @staticmethod
    def _extract_contact_details(text: str) -> dict:
        """Fix for HW25: pull a real email/phone/degree out of the resume text with
        simple regexes, instead of hard-coding the same fake contact details for every
        candidate. Falls back to an obviously generic placeholder only when genuinely
        nothing is found (these three fields are required by the Resume schema, but are
        never sent to the LLM - see the comment at the call site)."""
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text or "")
        phone = re.search(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", text or "")
        degree = re.search(r"(Bachelor|Master|Associate|MBA|PhD)[^\n]{0,60}", text or "", re.IGNORECASE)
        return {
            "email": email.group(0) if email else "unknown@example.com",
            "phone": phone.group(0) if phone else "Not provided",
            "education": degree.group(0).strip() if degree else "Not specified",
        }

    def display_ai_evaluation(self, candidate: dict, job_title: str, job_description: str, reranker: ReRanker):
        """Display AI evaluation using core ReRanker"""
        try:
            # Create objects for core ReRanker
            jd = JobDescription(
                jd_id=f"jd_{job_title.lower().replace(' ', '_')}",
                title=job_title or "Position",
                text=job_description
            )
            
            # Fix for HW25: every candidate used to be sent to the AI judge with the exact
            # same invented email, phone, 5 years of experience and "Bachelor's Degree" -
            # so the AI was really judging one fictional person, 50 times over. This now
            # pulls real experience/skills from the search result (real since the HW1 fix
            # in core/ingestion.py), and extracts a real email/phone/degree line from the
            # candidate's actual resume text where possible - with a clearly generic
            # fallback only for the (schema-required) fields that genuinely aren't there.
            # None of this is sent to the LLM anyway - ReRanker's prompt only uses
            # jd.text and resume.text (see core/re_ranker.py) - but a wrong number
            # shouldn't be presented as this candidate's real contact information.
            contact = self._extract_contact_details(candidate.get('text', ''))
            metadata = candidate.get('metadata', {}) or {}
            resume = Resume(
                candidate_id=candidate.get('candidate_id', 'unknown'),
                name=candidate.get('name', 'Unknown'),
                email=contact['email'],
                phone=contact['phone'],
                experience=candidate.get('experience') or metadata.get('experience') or 0,
                skills=candidate.get('skills') or metadata.get('skills', []),
                education=contact['education'],
                text=candidate.get('text', '')
            )
            
            # Use core ReRanker for evaluation
            # Results are redrawn on every Streamlit rerun; cache the LLM verdict per
            # candidate + job so a click does not spend another request (5/minute limit).
            cache = st.session_state.setdefault("ai_eval_cache", {})
            cache_key = (resume.candidate_id, job_title, hash(job_description))
            if cache_key not in cache:
                cache[cache_key] = reranker.evaluate_candidate(resume, jd)
            evaluation = cache[cache_key]
            if evaluation:
                eval_data = evaluation.model_dump()
                fit_score = eval_data.get('fit_score', 0)
                
                st.markdown("**AI Evaluation:**")
                # Fix for HW26: CandidateEvaluation.fit_score runs 0-100 (see
                # utils/schemas.py and the structured-output fix for HW16 in
                # core/re_ranker.py), not 0-10. Thresholds and the label are now on the
                # correct scale, so a 65 shows as "65/100" (info) instead of "65/10"
                # (which used to render as a false "success").
                if fit_score >= 80:
                    st.success(f"**AI Score: {fit_score}/100**")
                elif fit_score >= 60:
                    st.info(f"**AI Score: {fit_score}/100**")
                else:
                    st.warning(f"**AI Score: {fit_score}/100**")
                
                # Show strengths and gaps from core evaluation
                strengths = eval_data.get('strengths', [])
                if strengths:
                    st.write("**Strengths:**")
                    for s in strengths[:2]:
                        st.write(f"• {s}")
                
                gaps = eval_data.get('gaps', [])
                if gaps:
                    st.write("**Gaps:**")
                    for g in gaps[:2]:
                        st.write(f"• {g}")
                
                summary = eval_data.get('summary', '')
                if summary:
                    st.write("**Summary:**")
                    st.write(summary)
                    
        except Exception as e:
            st.info("AI evaluation not available")


# ============================================================================
# EXTRA UI: hero banner, KPI strip and talent-pool page
# ============================================================================

def render_hero():
    """Banner with the app title (generated SVG background, no image file needed)."""
    st.markdown(
        f'<div class="hf-hero" style="background-image:url(\'{uh.hero_uri()}\')">'
        f'<h1>HireFlow - AI Resume Search</h1>'
        f'<p>Find the right candidate faster: keyword + semantic search, AI fit evaluation and quality metrics in one place.</p>'
        f'<span class="pill">Hybrid BM25 + Vector search</span><span class="pill">AI fit scoring</span>'
        f'<span class="pill">RAGAS evaluation</span></div>', unsafe_allow_html=True)


def render_kpi_strip(profiles: list):
    """Headline numbers plus a row of profile pictures from the talent pool."""
    if not profiles:
        st.info("No resumes found yet - upload PDFs below to build your talent pool.")
        return
    avg_years = sum(p["years"] for p in profiles) / len(profiles)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Candidates in pool", len(profiles))
    k2.metric("Open job templates", len(get_jobs()))
    k3.metric("Avg. experience", f"{avg_years:.1f} yrs")
    k4.metric("Locations", len({p["location"] for p in profiles if p["location"]}))
    st.markdown(uh.avatar_strip(profiles, 14), unsafe_allow_html=True)


def render_talent_pool_page():
    """Browse every candidate: filters, insight charts and a picture gallery."""
    import pandas as pd
    st.header("👥 Talent Pool")
    profiles = get_profiles()
    if not profiles:
        st.info("No resumes found in data/resumes.")
        return
    render_kpi_strip(profiles)
    
    f1, f2, f3, f4 = st.columns(4)
    query = f1.text_input("Search name or skill", placeholder="e.g., QuickBooks")
    roles = f2.multiselect("Role", sorted({p["title"] for p in profiles}))
    levels = f3.multiselect("Level", ["Entry", "Mid", "Senior", "Leadership"])
    places = f4.multiselect("Location", sorted({p["location"] for p in profiles if p["location"]}))
    
    shown = [p for p in profiles
             if (not roles or p["title"] in roles) and (not levels or p["level"] in levels)
             and (not places or p["location"] in places)
             and (not query or query.lower() in (p["name"] + " " + " ".join(uh.skill_terms(p))).lower())]
    st.caption(f"Showing {len(shown)} of {len(profiles)} candidates")
    
    with st.expander("📊 Talent insights", expanded=False):
        df = pd.DataFrame(shown)
        if not df.empty:
            a, b = st.columns(2)
            a.markdown("**Candidates by role**")
            a.bar_chart(df["title"].value_counts())
            b.markdown("**Candidates by experience level**")
            b.bar_chart(df["level"].value_counts())
            c, d = st.columns(2)
            c.markdown("**Top locations**")
            c.bar_chart(df["location"].value_counts().head(8))
            d.markdown("**Most common skills**")
            skills = pd.Series([sk for p in shown for sk in p["skills"]]).value_counts().head(10)
            d.bar_chart(skills)
    
    per_row = 4
    for start in range(0, len(shown), per_row):
        cols = st.columns(per_row)
        for col, p in zip(cols, shown[start:start + per_row]):
            with col.container(border=True):
                st.markdown(uh.gallery_card_html(p), unsafe_allow_html=True)
                with st.popover("View details", use_container_width=True):
                    st.markdown(f"**{p['name']}** · {p['title']}")
                    st.caption(f"{p['email']} · {p['phone']}")
                    st.write(p["summary"])
                    st.markdown(uh.chips(p["skills"], "skill") + uh.chips(p["software"], "sw") + uh.chips(p["certs"], "cert"),
                                unsafe_allow_html=True)
                    edu = p["education"]
                    if edu["degree"]:
                        st.caption(f"🎓 {edu['degree']}, {edu['school']} ({edu['year']})")
                    if p["file"] and Path(p["file"]).exists():
                        st.download_button("Download resume", Path(p["file"]).read_bytes(), file_name=p["filename"],
                                           mime="application/pdf", key=f"pool_dl_{p['id']}")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

@st.cache_resource(show_spinner="Starting HireFlow (checking which resumes are already indexed)...")
def get_system_manager() -> SystemManager:
    """Streamlit reruns this whole script on EVERY click.

    Without caching, each click would rebuild the system and re-embed every
    resume - with a 5 requests/minute limit, a few clicks would break the app.
    @st.cache_resource runs this once and reuses the result on every rerun.
    """
    manager = SystemManager()
    manager.initialize()
    return manager


def main():
    if st.session_state.get('page', 'main') == 'main':
        render_hero()
    else:
        st.title("HireFlow")

    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    # Build the system ONCE (see get_system_manager above)
    system_manager = get_system_manager()
    if not system_manager.is_initialized():
        st.error("System initialization failed. Please check configuration.")
        return

    # Create UI with dependency injection
    ui = HireFlowUI(system_manager)

    # Render status sidebar (always visible)
    ui.render_status_sidebar()

    # Render page based on navigation state
    if st.session_state.get('page') == 'memory_eval':
        ui.render_memory_evaluation_page()
    elif st.session_state.get('page') == 'pool':
        render_talent_pool_page()
    else:
        flash = st.session_state.pop('flash', None)
        if flash:
            st.success(flash)
        render_kpi_strip(get_profiles())
        st.divider()
        # Render main page UI components
        col1, col2 = st.columns([1, 1])

        with col1:
            ui.render_upload_section()

        with col2:
            ui.render_search_section()
        
        # Results use the full page width (cards + tabs need more room than a column)
        ui.render_search_results()

if __name__ == "__main__":
    main()
