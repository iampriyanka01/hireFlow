# core/hybrid_indexer.py

"""
core/hybrid_indexer.py  -  HYBRID SEARCH (BM25 keywords + vector meaning)
=========================================================================

Why hybrid?
    - BM25 (keyword search) is great at exact terms: "QuickBooks", "CPA".
      It cannot understand synonyms or intent.
    - Vector search (semantic) understands meaning: "bookkeeper" ~ "accountant".
      It can miss rare exact keywords.
    Running both and merging the lists gives better candidates than either alone.

How a query flows through search_resumes()
    query ─┬─> BM25 (in memory)   ─> a score for EVERY resume
           └─> VectorStore/Pinecone ─> top_k*2 nearest resumes
                          │
                          └─> combine_results() ─> one ranked top_k list

What lives where
    self.bm25_resumes / bm25_jds : in-memory BM25 indexes (rebuilt each run)
    self.resume_texts / jd_texts : lowercased texts, IN THE SAME ORDER as the
                                   BM25 index - position i in the BM25 scores
                                   refers to resume_texts[i]
    self.vector_store            : shared VectorStore (Pinecone, persistent)

Public API
    index_resumes(docs) / index_job_descriptions(docs)   -> build both indexes
    search_resumes(query, top_k) / search_job_descriptions(...)
    combine_results(...)  -> merge step (public for testing)
    get_index_stats()     -> readiness flags for diagnostics

Used by
    SearchRouter (deep search) and main.py.
"""

import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from core.vector_store import VectorStore
from utils.utils import get_logger

logger = get_logger(__name__)

# Fix for HW11: a shared tokenizer used at BOTH index time and query time, so BM25 sees
# the same vocabulary on both sides. Splitting on non-alphanumerics means 'Excel,' and
# 'Excel' become the same token, and dropping stopwords stops common words like 'the' or
# 'with' from diluting the score when the query is a whole job description.
_STOPWORDS = frozenset("""
    a an the and or but if is are was were be been being to of in on for with as at by
    from this that these those it its it's you your our we they he she his her them
    will would can could should shall may might must not no do does did doing have has
    had having i me my mine
""".split())


def _tokenize(text: str) -> List[str]:
    """Fix for HW11: lowercase, split on non-alphanumeric characters, drop stopwords
    and single characters. Used for both the BM25 corpus and every BM25 query."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOPWORDS and len(t) > 1]


def passes_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Fix for HW15/HW29: apply a hard filter to a result's metadata. Mirrors the shapes
    VectorStore.convert_filters_to_pinecone() understands, so the same filters dict works
    whether search goes through Pinecone, BM25, or both:
        {"location": "Austin"}              -> exact match
        {"location": ["Austin", "Dallas"]}  -> membership
        {"experience": {"min": 3}}          -> range (min and/or max)
    Skills are matched case-insensitively and any ONE required skill counts as a match,
    since candidates rarely have every requested skill.
    """
    for key, condition in filters.items():
        value = metadata.get(key)
        if key == "skills":
            have = {str(s).lower() for s in (value or [])}
            wanted = condition if isinstance(condition, (list, tuple, set)) else [condition]
            if not any(str(w).lower() in have for w in wanted):
                return False
        elif isinstance(condition, dict):
            if value is None:
                return False
            if "min" in condition and value < condition["min"]:
                return False
            if "max" in condition and value > condition["max"]:
                return False
        elif isinstance(condition, (list, tuple, set)):
            if value not in condition:
                return False
        else:
            if value != condition:
                return False
    return True

class HybridIndexer:
    """Owns the BM25 indexes and coordinates them with the VectorStore."""

    def __init__(self, vector_store: VectorStore = None):
        """Initialize both BM25 and vector search components.

        Pass in an existing VectorStore so every component shares one
        connection. If none is given, a new one is created as before.
        """
        if vector_store is None:
            vector_store = VectorStore()
            vector_store.initialize()   # Set up Pinecone vector store
        self.vector_store = vector_store
        # BM25 models are None until index_*() is called; search_*() returns []
        # while they are None.
        self.bm25_resumes = None        # BM25 index for resumes
        self.bm25_jds = None           # BM25 index for job descriptions
        self.resume_texts = []         # lowercased text, aligned with resume_docs and BM25
        self.jd_texts = []             # lowercased text, aligned with jd_docs and BM25
        # Fix for HW9/HW12: keep the actual Documents too (not just lowercased strings),
        # in the SAME order as resume_texts/jd_texts, so a BM25 result can carry the
        # resume's real metadata (name, skills, location, experience - see HW1) instead
        # of inventing one. index_resumes()/index_job_descriptions() now MERGE new
        # documents into these lists by id rather than replacing them (fix for HW12).
        self.resume_docs: List[Document] = []
        self.jd_docs: List[Document] = []
    
    def index_resumes(self, resumes: List[Document]) -> bool:
        """Build the BM25 index AND upload the resumes to the vector store.

        Steps:
          1. MERGE the given resumes into whatever was already indexed (fix for HW12)
          2. lowercase each resume text (BM25 is case-insensitive here)
          3. tokenize and build a BM25Okapi model (fix for HW11)
          4. if Pinecone is ready, embed + upsert the given resumes

        Returns:
            True on success, False if the list is empty or an error occurred.
        """
        if not resumes:
            return False
        
        try:
            # Fix for HW12: merge by candidate_id instead of replacing the whole index.
            # A dict preserves insertion order, so a resume that was already indexed keeps
            # its position (its BM25 corpus index doesn't shift); only a genuinely new
            # candidate_id is appended. Indexing one freshly-uploaded resume no longer
            # erases every resume indexed before it.
            existing = {d.metadata.get('candidate_id', d.metadata.get('filename', str(i))): d
                       for i, d in enumerate(self.resume_docs)}
            for doc in resumes:
                key = doc.metadata.get('candidate_id', doc.metadata.get('filename'))
                existing[key] = doc
            # Keep resume_docs and resume_texts filtered/aligned the same way, so
            # resume_texts[i] and resume_docs[i] always describe the same resume.
            self.resume_docs = [d for d in existing.values() if d.page_content.strip()]
            self.resume_texts = [d.page_content.lower() for d in self.resume_docs]
            
            if self.resume_texts:
                # Fix for HW11: tokenize on non-alphanumerics and drop stopwords (see
                # _tokenize above), instead of .split() which keeps punctuation attached
                # and keeps words like 'the'/'with' that add noise to a JD-length query.
                tokenized_texts = [_tokenize(text) for text in self.resume_texts]
                self.bm25_resumes = BM25Okapi(tokenized_texts)
            
            # Add to vector store if available. Only the resumes passed to THIS call are
            # embedded - Pinecone upsert is idempotent by id (fix for HW6), so this never
            # re-embeds resumes that were merged in from a previous call.
            if self.vector_store.is_ready():
                self.vector_store.add_resumes(resumes)
            
            return True
                
        except Exception as e:
            logger.error(f"Resume indexing failed: {e}")
            return False
    
    def index_job_descriptions(self, job_descriptions: List[Document]) -> bool:
        """Build the BM25 index AND upload the job descriptions to the vector store.

        Same merge + tokenize treatment as index_resumes() (fixes for HW11/HW12), applied
        to job postings.
        """
        if not job_descriptions:
            return False
        
        try:
            existing = {d.metadata.get('jd_id', d.metadata.get('filename', str(i))): d
                       for i, d in enumerate(self.jd_docs)}
            for doc in job_descriptions:
                key = doc.metadata.get('jd_id', doc.metadata.get('filename'))
                existing[key] = doc
            self.jd_docs = [d for d in existing.values() if d.page_content.strip()]
            self.jd_texts = [d.page_content.lower() for d in self.jd_docs]
            
            if self.jd_texts:
                tokenized_texts = [_tokenize(text) for text in self.jd_texts]
                self.bm25_jds = BM25Okapi(tokenized_texts)
            
            # Add to vector store if available
            if self.vector_store.is_ready():
                self.vector_store.add_job_descriptions(job_descriptions)
            
            return True
                
        except Exception as e:
            logger.error(f"Job description indexing failed: {e}")
            return False
    
    def search_resumes(self, query: str, top_k: int = 5,
                       filters: Optional[Dict[str, Any]] = None,
                       trace: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search resumes with BM25 + vector search and return a merged top_k list.

        1. BM25:   tokenise the query (fix for HW11), score every indexed resume.
        2. Vector: ask Pinecone for 2*top_k resumes (filter type='resume') so
                   the merge step has enough candidates to choose from.
        3. Merge:  combine_results() (Reciprocal Rank Fusion - fixes HW7/HW8/HW9/HW10).
        4. Filter: apply `filters` (e.g. required skills/experience) on a larger pool
                   than top_k, so filtering never starves an already-truncated list
                   (fix for HW15/HW29).

        Args:
            filters: optional hard filter, e.g. {"skills": ["QuickBooks"], "experience":
                     {"min": 3}}. See passes_filters() for the accepted shapes.
            trace:   optional dict. When given, this method (and combine_results())
                     fill it in-place with a step-by-step record of what happened -
                     the BM25 hits, the vector hits, the RRF merge, and the effect of
                     any filter - for a debugging/"show your work" UI. Leave as None
                     (the default) to skip this bookkeeping entirely.

        Returns:
            list of candidate dicts (candidate_id, text, name, skills, location,
            experience, metadata, combined_score, bm25_score, vector_score).
            [] if the BM25 index has not been built or an error occurs.
        """
        if not self.bm25_resumes:
            if trace is not None:
                trace['bm25'] = {'ready': False, 'note': 'BM25 index has not been built yet'}
                trace['vector'] = {'ready': False, 'note': 'search skipped - no BM25 index'}
            return []
        
        try:
            # Fix for HW11: tokenize the query the same way the corpus was tokenized.
            query_tokens = _tokenize(query)
            bm25_scores = self.bm25_resumes.get_scores(query_tokens)

            if trace is not None:
                # Same "only ranked if score > 0" rule combine_results() uses (HW10),
                # so this preview matches what actually feeds into the merge below.
                hits = sorted(((i, s) for i, s in enumerate(bm25_scores) if s > 0),
                              key=lambda pair: pair[1], reverse=True)
                trace['bm25'] = {
                    'ready': True,
                    'query_tokens': query_tokens,
                    'resumes_matched': len(hits),
                    'resumes_searched': len(bm25_scores),
                    'top_matches': [
                        {'name': self.resume_docs[i].metadata.get('name', 'Unknown'), 'bm25_score': round(float(s), 3)}
                        for i, s in hits[:10] if i < len(self.resume_docs)
                    ]
                }
            
            # Vector search (if available) - with type filter for resumes only
            vector_results = []
            if self.vector_store.is_ready():
                # Add filter to only get resume documents
                vector_filters = {"type": "resume"}
                vector_results = self.vector_store.search_resumes(query, top_k * 2, vector_filters)
                logger.info(f"Vector search returned {len(vector_results)} results for query: {query[:50]}...")

            if trace is not None:
                trace['vector'] = {
                    'ready': self.vector_store.is_ready(),
                    'top_matches': [
                        {'name': r.get('metadata', {}).get('name', 'Unknown'), 'similarity_score': round(r.get('score', 0.0), 3)}
                        for r in vector_results[:10]
                    ]
                }
            
            # Fix for HW15/HW29: fetch a larger merged pool than top_k when a filter is
            # given, so filtering happens BEFORE truncating - not after, which could
            # throw away every result that would have passed the filter.
            pool_size = max(top_k * 4, 20) if filters else top_k
            merged = self.combine_results(bm25_scores, vector_results, pool_size, is_jd=False, trace=trace)
            if filters:
                kept = [r for r in merged if passes_filters(r.get('metadata', r), filters)]
                if trace is not None:
                    kept_ids = {r.get('candidate_id') for r in kept}
                    trace['filters'] = {
                        'applied': filters,
                        'pool_before_filter': len(merged),
                        'kept_after_filter': len(kept),
                        'removed': [r.get('name') for r in merged if r.get('candidate_id') not in kept_ids][:10]
                    }
                merged = kept
            result = merged[:top_k]
            if trace is not None:
                trace['hybrid_final'] = [
                    {'name': r.get('name'), 'combined_score': round(r.get('combined_score', 0.0), 4)}
                    for r in result
                ]
            return result
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            if trace is not None:
                trace['error'] = str(e)
            return []
    
    def search_job_descriptions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search job descriptions with BM25 + vector search (mirror of search_resumes)."""
        if not self.bm25_jds:
            return []
        
        try:
            # Fix for HW11: same shared tokenizer as search_resumes()/index_job_descriptions().
            query_tokens = _tokenize(query)
            bm25_scores = self.bm25_jds.get_scores(query_tokens)
            
            # Vector search (if available) - with type filter for job descriptions only
            vector_results = []
            if self.vector_store.is_ready():
                # Add filter to only get job description documents
                filters = {"type": "job_description"}
                vector_results = self.vector_store.search_job_descriptions(query, top_k * 2, filters)
            
            # Combine results
            return self.combine_results(bm25_scores, vector_results, top_k, is_jd=True)
                
        except Exception as e:
            logger.error(f"JD search failed: {e}")
            return []
    
    def combine_results(self, bm25_scores: List[float], vector_results: List[Dict], 
                        top_k: int, is_jd: bool,
                        trace: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Merge BM25 and vector results into ONE ranked list with Reciprocal Rank Fusion.

        Args:
            bm25_scores:    one BM25 score per indexed document (same order as
                            resume_texts/resume_docs or jd_texts/jd_docs).
            vector_results: matches returned by VectorStore.search_*() (already sorted
                            best-first by Pinecone).
            top_k:          how many results to return.
            is_jd:          True to build job-shaped dicts, False for candidates.

        This replaces the old "concatenate both lists and sort by raw score" merge,
        fixing four related problems (former TODOs HW7/HW8/HW9/HW10):

          HW7  - BM25 scores are unbounded (5-15+) while cosine scores sit in 0-1, so
                 sorting them together let BM25 always dominate. Reciprocal Rank Fusion
                 (RRF) instead combines RANK POSITIONS: each engine that finds a document
                 contributes 1/(60+rank) to its combined_score, so neither engine's scale
                 can drown out the other.
          HW8  - The same person no longer appears twice. Both engines' results are
                 merged into ONE dict keyed by candidate_id/jd_id before ranking, so a
                 candidate found by both BM25 and the vector search gets one entry whose
                 score reflects both signals.
          HW9  - BM25-only results now carry the resume/JD's REAL metadata (name, skills,
                 location, experience - see HW1), read from self.resume_docs/jd_docs,
                 instead of a name guessed from the first (lowercased) line of text and
                 skills=[]/location='Unknown'/experience=5 for every single candidate.
          HW10 - A document that shares NO word with the query (BM25 score 0) is no
                 longer turned into a ranked result at all - it's dropped before RRF, so
                 it can no longer pad the results or crowd out a genuine vector match.
        """
        RRF_K = 60  # standard RRF constant - keeps a single #1 rank from dominating
        id_field = 'jd_id' if is_jd else 'candidate_id'
        docs = self.jd_docs if is_jd else self.resume_docs

        def blank_entry(entry_id: str, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
            """One result row, in the same shape the rest of the app already expects."""
            base = {'text': text, 'combined_score': 0.0, 'bm25_score': 0.0, 'vector_score': 0.0,
                   'metadata': metadata}
            if is_jd:
                base.update({'jd_id': entry_id, 'title': metadata.get('title', 'Unknown Job')})
            else:
                base.update({'candidate_id': entry_id, 'name': metadata.get('name', 'Unknown Candidate'),
                            'skills': metadata.get('skills', []), 'location': metadata.get('location', 'Unknown'),
                            'experience': metadata.get('experience', 0)})
            return base

        merged: Dict[str, Dict[str, Any]] = {}

        # --- BM25 side: only documents that actually share a query word get ranked (HW10) ---
        bm25_hits = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
        bm25_hits.sort(key=lambda pair: pair[1], reverse=True)
        for rank, (i, score) in enumerate(bm25_hits, start=1):
            if i >= len(docs):
                continue
            doc = docs[i]
            entry_id = doc.metadata.get(id_field) or f"{'jd' if is_jd else 'c'}_{i}"
            entry = merged.setdefault(entry_id, blank_entry(entry_id, doc.page_content, doc.metadata))
            entry['bm25_score'] = float(score)
            entry['combined_score'] += 1.0 / (RRF_K + rank)   # RRF fix for HW7

        # --- Vector side: Pinecone already returns best-first, so list position = rank ---
        for rank, vec_result in enumerate(vector_results, start=1):
            vmeta = vec_result.get('metadata', {}) or {}
            entry_id = vmeta.get(id_field) or f"{'jd' if is_jd else 'c'}_vec_{rank}"
            # HW8 fix: reuses the SAME id as the BM25 side when both engines found this
            # document, merging into the one entry instead of creating a duplicate.
            entry = merged.get(entry_id) or blank_entry(entry_id, vec_result.get('page_content', ''), vmeta)
            merged[entry_id] = entry
            entry['vector_score'] = vec_result.get('score', 0.0)
            entry['combined_score'] += 1.0 / (RRF_K + rank)   # RRF fix for HW7

        results = sorted(merged.values(), key=lambda r: r['combined_score'], reverse=True)

        if trace is not None and not is_jd:
            # Show the RRF arithmetic per candidate, so it's clear why one resume
            # outranked another even when neither score alone looks dominant.
            trace['merge'] = {
                'method': f'Reciprocal Rank Fusion (k={RRF_K}) - combines RANK POSITION from '
                          'each engine, not raw score, so BM25\'s unbounded scores can\'t '
                          'drown out the vector engine\'s 0-1 scores.',
                'candidates': [
                    {'name': r.get('name'), 'bm25_score': round(r['bm25_score'], 3),
                     'vector_score': round(r['vector_score'], 3), 'combined_score': round(r['combined_score'], 4)}
                    for r in results[:15]
                ]
            }

        return results[:top_k]
    def get_index_stats(self) -> Dict[str, Any]:
        """Return readiness flags: resumes_ready, jds_ready, vector_store_ready, hybrid_ready.

        `hybrid_ready` is True only when the resume BM25 index exists AND the
        vector store is connected - i.e. both halves of hybrid search work.
        """
        return {
            'resumes_ready': bool(self.bm25_resumes),
            'jds_ready': bool(self.bm25_jds),
            'vector_store_ready': self.vector_store.is_ready(),
            'hybrid_ready': bool(self.bm25_resumes) and self.vector_store.is_ready()
        }
