# core/hybrid_indexer.py

"""
Hybrid search system combining BM25 (lexical) and vector (semantic) search.
Provides comprehensive candidate and job matching capabilities.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from core.vector_store import VectorStore
from utils.utils import get_logger

logger = get_logger(__name__)

class HybridIndexer:
    """Combines BM25 keyword search with vector semantic search for better results"""

    def __init__(self, vector_store: VectorStore = None):
        """Initialize both BM25 and vector search components.

        Pass in an existing VectorStore so every component shares one
        connection. If none is given, a new one is created as before.
        """
        if vector_store is None:
            vector_store = VectorStore()
            vector_store.initialize()   # Set up Pinecone vector store
        self.vector_store = vector_store
        self.bm25_resumes = None        # BM25 index for resumes
        self.bm25_jds = None           # BM25 index for job descriptions
        # TODO: HW12 [Medium] - Uploading a resume erases the others
        # index_resumes() starts again from an empty list. The Streamlit upload calls it
        # with ONE new resume. After an upload, how many resumes can BM25 still find?
        # Add a way to append documents to the existing index.
        self.resume_texts = []         # Text content for BM25 resume search
        self.jd_texts = []             # Text content for BM25 JD search
    
    def index_resumes(self, resumes: List[Document]) -> bool:
        """Index resumes for both keyword and semantic search"""
        if not resumes:
            return False
        
        try:
            # Prepare texts for BM25
            self.resume_texts = []
            for resume in resumes:
                text = resume.page_content.lower()
                if text.strip():
                    self.resume_texts.append(text)
            
            if self.resume_texts:
                # Build BM25 model
                # TODO: HW11 [Easy] - Tokenization
                # .split() keeps punctuation attached, so 'Excel,' and 'Excel' are different
                # tokens. It also keeps words like 'the' and 'with'. When the query is a whole
                # job description, what happens to the BM25 scores?
                # Tokenize on non-alphanumerics and drop stopwords - for the query as well.
                # Hint: Session 4 - BM25, TF-IDF, stopwords.
                tokenized_texts = [text.split() for text in self.resume_texts]
                self.bm25_resumes = BM25Okapi(tokenized_texts)
            
            # Add to vector store if available
            if self.vector_store.is_ready():
                self.vector_store.add_resumes(resumes)
            
            return True
                
        except Exception as e:
            logger.error(f"Resume indexing failed: {e}")
            return False
    
    def index_job_descriptions(self, job_descriptions: List[Document]) -> bool:
        """Index job descriptions for both keyword and semantic search"""
        if not job_descriptions:
            return False
        
        try:
            # Prepare texts for BM25
            self.jd_texts = []
            for jd in job_descriptions:
                text = jd.page_content.lower()
                if text.strip():
                    self.jd_texts.append(text)
            
            if self.jd_texts:
                # Build BM25 model
                tokenized_texts = [text.split() for text in self.jd_texts]
                self.bm25_jds = BM25Okapi(tokenized_texts)
            
            # Add to vector store if available
            if self.vector_store.is_ready():
                self.vector_store.add_job_descriptions(job_descriptions)
            
            return True
                
        except Exception as e:
            logger.error(f"Job description indexing failed: {e}")
            return False
    
    def search_resumes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search resumes using both BM25 and vector search, then combine results"""
        if not self.bm25_resumes:
            return []
        
        try:
            # BM25 search
            query_tokens = query.lower().split()
            bm25_scores = self.bm25_resumes.get_scores(query_tokens)
            
            # Vector search (if available) - with type filter for resumes only
            vector_results = []
            if self.vector_store.is_ready():
                # Add filter to only get resume documents
                filters = {"type": "resume"}
                vector_results = self.vector_store.search_resumes(query, top_k * 2, filters)
                logger.info(f"Vector search returned {len(vector_results)} results for query: {query[:50]}...")
            
            # Combine results
            return self.combine_results(bm25_scores, vector_results, top_k, is_jd=False)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_job_descriptions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search job descriptions using both BM25 and vector search, then combine results"""
        if not self.bm25_jds:
            return []
        
        try:
            # BM25 search
            query_tokens = query.lower().split()
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
                        top_k: int, is_jd: bool) -> List[Dict[str, Any]]:
        """Merge BM25 keyword and vector semantic results into unified ranked list"""
        results = []
        
        # Get metadata based on type
        metadata_list = self.jd_texts if is_jd else self.resume_texts
        
        # Process BM25 results - create normalized results
        bm25_results = []
        # TODO: HW10 [Easy] - Irrelevant resumes pad the results
        # Every resume gets a BM25 result here, including those scoring 0 - meaning
        # not one query word matched. What should happen to zero-score documents?
        for i, score in enumerate(bm25_scores):
            if i < len(metadata_list):
                # Create consistent metadata for BM25 results
                if is_jd:
                    bm25_results.append({
                        'jd_id': f"jd_{i}",
                        'text': self.jd_texts[i],
                        'title': f"Job {i+1}",
                        'combined_score': float(score),
                        'bm25_score': float(score),
                        'vector_score': 0.0,
                        'source': 'bm25'
                    })
                else:
                    # For resumes
                    resume_text = self.resume_texts[i]
                    lines = resume_text.split('\n')
                    # TODO: HW9 [Medium] - Keyword results carry invented values
                    # Look at what BM25 results report: a name taken from the first line of
                    # LOWERCASED text, skills=[], location='Unknown', experience=5 for everyone.
                    # These are placeholders, and the re-ranker and UI treat them as facts.
                    # Keep the original Documents (not just lowercased text) so every result
                    # carries the resume's real metadata. Related: HW1.
                    name = lines[0][:50] if lines else f"Candidate {i+1}"
                    
                    bm25_results.append({
                        # TODO: HW8 [Medium] - The same person can appear twice
                        # BM25 results get ids like 'c_3'. Vector results get the real candidate_id,
                        # like 'c_Kevin_Torres_Resume_36'. The same resume found by both engines ends up
                        # as two different results. Use one id for both, then merge them.
                        'candidate_id': f"c_{i}",
                        'text': resume_text,
                        'name': name.strip(),
                        'skills': [],
                        'location': 'Unknown',
                        'experience': 5,
                        'combined_score': float(score),
                        'bm25_score': float(score),
                        'vector_score': 0.0,
                        'source': 'bm25'
                    })
        
        # Process vector results - create normalized results  
        vector_formatted = []
        for vec_result in vector_results:
            if is_jd:
                jd_id = vec_result.get('metadata', {}).get('jd_id', f"jd_vec_{len(vector_formatted)}")
                vector_formatted.append({
                    'jd_id': jd_id,
                    'text': vec_result.get('page_content', ''),
                    'title': vec_result.get('metadata', {}).get('title', 'Unknown Job'),
                    'combined_score': vec_result.get('score', 0.0),
                    'bm25_score': 0.0,
                    'vector_score': vec_result.get('score', 0.0),
                    'source': 'vector'
                })
            else:
                # For resumes
                candidate_id = vec_result.get('metadata', {}).get('candidate_id', f"c_vec_{len(vector_formatted)}")
                vector_formatted.append({
                    'candidate_id': candidate_id,
                    'text': vec_result.get('page_content', ''),
                    'name': vec_result.get('metadata', {}).get('name', 'Unknown Candidate'),
                    'skills': vec_result.get('metadata', {}).get('skills', []),
                    'location': vec_result.get('metadata', {}).get('location', 'Unknown'),
                    'experience': vec_result.get('metadata', {}).get('experience', 5),
                    'combined_score': vec_result.get('score', 0.0),
                    'bm25_score': 0.0,
                    'vector_score': vec_result.get('score', 0.0),
                    'source': 'vector'
                })
        
        # Combine all results
        all_results = bm25_results + vector_formatted
        
        # Sort by combined score and return top results
        # TODO: HW7 [Hard] - Hybrid search is really keyword search
        # BM25 scores are unbounded (often 5-15). Cosine scores sit between 0 and 1.
        # This sorts both kinds of score together. Run main.py and compare the
        # scores in '1. Vector Search' with '2. Hybrid Search'. Can a vector result
        # ever reach the top of this list?
        # Replace this with Reciprocal Rank Fusion: score = sum of 1 / (60 + rank).
        # Hint: Session 4 - hybrid search and RRF. Master Recap section 3.6.
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top_k results, removing 'source' field
        for result in all_results[:top_k]:
            result.pop('source', None)
            results.append(result)
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get status of BM25 and vector search components"""
        return {
            'resumes_ready': bool(self.bm25_resumes),
            'jds_ready': bool(self.bm25_jds),
            'vector_store_ready': self.vector_store.is_ready(),
            'hybrid_ready': bool(self.bm25_resumes) and self.vector_store.is_ready()
        }
