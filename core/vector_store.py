"""
core/vector_store.py  -  SEMANTIC SEARCH with Pinecone
======================================================

Role in the pipeline
    Stores one embedding vector per document in Pinecone and finds the
    documents whose meaning is closest to a query ("semantic search").

    Document text ─> Google embedding model ─> vector (list of floats)
                                              ─> Pinecone upsert
    Query text    ─> Google embedding model ─> vector ─> Pinecone query
                                              ─> top_k nearest documents

What gets stored in Pinecone for each document
    id        : "resume_<candidate_id>" or "jd_<jd_id>" (stable - re-indexing the
                same file overwrites its vector instead of duplicating it; fix for HW6)
    namespace : "resumes" or "job_descriptions" - kept in separate namespaces so a
                resume search can never return a job posting (fix for HW4)
    values    : the embedding vector
    metadata  : the Document metadata + 'page_content' (full text) + 'type'
               ('resume' or 'job_description'). Keeping the text in metadata
               means search results need no second lookup.

Lifecycle
    vs = VectorStore()
    vs.initialize()        # connect, make sure index exists, load embeddings
    vs.add_resumes(docs)   # embed + upsert
    vs.search_resumes(...) # embed query + similarity search

Configuration (utils/config.py)
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_DIMENSION, PINECONE_METRIC

Used by
    HybridIndexer (semantic half of hybrid search), SearchRouter (shallow
    search), and main.py (direct vector search + diagnostics).
"""

import time
from typing import Dict, Any, List
from pinecone import Pinecone, ServerlessSpec       
from utils.utils import get_embeddings
from utils.config import (
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
)
from utils.utils import get_logger, is_quota_error

logger = get_logger(__name__)

# Fix for HW4: resumes and job descriptions live in separate Pinecone namespaces so a
# resume search can never accidentally return a job posting (or vice versa), even
# without an explicit metadata filter.
RESUME_NAMESPACE = "resumes"
JD_NAMESPACE = "job_descriptions"


class VectorStore:
    """Thin wrapper around a Pinecone index plus the embedding model.

    Every public method returns a safe value (False / [] / status dict)
    instead of raising, so the rest of the app keeps running when Pinecone or
    the embedding API is unavailable. Check `is_ready()` first.
    """
    
    def __init__(self):
        """Initialize Pinecone client and components"""
        self.client = None       # Pinecone client instance
        self.index = None        # Pinecone index for vector operations
        self.embeddings = None   # Google embedding model
        self._ready = False      # Ready status flag
        
    def initialize(self) -> bool:
        """Connect to Pinecone and get everything ready for use.

        Order of operations:
          1. create the Pinecone client from the API key
          2. ensure_index()  - create the index if it is missing
          3. load the Google embedding model
          4. open a handle to the index
        Only after all four succeed is `_ready` set to True.

        Returns:
            True on success, False on any failure (the reason is logged).
        """
        try:
            if not PINECONE_API_KEY:
                logger.warning("PINECONE_API_KEY not set")
                return False
            
            # Create Pinecone client first
            self.client = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("Pinecone client created successfully")
            
            # Ensure index exists
            if not self.ensure_index():
                logger.error("Failed to ensure index exists")
                return False
            
            # Get embeddings
            self.embeddings = get_embeddings()
            if not self.embeddings:
                logger.error("Failed to get embeddings")
                return False
            
            # Get the index
            self.index = self.client.Index(PINECONE_INDEX_NAME)
            logger.info("Vector store initialization completed successfully")
            
            self._ready = True
            return True
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            self._ready = False
            return False
    
    def ensure_index(self, allow_destructive_recreate: bool = False) -> bool:
        """Make sure the Pinecone index exists with the configured dimension.

        - index exists, same dimension  -> nothing to do
        - index exists, wrong dimension -> refuses by default (fix for HW5, below)
        - index missing                 -> created as a serverless index
                                           (AWS, us-east-1)
        A short sleep waits for Pinecone to finish creating/deleting.

        Args:
            allow_destructive_recreate: must be explicitly True to delete and recreate
                an index whose dimension doesn't match. Defaults to False so a config
                change can never silently wipe a production index (see HW5).
        """
        try:
            if not self.client:
                logger.error("Pinecone client not initialized")
                return False
            
            # Check if index exists
            existing_indexes = [index.name for index in self.client.list_indexes()]
            if PINECONE_INDEX_NAME in existing_indexes:
                # Get index info
                index_info = self.client.describe_index(PINECONE_INDEX_NAME)
                current_dimension = index_info.dimension
                
                if current_dimension == PINECONE_DIMENSION:
                    logger.info(f"Index {PINECONE_INDEX_NAME} already exists with correct dimensions")
                    return True
                else:
                    logger.warning(f"Index dimension mismatch: {current_dimension} vs {PINECONE_DIMENSION}")
                    # Fix for HW5: deleting and recreating on every mismatch is convenient
                    # while developing, and destroys every stored vector in production. Now
                    # this only happens if the caller explicitly opts in; otherwise we fail
                    # loudly so a human decides (e.g. re-point PINECONE_DIMENSION at the
                    # existing index's real size instead of deleting real data).
                    if not allow_destructive_recreate:
                        logger.error(
                            f"Index '{PINECONE_INDEX_NAME}' has dimension {current_dimension}, "
                            f"but PINECONE_DIMENSION is set to {PINECONE_DIMENSION}. Refusing to "
                            "delete the existing index. Either update PINECONE_DIMENSION to match "
                            "it, or call ensure_index(allow_destructive_recreate=True) if you "
                            "really want to wipe and recreate it."
                        )
                        return False
                    self.client.delete_index(PINECONE_INDEX_NAME)
                    logger.info(f"Deleted index {PINECONE_INDEX_NAME} due to dimension mismatch")
                    time.sleep(5)  # Wait for deletion
            else:
                logger.info(f"Index {PINECONE_INDEX_NAME} not found, creating new one")
            
            # Create new index
            from pinecone import ServerlessSpec
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
            self.client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric=PINECONE_METRIC,
                spec=spec
            )
            logger.info(f"Created new index {PINECONE_INDEX_NAME}")
            time.sleep(10)  # Wait for index to be ready
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure index: {e}")
            return False
    
    def is_ready(self) -> bool:
        """True once initialize() has completed successfully."""
        return self._ready
    
    def add_resumes(self, documents: List[Any]) -> bool:
        """Embed resume Documents and upsert them into Pinecone.

        For each Document: copy its metadata, add the full text as
        'page_content' and tag it type='resume'. Upsert means "insert or
        overwrite by id", so re-running with the same ids does not duplicate.

        Returns:
            True if the upsert succeeded, False otherwise (incl. quota errors).
        """
        if not self._ready or not self.index:
            return False
        
        try:
            # Fix for HW3: one embedding call for the WHOLE batch instead of one call per
            # resume (50 resumes = 50 API calls -> 1 call). embed_documents() already
            # accepts a list and returns the vectors in the same order.
            all_values = self.embeddings.embed_documents([doc.page_content for doc in documents])

            # Prepare data for Pinecone
            vectors = []
            for doc, values in zip(documents, all_values):
                # Create metadata with page content
                metadata = doc.metadata.copy()
                metadata['page_content'] = doc.page_content
                metadata['type'] = 'resume'
                
                vectors.append({
                    # Fix for HW6: the id is derived only from candidate_id, not from the
                    # document's position in the list, so re-indexing the same resume (in
                    # any order) overwrites its existing vector instead of duplicating it.
                    "id": f"resume_{metadata.get('candidate_id', 'unknown')}",
                    "values": values,
                    "metadata": metadata
                })
            
            # Fix for HW4: resumes go in their own namespace, so a resume search can
            # never return a job posting even without an explicit type filter.
            self.index.upsert(vectors=vectors, namespace=RESUME_NAMESPACE)
            return True
        except Exception as e:
            if is_quota_error(e):
                logger.warning("API quota exceeded - indexing skipped")
            else:
                logger.error(f"Failed to add resumes: {e}")
            return False
    
    def add_job_descriptions(self, documents: List[Any]) -> bool:
        """Embed job-description Documents and upsert them (tagged type='job_description').

        Identical to add_resumes() except for the id prefix ("jd_") and type tag.
        """
        if not self._ready or not self.index:
            return False
        
        try:
            # Fix for HW3: batch-embed all job descriptions in one call (see add_resumes).
            all_values = self.embeddings.embed_documents([doc.page_content for doc in documents])

            # Prepare data for Pinecone
            vectors = []
            for doc, values in zip(documents, all_values):
                # Create metadata with page content
                metadata = doc.metadata.copy()
                metadata['page_content'] = doc.page_content
                metadata['type'] = 'job_description'
                
                vectors.append({
                    # Fix for HW6: id derived only from jd_id, not list position (see add_resumes).
                    "id": f"jd_{metadata.get('jd_id', 'unknown')}",
                    "values": values,
                    "metadata": metadata
                })
            
            # Fix for HW4: job descriptions get their own namespace (see add_resumes).
            self.index.upsert(vectors=vectors, namespace=JD_NAMESPACE)
            return True
        except Exception as e:
            if is_quota_error(e):
                logger.warning("API quota exceeded - indexing skipped")
            else:
                logger.error(f"Failed to add job descriptions: {e}")
            return False
    
    def search_resumes(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find the resumes most similar in meaning to `query`.

        Args:
            query:   free-text search, e.g. "senior accountant with QuickBooks".
            top_k:   number of matches to return.
            filters: optional metadata filter, e.g. {"type": "resume"}
                     (converted by convert_filters_to_pinecone()).

        Returns:
            list of {'page_content': str, 'metadata': dict, 'score': float}
            sorted best-first. 'score' is the similarity (higher = closer).
            Returns [] on any error.
        """
        if not self._ready or not self.index:
            return []
        
        try:
            # Fix for HW2: embed_query() is the correct method for a search query.
            # embed_documents() and embed_query() can apply different Google "task type"
            # hints (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY internally); using the document
            # embedder for a query throws away that asymmetric-search optimization.
            query_vector = self.embeddings.embed_query(query)
            
            # Perform similarity search. Fix for HW4: query the resumes namespace, so a
            # resume search physically cannot return a job description.
            if filters:
                pinecone_filters = self.convert_filters_to_pinecone(filters)
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=pinecone_filters,
                    namespace=RESUME_NAMESPACE,
                    include_metadata=True
                )
            else:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    namespace=RESUME_NAMESPACE,
                    include_metadata=True
                )
            
            # Check if results exist and have matches
            if not results or not hasattr(results, 'matches') or not results.matches:
                logger.warning(f"No results or matches found for query: {query}")
                return []
            
            return [
                {
                    'page_content': item.metadata.get('page_content', ''),
                    'metadata': item.metadata,
                    'score': item.score
                }
                for item in results.matches
            ]
        except Exception as e:
            if is_quota_error(e):
                logger.warning("API quota exceeded - search skipped")
            else:
                logger.error(f"Resume search failed: {e}")
            return []
    
    def search_job_descriptions(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find the job descriptions most similar in meaning to `query`.

        Same behaviour and return format as search_resumes().
        """
        if not self._ready or not self.index:
            return []
        
        try:
            # Fix for HW2 (continued): use embed_query(), same reasoning as search_resumes().
            query_vector = self.embeddings.embed_query(query)
            
            # Fix for HW4: query the job_descriptions namespace, mirroring search_resumes().
            if filters:
                pinecone_filters = self.convert_filters_to_pinecone(filters)
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=pinecone_filters,
                    namespace=JD_NAMESPACE,
                    include_metadata=True
                )
            else:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    namespace=JD_NAMESPACE,
                    include_metadata=True
                )
            
            # Check if results exist and have matches
            if not results or not hasattr(results, 'matches') or not results.matches:
                logger.warning(f"No results or matches found for query: {query}")
                return []
            
            return [
                {
                    'page_content': item.metadata.get('page_content', ''),
                    'metadata': item.metadata,
                    'score': item.score
                }
                for item in results.matches
            ]
        except Exception as e:
            if is_quota_error(e):
                logger.warning("API quota exceeded - search skipped")
            else:
                logger.error(f"Job description search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics for diagnostics.

        Returns a dict with status ('ready' / 'not_ready' / 'error') and, when
        ready, total_vector_count, dimension and metric.
        """
        if not self._ready or not self.index:
            return {"status": "not_ready"}
        
        try:
            stats = self.index.describe_index_stats()

            def field(name):
                # the stats response may be an object or a dict, depending on client version
                if isinstance(stats, dict):
                    return stats.get(name)
                return getattr(stats, name, None)

            return {
                "status": "ready",
                "total_vector_count": field("total_vector_count"),
                "dimension": field("dimension"),
                "metric": field("metric")
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "message": str(e)}
    
    def convert_filters_to_pinecone(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Translate simple Python filters into Pinecone's filter syntax.

        Examples:
            {"type": "resume"}                -> {"type": {"$eq": "resume"}}
            {"location": ["Austin", "Dallas"]} -> {"location": {"$in": [...]}}
            {"experience": {"min": 3}}        -> {"experience": {"$gte": 3}}
            {"experience": {"min": 3, "max": 8}} -> {"$gte": 3, "$lte": 8}
        """
        pinecone_filters = {}
        
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                pinecone_filters[key] = {"$eq": value}
            elif isinstance(value, list):
                pinecone_filters[key] = {"$in": value}
            elif isinstance(value, dict):
                if "min" in value and "max" in value:
                    pinecone_filters[key] = {
                        "$gte": value["min"],
                        "$lte": value["max"]
                    }
                elif "min" in value:
                    pinecone_filters[key] = {"$gte": value["min"]}
                elif "max" in value:
                    pinecone_filters[key] = {"$lte": value["max"]}
        
        return pinecone_filters
