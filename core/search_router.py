
"""
core/search_router.py  -  THE SEARCH "FRONT DOOR"
=================================================

Role in the pipeline
    Every user search goes through SearchRouter.search(). The router decides
    HOW MUCH work a query deserves and runs the matching strategy:

        query ──> route_search() ──┬─> "shallow" : vector search only
                                   │              (fast, 1 embedding call)
                                   └─> "deep"    : hybrid BM25+vector search
                                                   + LLM re-ranking of results

    search_mode values
        "shallow" / "deep" : forced by the caller
        "auto"             : the LLM classifies the query; if the LLM is
                             unavailable or answers oddly, a word-count rule
                             is used (<= 3 words -> shallow, else deep)

    Fallbacks (so a search rarely returns nothing)
        shallow fails / vector store not ready -> tries deep
        deep fails                             -> tries shallow
        Each side falls back only ONCE (the `_fell_back` flag prevents an
        endless shallow <-> deep loop).

Implementation
    The strategies are wired together with LangChain's `RunnableBranch`
    (an if / elif / else for runnables) inside build_search_chain().

Output format of search()
    {"results": [ {candidate_id, name, text, score, metadata,
                   search_type, source_query, ...}, ... ],
     "search_type": "shallow_vector" | "deep_hybrid_reranked" | "failed",
     "query": ..., "top_k": ...}

Used by
    main.py demos, RAGEvaluator (which searches and then scores the results),
    and the Streamlit UI.
"""

from typing import Dict, Any, List, Optional
import re
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from core.hybrid_indexer import HybridIndexer
from core.vector_store import VectorStore
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.utils import get_logger, get_rate_limiter

logger = get_logger(__name__)

class SearchRouter:
    """Chooses between shallow (vector) and deep (hybrid + LLM rerank) search."""
    
    def __init__(self, hybrid_indexer: HybridIndexer = None):
        """Wire the router to an existing HybridIndexer and prepare the LLM.

        Pass the SAME indexer that was filled with resumes; otherwise the router
        would search an empty private index. The vector store is taken from that
        indexer so all components share one Pinecone connection.

        If GOOGLE_API_KEY is missing (or the client fails to start) `self.llm`
        stays None: auto-routing then uses the word-count rule and deep search
        skips LLM re-ranking.
        """
        # Use the indexer that main.py / app.py already filled with resumes.
        # A new HybridIndexer here would be empty, so every search would return nothing.
        self.hybrid_indexer = hybrid_indexer if hybrid_indexer is not None else HybridIndexer()
        # Share its vector store. A fresh VectorStore() is never initialized,
        # so is_ready() would always be False.
        self.vector_store = self.hybrid_indexer.vector_store
        
        self.llm = None
        if GOOGLE_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    rate_limiter=get_rate_limiter(),  # max 5 requests/minute on the free tier
                )
            except Exception as e:
                pass
        
        self.search_chain = self.build_search_chain()
    
    def build_search_chain(self):
        """Define the two strategies plus the routing rule and combine them in a RunnableBranch.

        The three inner functions below are closures over `self`:
          shallow_search(inputs) - vector-only search
          deep_search(inputs)    - hybrid search + optional LLM re-ranking
          route_search(inputs)   - returns "shallow" or "deep"
        `inputs` is a dict: query, top_k, filters, search_mode, jd_context.
        """
        
        def shallow_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Fast path: one embedding call + one Pinecone query (no BM25, no LLM).

            Falls back to deep_search if the vector store is not ready or an
            error occurs (only once - see `_fell_back`).
            """
            query = inputs["query"]
            top_k = inputs.get("top_k", 5)
            filters = inputs.get("filters")
            trace = inputs.get("trace")
            
            try:
                if self.vector_store.is_ready():
                    results = self.vector_store.search_resumes(query, top_k, filters)
                    
                    if trace is not None:
                        trace['bm25'] = {'ready': False, 'note': 'skipped - shallow mode only uses vector search'}
                        trace['vector'] = {
                            'ready': True,
                            'top_matches': [
                                {'name': r.get('metadata', {}).get('name', 'Unknown'), 'similarity_score': round(r.get('score', 0.0), 3)}
                                for r in results[:10]
                            ]
                        }
                        trace['llm_rerank'] = {'attempted': False, 'reason_skipped': 'shallow mode does not re-rank'}
                    
                    # Format results for consistency
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'candidate_id': result.get('metadata', {}).get('candidate_id', 'unknown'),
                            'name': result.get('metadata', {}).get('name', 'Unknown'),
                            'text': result.get('page_content', ''),
                            'score': result.get('score', 0.0),
                            'metadata': result.get('metadata', {}),
                            'search_type': 'shallow_vector',
                            'source_query': query
                        })
                    
                    if trace is not None:
                        trace['final'] = [{'name': r['name'], 'score': round(r['score'], 4)} for r in formatted_results]
                    
                    return {
                        "results": formatted_results,
                        "search_type": "shallow_vector",
                        "query": query,
                        "top_k": top_k
                    }
                elif not inputs.get("_fell_back"):
                    return deep_search(inputs)
                else:
                    return {"results": [], "search_type": "failed", "query": query, "top_k": top_k}

            except Exception as e:
                logger.error(f"Shallow search failed: {e}")
                if inputs.get("_fell_back"):
                    return {"results": [], "search_type": "failed", "query": query, "top_k": top_k}
                return deep_search(inputs)
        
        def deep_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Thorough path: BM25 + vector search, then LLM re-ranking.

            1. hybrid_indexer.search_resumes()  -> merged candidate list
            2. if an LLM is available, llm_rerank() reorders the list
            3. results are reshaped into the common output format
            On error it falls back to shallow_search (only once).
            """
            query = inputs["query"]
            top_k = inputs.get("top_k", 5)
            filters = inputs.get("filters")
            jd_context = inputs.get("jd_context", "")
            trace = inputs.get("trace")
            
            try:
                # Fix for HW15: a filter EXCLUDES, a score RANKS - they must not be applied
                # in the wrong order. Ask HybridIndexer for a pool larger than top_k when
                # there is a filter (it applies the filter before truncating internally -
                # see HW29's fix there), so filtering here can never leave fewer than
                # top_k results when enough candidates would have passed.
                results = self.hybrid_indexer.search_resumes(query, top_k, filters=filters, trace=trace)
                
                if results and self.llm:
                    # Fix for HW14: only rerank as many candidates as the prompt can
                    # sensibly hold, and pass a resume excerpt (see llm_rerank below).
                    rerank_pool = results[:max(top_k * 2, 10)]
                    reranked = self.llm_rerank(rerank_pool, query, jd_context, top_k, trace=trace)
                    results = reranked + results[len(rerank_pool):]
                elif trace is not None:
                    trace['llm_rerank'] = {
                        'attempted': False,
                        'reason_skipped': 'no results to rerank' if not results else 'LLM not configured (missing GOOGLE_API_KEY)'
                    }
                
                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'candidate_id': result.get('candidate_id', 'unknown'),
                        'name': result.get('name', 'Unknown'),
                        'text': result.get('text', ''),
                        'score': result.get('combined_score', 0.0),
                        'metadata': result.get('metadata', {}),
                        'search_type': 'deep_hybrid_reranked',
                        'source_query': query,
                        'query_count': result.get('query_count', 1),
                        'llm_reranked': 'llm_reranked' in result
                    })
                
                if trace is not None:
                    trace['final'] = [
                        {'name': r['name'], 'score': round(r['score'], 4), 'llm_reranked': r['llm_reranked']}
                        for r in formatted_results
                    ]
                
                return {
                    "results": formatted_results,
                    "search_type": "deep_hybrid_reranked",
                    "query": query,
                    "top_k": top_k
                }
                
            except Exception as e:
                logger.error(f"Deep search failed: {e}")
                # Fall back to shallow search - once. Without the flag, shallow and
                # deep kept falling back to each other forever (RecursionError).
                return shallow_search({**inputs, "_fell_back": True})
        
        def route_search(inputs: Dict[str, Any]) -> str:
            """Return "shallow" or "deep" for this query.

            Priority: (1) an explicit search_mode from the caller,
                      (2) the LLM's classification in "auto" mode,
                      (3) a word-count heuristic (<= 3 words -> shallow).
            """
            query = inputs.get("query", "")
            search_mode = inputs.get("search_mode", "auto")
            trace = inputs.get("trace")
            
            if search_mode == "shallow":
                if trace is not None:
                    trace['route'] = {'requested_mode': search_mode, 'chosen': 'shallow', 'reason': 'forced by caller'}
                return "shallow"
            elif search_mode == "deep":
                if trace is not None:
                    trace['route'] = {'requested_mode': search_mode, 'chosen': 'deep', 'reason': 'forced by caller'}
                return "deep"
            
            if self.llm:
                try:
                    routing_prompt = f"""
                    Analyze this search query and determine if it needs deep search or shallow search.
                    
                    Query: "{query}"
                    
                    Choose:
                    - "shallow" for simple, specific queries (e.g., "accountant", "Python developer")
                    - "deep" for complex, nuanced queries (e.g., "senior accountant with QuickBooks and tax experience")
                    
                    Return only "shallow" or "deep".
                    """
                    
                    response = self.llm.invoke(routing_prompt)
                    decision = response.content.strip().lower()
                    
                    if decision in ["shallow", "deep"]:
                        if trace is not None:
                            trace['route'] = {'requested_mode': search_mode, 'chosen': decision,
                                              'reason': 'LLM classified the query'}
                        return decision
                    else:
                        logger.warning(f"Routing LLM returned an unrecognized decision: {decision!r}")
                        
                except Exception as e:
                    logger.warning(f"Routing LLM call failed, falling back to word-count rule: {e}")
            
            query_words = len(query.split())
            chosen = "shallow" if query_words <= 3 else "deep"
            if trace is not None:
                trace['route'] = {'requested_mode': search_mode, 'chosen': chosen,
                                  'reason': f'word-count heuristic ({query_words} words, threshold is 3)'}
            return chosen
        
        def decide_route(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Fix for HW13: RunnableBranch evaluates its conditions in order, and each
            condition used to call route_search() itself - which calls the LLM. For a
            'deep' query, that meant TWO LLM calls just to choose a route (one per
            condition checked before the match), doubling the cost of every search. This
            runs route_search() exactly ONCE and stores the result in the inputs, so the
            branch below only ever reads a plain dict value - no extra LLM calls.
            """
            return {**inputs, "_route": route_search(inputs)}

        chain = RunnableLambda(decide_route) | RunnableBranch(
            (lambda x: x["_route"] == "shallow", shallow_search),
            (lambda x: x["_route"] == "deep", deep_search),
            deep_search  # Default fallback
        )
        
        return chain
    
    def llm_rerank(self, results: List[Dict], query: str, jd_context: str, top_k: int,
                   trace: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Ask the LLM to reorder candidates by fit for the query / job.

        The prompt lists the candidates numbered 1..N and the LLM replies with
        the new order (e.g. "3,1,2"). The reply is parsed with a regex. The
        new order is accepted ONLY if it contains exactly N numbers; otherwise
        (or on any error) the original order is returned unchanged.
        Reordered items get llm_reranked=True and llm_rank=<position>.

        Args:
            trace: optional dict. When given, filled with exactly what was sent to
                   the LLM (each candidate's excerpt), its raw reply, how that reply
                   was parsed, and whether the new order was accepted or rejected -
                   so a debugging UI can show precisely what the AI step did.
        """
        if not results or not self.llm:
            if trace is not None:
                trace['llm_rerank'] = {'attempted': False,
                                       'reason_skipped': 'no results' if not results else 'LLM not configured'}
            return results
        
        try:
            candidates_info = []
            candidates_sent = []
            for i, result in enumerate(results):
                candidate_info = f"{i+1}. {result.get('name', 'Unknown')} - "
                candidate_info += f"Score: {result.get('combined_score', 0):.4f}"
                
                metadata = result.get('metadata', {})
                if metadata.get('skills'):
                    candidate_info += f", Skills: {', '.join(metadata['skills'][:5])}"
                if metadata.get('experience'):
                    candidate_info += f", Experience: {metadata['experience']} years"
                if metadata.get('location'):
                    candidate_info += f", Location: {metadata['location']}"

                # Fix for HW14: a recruiter (human or AI) cannot judge fit from a name and
                # a retrieval score alone. Now that skills/experience are real (HW1/HW9),
                # also include a short excerpt of the actual resume text, so the LLM has
                # something concrete to reason about beyond the numbers above.
                excerpt = (result.get('text', '') or '').strip().replace('\n', ' ')[:300]
                if excerpt:
                    candidate_info += f"\n   Resume excerpt: {excerpt}..."
                
                candidates_info.append(candidate_info)
                if trace is not None:
                    candidates_sent.append({'position': i + 1, 'name': result.get('name', 'Unknown'),
                                            'excerpt_sent_to_llm': excerpt[:200] + ('...' if len(excerpt) > 200 else '')})
            
            rerank_prompt = f"""
            You are an expert recruiter. Given a job search query and candidate results, rerank the candidates by their fit for the role.
            
            Search Query: "{query}"
            Job Context: "{jd_context if jd_context else 'General search'}"
            
            Current Results (ranked by algorithm):
            {chr(10).join(candidates_info)}
            
            Instructions:
            1. Consider the search query and job context
            2. Reorder the candidates by their fit for the role
            3. Return only the numbers in the new order (e.g., "3,1,2,4,5")
            4. Focus on semantic fit, not just scores
            
            Reranked order:
            """
            
            response = self.llm.invoke(rerank_prompt)
            reranked_order = response.content.strip()

            if trace is not None:
                trace['llm_rerank'] = {
                    'attempted': True,
                    'candidates_sent_to_llm': candidates_sent,
                    'raw_llm_response': reranked_order,
                }
            
            try:
                numbers = [int(x) for x in re.findall(r'\d+', reranked_order)]
                
                if len(numbers) == len(results):
                    reranked_results = []
                    for num in numbers:
                        if 1 <= num <= len(results):
                            result = results[num - 1].copy()
                            result['llm_reranked'] = True
                            result['llm_rank'] = len(reranked_results) + 1
                            reranked_results.append(result)
                    
                    if trace is not None:
                        trace['llm_rerank']['parsed_order'] = numbers
                        trace['llm_rerank']['accepted'] = True
                        trace['llm_rerank']['order_after_ai_rerank'] = [r.get('name') for r in reranked_results]
                    
                    return reranked_results
                else:
                    if trace is not None:
                        trace['llm_rerank']['parsed_order'] = numbers
                        trace['llm_rerank']['accepted'] = False
                        trace['llm_rerank']['reason_rejected'] = (
                            f"LLM returned {len(numbers)} numbers but there are {len(results)} "
                            "candidates - keeping the original order instead of guessing.")
                    
            except Exception as e:
                if trace is not None:
                    trace['llm_rerank']['accepted'] = False
                    trace['llm_rerank']['reason_rejected'] = f"couldn't parse the LLM's reply: {e}"
            
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            if trace is not None:
                trace['llm_rerank'] = {'attempted': True, 'accepted': False, 'reason_rejected': f"LLM call failed: {e}"}
        
        return results
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None, 
               search_mode: str = "auto", jd_context: str = "",
               trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Public entry point: run one search through the routing chain.

        Args:
            query:       what the recruiter typed.
            top_k:       max number of candidates to return.
            filters:     optional metadata filters (used by shallow search).
            search_mode: "auto" (default), "shallow" or "deep".
            jd_context:  optional job-description text to help LLM re-ranking.
            trace:       optional dict. Pass an empty {} to have this search fill it
                         in-place with a step-by-step record (route chosen, BM25 hits,
                         vector hits, the RRF merge, any filter, and the AI re-rank
                         step if one ran) - for a "show your work" debugging UI.
                         Leave as None (the default) to skip this bookkeeping.

        Returns:
            dict described at the top of this file. On an unexpected error an
            empty result with search_type="fallback" and an "error" message.
        """
        
        inputs = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "search_mode": search_mode,
            "jd_context": jd_context,
            "trace": trace
        }
        
        try:
            result = self.search_chain.invoke(inputs)
            return result
            
        except Exception as e:
            # Fallback to simple search
            return {
                "results": [],
                "search_type": "fallback",
                "query": query,
                "top_k": top_k,
                "error": str(e)
            }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Return availability flags (LLM, vector store, hybrid index) and the supported modes."""
        return {
            "llm_available": self.llm is not None,
            "vector_store_ready": self.vector_store.is_ready(),
            "hybrid_indexer_ready": self.hybrid_indexer.get_index_stats()['hybrid_ready'],
            "search_modes": ["auto", "shallow", "deep"]
        }
