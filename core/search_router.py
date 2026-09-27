
"""
Intelligent search routing system using LangChain RunnableBranch.
Routes queries to shallow or deep search based on complexity and LLM analysis.
"""

from typing import Dict, Any, List, Optional
import re
from langchain_core.runnables import RunnableBranch
from langchain_google_genai import ChatGoogleGenerativeAI
from core.hybrid_indexer import HybridIndexer
from core.vector_store import VectorStore
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.utils import get_logger, get_rate_limiter

logger = get_logger(__name__)

class SearchRouter:
    """Routes search queries to appropriate search strategy (shallow vs deep)"""
    
    def __init__(self, hybrid_indexer: HybridIndexer = None):
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
        """Build the RunnableBranch search chain"""
        
        def shallow_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Fast vector-only search"""
            query = inputs["query"]
            top_k = inputs.get("top_k", 5)
            filters = inputs.get("filters")
            
            pass
            
            try:
                if self.vector_store.is_ready():
                    results = self.vector_store.search_resumes(query, top_k, filters)
                    
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
            """Comprehensive hybrid search with multi-query and reranking"""
            query = inputs["query"]
            top_k = inputs.get("top_k", 5)
            filters = inputs.get("filters")
            jd_context = inputs.get("jd_context", "")
            
            pass
            
            try:
                # Perform hybrid search
                # TODO: HW15 [Easy] - Deep search ignores filters
                # deep_search reads 'filters' from the inputs and never uses them.
                # Where should a hard filter (e.g. minimum years) be applied - before
                # ranking or after? What goes wrong if you filter a truncated top-k?
                # Hint: Session 6 - a filter excludes, a score ranks.
                results = self.hybrid_indexer.search_resumes(query, top_k)
                
                if results and self.llm:
                    # LLM reranking
                    reranked_results = self.llm_rerank(results, query, jd_context, top_k)
                    results = reranked_results
                
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
            """Route to appropriate search strategy based on query complexity and user preference"""
            query = inputs.get("query", "")
            search_mode = inputs.get("search_mode", "auto")
            
            if search_mode == "shallow":
                return "shallow"
            elif search_mode == "deep":
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
                        return decision
                    else:
                        pass
                        
                except Exception as e:
                    pass
            
            query_words = len(query.split())
            if query_words <= 3:
                return "shallow"
            else:
                return "deep"
        
        # TODO: HW13 [Medium] - The route is chosen twice per search
        # RunnableBranch evaluates each condition in turn, and each condition calls
        # route_search() - which calls the LLM. How many LLM calls does one search
        # spend just to choose a route? At 5 requests/minute, that matters.
        # Decide the route once (e.g. a RunnableLambda that adds it to the inputs),
        # then branch on the stored value. Should routing need an LLM at all?
        # Hint: Session 5 - Branched RAG; Session 6 - Runnables.
        chain = RunnableBranch(
            (lambda x: route_search(x) == "shallow", shallow_search),
            (lambda x: route_search(x) == "deep", deep_search),
            deep_search  # Default fallback
        )
        
        return chain
    
    def llm_rerank(self, results: List[Dict], query: str, jd_context: str, top_k: int) -> List[Dict]:
        """Use LLM to intelligently rerank search results"""
        if not results or not self.llm:
            return results
        
        try:
            pass
            
            candidates_info = []
            for i, result in enumerate(results):
                candidate_info = f"{i+1}. {result.get('name', 'Unknown')} - "
                candidate_info += f"Score: {result.get('combined_score', 0):.4f}"
                
                metadata = result.get('metadata', {})
                if 'skills' in metadata:
                    candidate_info += f", Skills: {', '.join(metadata['skills'][:3])}"
                if 'experience' in metadata:
                    candidate_info += f", Experience: {metadata['experience']} years"
                
                candidates_info.append(candidate_info)
            
            # TODO: HW14 [Medium] - The LLM reranks without seeing the resumes
            # The prompt gives the LLM each candidate's name and a retrieval score - and
            # the skills and experience fields, which are always empty (HW1, HW9).
            # How could a recruiter, human or AI, rank candidates from that?
            # Include a short excerpt of each resume.
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
                    
                    return reranked_results
                else:
                    pass
                    
            except Exception as e:
                pass
            
        except Exception as e:
            pass
        
        return results
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None, 
               search_mode: str = "auto", jd_context: str = "") -> Dict[str, Any]:
        """Execute search with routing between shallow and deep strategies"""
        
        inputs = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "search_mode": search_mode,
            "jd_context": jd_context
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
        """Get search router statistics"""
        return {
            "llm_available": self.llm is not None,
            "vector_store_ready": self.vector_store.is_ready(),
            "hybrid_indexer_ready": self.hybrid_indexer.get_index_stats()['hybrid_ready'],
            "search_modes": ["auto", "shallow", "deep"]
        }
