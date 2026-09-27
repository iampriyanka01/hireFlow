"""Multi-Query Utility for HireFlow."""

from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.utils import get_logger, get_rate_limiter

logger = get_logger(__name__)

class MultiQueryGenerator:
    """Simple multi-query generator using LLM"""
    
    def __init__(self):
        if not GOOGLE_API_KEY:
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    rate_limiter=get_rate_limiter(),  # max 5 requests/minute on the free tier
                )
            except Exception as e:
                self.llm = None
    
    def generate_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """Generate multiple search queries from a single query"""
        if not original_query.strip():
            return [original_query]
        
        if self.llm:
            try:
                return self.generate_with_llm(original_query, num_queries)
            except Exception as e:
                pass
        
        return self.generate_fallback(original_query, num_queries)
    
    def generate_with_llm(self, original_query: str, num_queries: int) -> List[str]:
        """Generate queries using LLM"""
        prompt = f"""
        You are a search query expert. Given a user's search query, generate {num_queries} different but related search queries that would help find relevant results.

        Original Query: "{original_query}"

        Generate {num_queries} different search queries that:
        1. Are semantically related to the original query
        2. Use different but relevant keywords
        3. Cover different aspects of the search intent
        4. Are specific and searchable
        5. Would help find candidates with different backgrounds

        Return only the queries, one per line, without numbering or explanations.

        Example for "senior accountant":
        senior accountant with 5 years experience
        CPA certified accountant financial reporting
        accounting professional QuickBooks Excel
        """

        try:
            # TODO: HW19 [Easy] - The original query is dropped
            # The generated list never includes the original query, and lines the LLM
            # numbers anyway ('1. senior accountant...') keep their numbering.
            # Should the user's own wording be one of the searches?
            response = self.llm.invoke(prompt)
            queries = response.content.strip().split('\n')
            
            cleaned_queries = []
            for query in queries:
                query = query.strip()
                if query and len(query) > 3:
                    cleaned_queries.append(query)
            
            if len(cleaned_queries) >= num_queries:
                return cleaned_queries[:num_queries]
            elif cleaned_queries:
                fallback = self.generate_fallback(original_query, num_queries - len(cleaned_queries))
                return cleaned_queries + fallback
            else:
                return self.generate_fallback(original_query, num_queries)
                
        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            return self.generate_fallback(original_query, num_queries)
    
    def generate_fallback(self, original_query: str, num_queries: int) -> List[str]:
        """Fallback query generation using simple rules"""
        queries = [original_query]
        
        words = original_query.lower().split()
        
        if len(words) >= 2:
            if len(words) >= 3:
                queries.append(' '.join(words[:3]))
            else:
                queries.append(' '.join(words[:2]))
            
            if len(words) >= 4:
                queries.append(' '.join([words[0], words[-1]]))
            
            if len(words) >= 2:
                queries.append(words[0])
        
        while len(queries) < num_queries:
            queries.append(original_query)
        
        return queries[:num_queries]
    
    def set_temperature(self, temperature: float):
        """Set the creativity level for query generation (0.0 to 1.0)"""
        if self.llm and 0.0 <= temperature <= 1.0:
            self.llm.temperature = temperature
            pass
        else:
            pass

