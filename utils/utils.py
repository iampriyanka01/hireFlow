"""
Common utility functions for text processing, PDF handling, error detection, embeddings, and filtering.
Centralized utilities used across the HireFlow project.
"""

import logging
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_logger(name: str) -> logging.Logger:
    """Create standardized logger with timestamp formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    return logging.getLogger(name)

def clean_text(text: str) -> str:
    """Normalize text by removing extra whitespace and invalid characters"""
    if not text:
        return ""
    
    # Normalize whitespace (multiple spaces/tabs/newlines to single space)
    text = re.sub(r'\s+', ' ', text)
    # Keep only alphanumeric, common punctuation, and symbols
    text = re.sub(r'[^\w\s\.\,\-\+\@\#\&\*\(\)]', '', text)
    return text.strip()

def truncate_text(text: str, max_length: int = 8000) -> str:
    """Cut text at max length and add ellipsis if truncated"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

def load_pdf(file_path: str) -> str:
    """Extract all text content from PDF file using LangChain"""
    logger = get_logger(__name__)
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            return ""
        
        # Combine all pages into single text
        full_text = "\n".join([doc.page_content for doc in documents])
        return full_text.strip()
        
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return ""

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split large text into overlapping chunks for processing"""
    logger = get_logger(__name__)
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # Split on paragraphs first, then lines, etc.
        )
        return splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        return [text]  # Return original text if splitting fails

_rate_limiter = None

def get_rate_limiter():
    """One shared limiter for every LLM in the app.

    The free tier allows only 5 requests per MINUTE. This must be a single
    shared object: if each component made its own limiter, each would allow
    5/min and together they would still exceed the limit. Calls over the
    limit wait in a queue instead of failing with a 429 error.
    """
    global _rate_limiter
    if _rate_limiter is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from utils.config import LLM_REQUESTS_PER_MINUTE
        _rate_limiter = InMemoryRateLimiter(
            requests_per_second=LLM_REQUESTS_PER_MINUTE / 60,
            check_every_n_seconds=0.5,
            max_bucket_size=1,
        )
    return _rate_limiter

def is_quota_error(error: Exception) -> bool:
    """Detect API quota/rate limit errors for graceful handling"""
    error_str = str(error).lower()
    return "429" in str(error) or "quota" in error_str or "rate limit" in error_str

def get_embeddings():
    """Create Google embedding model with quota error handling"""
    from utils.config import GOOGLE_API_KEY, PINECONE_DIMENSION
    import asyncio

    try:
        # Fix for Streamlit: Ensure event loop exists in thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",  # Google's text embedding model
            google_api_key=GOOGLE_API_KEY,
            # gemini-embedding-001 returns 3072 numbers unless told otherwise.
            # The index is created at PINECONE_DIMENSION, so they must agree.
            output_dimensionality=PINECONE_DIMENSION,
        )
    except Exception as e:
        if is_quota_error(e):
            print("API quota exceeded. Vector search disabled.")
        else:
            print(f"Embeddings failed: {e}")
        return None  # Allow system to continue without vector search

# Filtering utility functions
def filter_by_skills(candidates: List[Dict[str, Any]], required_skills: List[str]) -> List[Dict[str, Any]]:
    """Filter candidates who have all required skills"""
    if not required_skills:
        return candidates
    
    filtered_candidates = []
    
    for candidate in candidates:
        metadata = candidate.get('metadata', {})
        candidate_skills = metadata.get('skills', [])
        
        has_required_skills = all(
            skill.lower() in [cs.lower() for cs in candidate_skills]
            for skill in required_skills
        )
        
        if has_required_skills:
            filtered_candidates.append(candidate)
    
    return filtered_candidates

def filter_by_location(candidates: List[Dict[str, Any]], target_locations: List[str]) -> List[Dict[str, Any]]:
    """Filter candidates by target locations (case-insensitive partial match)"""
    if not target_locations:
        return candidates
    
    filtered_candidates = []
    
    for candidate in candidates:
        metadata = candidate.get('metadata', {})
        candidate_location = metadata.get('location', '').lower()
        
        location_match = any(
            target_loc.lower() in candidate_location 
            for target_loc in target_locations
        )
        
        if location_match:
            filtered_candidates.append(candidate)
    
    return filtered_candidates

def filter_by_experience(candidates: List[Dict[str, Any]], min_experience: float) -> List[Dict[str, Any]]:
    """Filter candidates who meet minimum experience requirement"""
    if min_experience is None:
        return candidates
    
    filtered_candidates = []
    
    for candidate in candidates:
        metadata = candidate.get('metadata', {})
        candidate_experience = metadata.get('experience')
        
        if candidate_experience is None:
            continue
        
        if candidate_experience >= min_experience:
            filtered_candidates.append(candidate)
    
    return filtered_candidates

def apply_filters(
    candidates: List[Dict[str, Any]], 
    required_skills: List[str] = None,
    target_locations: List[str] = None,
    min_experience: float = None
) -> List[Dict[str, Any]]:
    """Apply all filters (skills, location, experience) to candidate list"""
    filtered_candidates = candidates
    
    if required_skills:
        filtered_candidates = filter_by_skills(filtered_candidates, required_skills)
    
    if target_locations:
        filtered_candidates = filter_by_location(filtered_candidates, target_locations)
    
    if min_experience is not None:
        filtered_candidates = filter_by_experience(filtered_candidates, min_experience)
    
    return filtered_candidates

