"""
core/memory_rag.py  -  SESSION MEMORY (search history)
======================================================

Role in the pipeline
    A very small memory layer that remembers what the recruiter did during
    the current session, so the UI / CLI can show "recent searches" and
    simple usage statistics.

How it stores data
    Every event is saved as a pair of chat messages in a LangChain buffer:
        HumanMessage("Search: senior accountant")
        AIMessage("Found 3 results")
    Searches are recognised later by the "Search:" prefix.

Important
    Memory lives in RAM only. It resets when the program stops and is NOT
    fed back into search or ranking - it is purely a log for display.

Public API
    record_search(query, results_count)
    record_candidate_view(candidate_name)
    get_search_history()  -> last 5 queries
    get_memory_stats()    -> total messages / searches / candidate views
"""

from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logger = logging.getLogger(__name__)

class MemoryRAG:
    """Tracks searches and candidate views for the current session."""
    
    def __init__(self):
        """Create an empty in-memory message log.

        Fix for HW23: ConversationBufferMemory is deprecated (main.py printed a
        deprecation warning about it on every startup). This class only ever needed a
        plain running log of messages - not an LLM-facing memory object with prompt
        templates and variable injection - so a plain list replaces it.
        """
        self.messages: List[BaseMessage] = []
    
    def record_search(self, query: str, results_count: int):
        """Log one search as a Human/AI message pair ("Search: <query>" / "Found N results")."""
        self.messages.append(HumanMessage(content=f"Search: {query}"))
        self.messages.append(AIMessage(content=f"Found {results_count} results"))
    
    def record_candidate_view(self, candidate_name: str):
        """Log that the user opened a candidate's profile."""
        self.messages.append(HumanMessage(content=f"Viewed candidate: {candidate_name}"))
        self.messages.append(AIMessage(content="Candidate interaction recorded"))
    
    def get_search_history(self) -> list:
        """Return the last 5 search queries (oldest first), with the "Search: " prefix removed."""
        queries = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage) and msg.content.startswith("Search:"):
                queries.append(msg.content.replace("Search: ", ""))
        return queries[-5:]
    
    def get_memory_stats(self) -> dict:
        """Return counters: total messages, number of searches, number of candidate views.

        Note: candidate views are counted by looking for the word "candidate"
        in user messages, which is a simple heuristic.
        """
        return {
            'total_messages': len(self.messages),
            'search_count': len([m for m in self.messages
                               if isinstance(m, HumanMessage) and m.content.startswith("Search:")]),
            'candidate_views': len([m for m in self.messages
                                  if isinstance(m, HumanMessage) and "candidate" in m.content.lower()])
        }
