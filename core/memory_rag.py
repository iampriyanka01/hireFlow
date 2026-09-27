"""
Conversation memory system for tracking search history and context.
Uses LangChain's memory to maintain session context across searches.
"""

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)

class MemoryRAG:
    """Conversation memory system for maintaining search context and history"""
    
    def __init__(self):
        """Initialize LangChain conversation buffer for search tracking"""
        # TODO: HW23 [Easy] - Deprecated memory class
        # Run main.py and read the warning printed here. ConversationBufferMemory is
        # deprecated. This class only stores messages - replace it with a plain list
        # of HumanMessage / AIMessage objects.
        # Hint: Session 6 - Part 11, memory.
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def record_search(self, query: str, results_count: int):
        """Store search query and result count in conversation memory"""
        self.memory.chat_memory.add_user_message(f"Search: {query}")
        self.memory.chat_memory.add_ai_message(f"Found {results_count} results")
    
    def record_candidate_view(self, candidate_name: str):
        """Record candidate viewing"""
        self.memory.chat_memory.add_user_message(f"Viewed candidate: {candidate_name}")
        self.memory.chat_memory.add_ai_message("Candidate interaction recorded")
    
    def get_search_history(self) -> list:
        """Get recent search queries"""
        queries = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage) and msg.content.startswith("Search:"):
                queries.append(msg.content.replace("Search: ", ""))
        return queries[-5:]
    
    def get_memory_stats(self) -> dict:
        """Get simple memory stats"""
        return {
            'total_messages': len(self.memory.chat_memory.messages),
            'search_count': len([m for m in self.memory.chat_memory.messages 
                               if isinstance(m, HumanMessage) and m.content.startswith("Search:")]),
            'candidate_views': len([m for m in self.memory.chat_memory.messages 
                                  if isinstance(m, HumanMessage) and "candidate" in m.content.lower()])
        }
