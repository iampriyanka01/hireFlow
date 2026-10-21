"""
main.py  -  HireFlow command-line demo and orchestrator
=======================================================

This is the entry point that WIRES ALL MODULES TOGETHER and demonstrates them:
- Complete document processing pipeline (PDF → Text → Embeddings → Index)
- Multiple search modes (Vector, Hybrid BM25+Vector, AI-Enhanced)
- AI-powered evaluation and re-ranking with Gemini LLM
- Memory system for conversation tracking
- Quality evaluation with RAGAS metrics
- System health monitoring and diagnostics

Usage: python main.py

Program flow (see main() at the bottom)
    1. check_prerequisites()      API keys set? data folders contain PDFs?
    2. initialize_system()        create every component (order matters, below)
    3. load_and_index_documents() PDFs -> Documents -> BM25 + Pinecone indexes
    4. automatic demos            search modes, AI evaluation, memory, diagnostics
    5. interactive_menu()         run any demo again, RAGAS evaluation, re-index

Component wiring (who shares what)
    VectorStore ──shared──> HybridIndexer ──shared──> SearchRouter ──> RAGEvaluator
    ReRanker, ResumeParser, JobParser and MemoryRAG are independent.
    Sharing ONE VectorStore / HybridIndexer means one Pinecone connection and
    one filled index; creating a second copy would give an empty index.

Return codes: main() returns 0 on success / user exit and 1 on setup failure.
"""

import os
import sys
from typing import Dict, List
from core.ingestion import load_resumes, load_job_descriptions
from core.vector_store import VectorStore
from core.hybrid_indexer import HybridIndexer
from core.search_router import SearchRouter
from core.parsing import ResumeParser, JobParser
from core.re_ranker import ReRanker
from core.evaluator import RAGEvaluator
from core.memory_rag import MemoryRAG
from utils.config import GOOGLE_API_KEY, PINECONE_API_KEY
from utils.utils import get_logger, load_pdf
from utils.schemas import JobDescription, Resume

logger = get_logger(__name__)

class HireFlowDemo:
    """Holds every HireFlow component and the demo actions that use them.

    Attributes:
        components:   dict name -> instance (filled by initialize_system()).
        system_ready: True once initialisation succeeded.
        resume_dir / jd_dir: folders that are scanned for PDFs.
    """
    
    def __init__(self):
        """Create the (empty) component registry and set the data folder paths."""
        self.components = {
            'vector_store': None,
            'hybrid_indexer': None,
            'search_router': None,
            'reranker': None,
            'evaluator': None,
            'memory': None,
            'resume_parser': None,
            'job_parser': None
        }
        self.system_ready = False
        self.resume_dir = "data/resumes"
        self.jd_dir = "data/jds"
        
    def show_banner(self):
        """Print the title banner."""
        print("\n" + "="*70)
        print("HireFlow - AI-Powered Candidate Search Engine")
        print("Complete System Demonstration & Testing Platform")
        print("="*70)
        
    def check_prerequisites(self) -> bool:
        """Verify the environment before doing any expensive work.

        Checks: GOOGLE_API_KEY and PINECONE_API_KEY are set, both data folders
        exist and each contains at least one .pdf file.

        Returns:
            True if everything is present, False (with a printed reason) otherwise.
        """
        print("\nChecking Prerequisites...")
        
        # Check API keys
        if not GOOGLE_API_KEY:
            print("ERROR: GOOGLE_API_KEY not set in environment")
            return False
        print("OK: Google API Key configured")
        
        if not PINECONE_API_KEY:
            print("ERROR: PINECONE_API_KEY not set in environment")
            return False
        print("OK: Pinecone API Key configured")
        
        # Check data directories
        if not os.path.exists(self.resume_dir):
            print(f"ERROR: Resume directory not found: {self.resume_dir}")
            return False
        if not os.path.exists(self.jd_dir):
            print(f"ERROR: Job description directory not found: {self.jd_dir}")
            return False
        
        resume_count = len([f for f in os.listdir(self.resume_dir) if f.endswith('.pdf')])
        jd_count = len([f for f in os.listdir(self.jd_dir) if f.endswith('.pdf')])
        
        print(f"OK: Found {resume_count} resumes and {jd_count} job descriptions")
        
        if resume_count == 0 or jd_count == 0:
            print("ERROR: No PDF files found in data directories")
            return False
            
        return True
    
    def initialize_system(self) -> bool:
        """Create and connect every component in dependency order.

        Order: VectorStore -> HybridIndexer(vector_store) -> SearchRouter(indexer)
               -> ReRanker / ResumeParser / JobParser -> RAGEvaluator(router)
               -> MemoryRAG.

        Returns:
            True if all components were created, False on any error.
        """
        print("\nInitializing HireFlow System...")
        
        try:
            # Initialize vector store
            print("   Initializing Vector Store (Pinecone)...")
            self.components['vector_store'] = VectorStore()
            if not self.components['vector_store'].initialize():
                print("ERROR: Vector store initialization failed")
                return False
            print("   OK: Vector Store ready")
            
            # Initialize hybrid indexer
            print("   Initializing Hybrid Indexer (BM25 + Vector)...")
            # Share the vector store so every component uses one Pinecone connection
            self.components['hybrid_indexer'] = HybridIndexer(vector_store=self.components['vector_store'])
            print("   OK: Hybrid Indexer ready")
            
            # Initialize search router
            print("   Initializing Search Router...")
            # Give the router the SAME indexer that load_and_index_documents() fills.
            # SearchRouter() on its own builds a private, empty indexer.
            self.components['search_router'] = SearchRouter(hybrid_indexer=self.components['hybrid_indexer'])
            print("   OK: Search Router ready")
            
            # Initialize AI components
            print("   Initializing AI Components (Gemini LLM)...")
            self.components['reranker'] = ReRanker()
            self.components['resume_parser'] = ResumeParser()
            self.components['job_parser'] = JobParser()
            print("   OK: AI Components ready")
            
            # Initialize evaluation and memory
            print("   Initializing Evaluation & Memory...")
            self.components['evaluator'] = RAGEvaluator(search_router=self.components['search_router'])
            self.components['memory'] = MemoryRAG()
            print("   OK: Evaluation & Memory ready")
            
            self.system_ready = True
            print("\nHireFlow System Successfully Initialized!")
            return True
            
        except Exception as e:
            print(f"ERROR: System initialization failed: {e}")
            return False
    
    def load_and_index_documents(self) -> bool:
        """Read the PDFs and fill the search indexes.

        1. load_resumes() / load_job_descriptions()  (core/ingestion.py)
        2. HybridIndexer.index_*()  -> builds BM25 AND uploads to Pinecone (once)
        Also used by menu option 6 to re-index.
        """
        print("\nLoading and Indexing Documents...")
        
        try:
            # Load documents
            print("   Loading resumes...")
            resumes = load_resumes(self.resume_dir)
            print(f"   OK: Loaded {len(resumes)} resumes")
            
            print("   Loading job descriptions...")
            jds = load_job_descriptions(self.jd_dir)
            print(f"   OK: Loaded {len(jds)} job descriptions")
            
            # Index in hybrid indexer
            print("   Indexing resumes in hybrid search...")
            if not self.components['hybrid_indexer'].index_resumes(resumes):
                print("ERROR: Resume indexing failed")
                return False
            print("   OK: Resumes indexed in hybrid search")
            
            print("   Indexing job descriptions...")
            if not self.components['hybrid_indexer'].index_job_descriptions(jds):
                print("ERROR: Job description indexing failed")
                return False
            print("   OK: Job descriptions indexed")
            
            # Fix for HW24: index_resumes()/index_job_descriptions() above already embed
            # and upsert to Pinecone when the vector store is ready (see
            # HybridIndexer.index_resumes() in core/hybrid_indexer.py). Calling
            # vector_store.add_resumes()/add_job_descriptions() again here used to embed
            # and upload every document a SECOND time - doubling the Gemini embedding
            # calls and the Pinecone upserts for no benefit, since upsert-by-id (HW6)
            # just overwrote the same vectors with an identical copy.
            
            return True
            
        except Exception as e:
            print(f"ERROR: Document loading failed: {e}")
            return False
    
    def test_search_modes(self):
        """Demo: run the same query through the three search strategies and compare.

        1. Vector search  - semantic only (VectorStore)
        2. Hybrid search  - BM25 + vector (HybridIndexer)
        3. Router "deep"  - hybrid + LLM re-ranking (SearchRouter)
        The search is also recorded in MemoryRAG.
        """
        print("\nSearch Mode Demonstrations")
        print("-" * 40)
        
        # Get a sample job description for testing
        jd_files = [f for f in os.listdir(self.jd_dir) if f.endswith('.pdf')]
        if not jd_files:
            print("ERROR: No job descriptions available for demo")
            return
            
        sample_jd_path = os.path.join(self.jd_dir, jd_files[0])
        sample_jd_text = load_pdf(sample_jd_path)
        
        if not sample_jd_text:
            print("ERROR: Could not load sample job description")
            return
            
        # Extract a search query from the JD
        search_query = "senior accountant with Excel and QuickBooks experience"
        print(f"Demo Query: '{search_query}'")
        
        # 1. Vector Search Only
        print("\n1. Vector Search (Semantic Only)")
        vector_results = self.components['vector_store'].search_resumes(search_query, top_k=3)
        print(f"   Found {len(vector_results)} candidates")
        self.display_results(vector_results, max_show=2)
        
        # 2. Hybrid Search (BM25 + Vector)
        print("\n2. Hybrid Search (BM25 + Vector)")
        hybrid_results = self.components['hybrid_indexer'].search_resumes(search_query, top_k=3)
        print(f"   Found {len(hybrid_results)} candidates")
        self.display_results(hybrid_results, max_show=2)
        
        # 3. AI-Enhanced Search with Router
        print("\n3. AI-Enhanced Search (Router + Re-ranking)")
        router_results = self.components['search_router'].search(search_query, top_k=3, search_mode="deep")
        candidates = router_results.get('results', [])
        print(f"   Found {len(candidates)} candidates")
        self.display_results(candidates, max_show=2)
        
        # Record in memory
        self.components['memory'].record_search(search_query, len(candidates))
    
    def test_ai_evaluation(self):
        """Demo: parse a job description and let the LLM evaluate the top candidate.

        Flow: load first JD PDF -> JobParser -> JobDescription model ->
        search "senior accountant" -> take result #1 -> ReRanker.evaluate_candidate()
        -> print fit score, strengths, gaps and summary.
        """
        print("\nAI Evaluation Demonstration")
        print("-" * 40)
        
        # Get a sample job and candidate
        jd_files = [f for f in os.listdir(self.jd_dir) if f.endswith('.pdf')]
        if not jd_files:
            print("ERROR: No job descriptions available for evaluation")
            return
            
        sample_jd_path = os.path.join(self.jd_dir, jd_files[0])
        sample_jd_text = load_pdf(sample_jd_path)
        
        # Search for candidates
        search_query = "senior accountant"
        results = self.components['search_router'].search(search_query, top_k=2, search_mode="deep")
        candidates = results.get('results', [])
        
        if not candidates:
            print("ERROR: No candidates found for evaluation")
            return
            
        candidate = candidates[0]
        print(f"Evaluating candidate: {candidate.get('name', 'Unknown')}")
        
        # Parse job description for structured evaluation
        try:
            parsed_jd = self.components['job_parser'].parse_job_description(sample_jd_text, "demo_jd")
            jd_obj = JobDescription(**parsed_jd)
            
            # AI evaluation - the ReRanker method is evaluate_candidate(Resume, JobDescription)
            resume_obj = Resume(
                candidate_id=candidate.get('candidate_id', 'unknown'),
                name=candidate.get('name', 'Unknown'),
                text=candidate.get('text', ''),
            )
            evaluation = self.components['reranker'].evaluate_candidate(resume_obj, jd_obj)
            
            print("\nAI Evaluation Results:")
            print(f"   Fit Score: {evaluation.fit_score}/100")
            print(f"   Strengths: {', '.join(evaluation.strengths[:3])}")
            print(f"   Gaps: {', '.join(evaluation.gaps[:2])}")
            print(f"   Summary: {evaluation.summary[:100]}...")
            
        except Exception as e:
            print(f"ERROR: AI evaluation failed: {e}")
    
    def test_quality_metrics(self):
        """Demo: score one query with RAGAS (slow on the free API tier).

        Prints the four metrics, the weighted overall score and a verdict
        (>= 0.8 excellent, >= 0.6 good, otherwise needs improvement).
        """
        print("\nSearch Quality Evaluation (RAGAS)")
        print("-" * 40)
        
        try:
            # Evaluate search quality
            test_query = "senior accountant with CPA"
            expected_skills = ["accounting", "CPA", "financial reporting", "Excel"]
            
            print(f"Evaluating query: '{test_query}'")
            print(f"Expected skills: {', '.join(expected_skills)}")
            
            metrics = self.components['evaluator'].evaluate_search_quality(
                test_query, expected_skills, search_mode="deep", top_k=5
            )
            
            print("\nRAGAS Quality Metrics:")
            print(f"   Answer Relevancy: {metrics.answer_relevancy:.2f}")
            print(f"   Context Precision: {metrics.context_precision:.2f}")
            print(f"   Faithfulness: {metrics.faithfulness:.2f}")
            print(f"   Answer Correctness: {metrics.answer_correctness:.2f}")
            print(f"   Overall Score: {metrics.overall_score:.2f}")
            
            if metrics.overall_score >= 0.8:
                print("   RESULT: Excellent search quality!")
            elif metrics.overall_score >= 0.6:
                print("   RESULT: Good search quality")
            else:
                print("   RESULT: Search quality needs improvement")
                
        except Exception as e:
            print(f"ERROR: Quality evaluation failed: {e}")
    
    def test_memory_system(self):
        """Demo: print memory statistics and the last few searches."""
        print("\nMemory System Demonstration")
        print("-" * 40)
        
        # Show memory stats
        stats = self.components['memory'].get_memory_stats()
        print(f"Memory Statistics:")
        print(f"   Total Messages: {stats['total_messages']}")
        print(f"   Search Queries: {stats['search_count']}")
        print(f"   Candidate Views: {stats['candidate_views']}")
        
        # Show recent searches
        recent_searches = self.components['memory'].get_search_history()
        if recent_searches:
            print(f"\nRecent Searches:")
            for i, query in enumerate(recent_searches[-3:], 1):
                print(f"   {i}. {query}")
        else:
            print("\nNo search history yet")
    
    def check_system_status(self):
        """Print diagnostics: Pinecone vector count / dimension and the hybrid indexer's readiness flags."""
        print("\nSystem Diagnostics")
        print("-" * 40)
        
        # Vector store stats
        if self.components['vector_store']:
            vs_stats = self.components['vector_store'].get_stats()
            print(f"Vector Store Status: {vs_stats.get('status', 'unknown')}")
            if vs_stats.get('status') == 'ready':
                print(f"   Total Vectors: {vs_stats.get('total_vector_count', 0)}")
                print(f"   Dimensions: {vs_stats.get('dimension', 0)}")
        
        # Hybrid indexer stats
        if self.components['hybrid_indexer']:
            hybrid_stats = self.components['hybrid_indexer'].get_index_stats()
            print(f"Hybrid Indexer Status:")
            print(f"Resumes Indexed: {len(self.components['hybrid_indexer'].resume_texts)}")
            print(f"JDs Indexed: {len(self.components['hybrid_indexer'].jd_texts)}")
            print(f"Resumes Ready: {hybrid_stats.get('resumes_ready', False)}")
            print(f"JDs Ready: {hybrid_stats.get('jds_ready', False)}")
            print(f"Vector Ready: {hybrid_stats.get('vector_store_ready', False)}")
            print(f"Hybrid Ready: {hybrid_stats.get('hybrid_ready', False)}")
    
    def display_results(self, results: List[Dict], max_show: int = 3):
        """Print up to `max_show` results as "name / score / experience / top-3 skills".

        Accepts both result shapes in the project: vector results (values under
        'metadata') and hybrid / router results (values at the top level).
        """
        for i, result in enumerate(results[:max_show], 1):
            metadata = result.get('metadata') or result
            name = metadata.get('name') or result.get('name') or f'Candidate {i}'
            score = result.get('score', result.get('combined_score', 0)) or 0
            skills = metadata.get('skills') or result.get('skills') or []
            experience = metadata.get('experience', result.get('experience', 'N/A'))
            
            print(f"   {i}. {name}")
            print(f"      Score: {score:.3f} | Experience: {experience} | Skills: {', '.join(skills[:3])}")
    
    def interactive_menu(self):
        """Loop showing a 7-option menu until the user exits (option 7) or presses Ctrl+C."""
        print("\nInteractive HireFlow Explorer")
        print("=" * 50)
        
        while True:
            print("\nAvailable Demonstrations:")
            print("1. Search Mode Comparison")
            print("2. AI Evaluation Demo")
            print("3. Quality Evaluation (RAGAS)")
            print("4. Memory System")
            print("5. System Diagnostics")
            print("6. Reload & Re-index Documents")
            print("7. Exit")
            
            try:
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == "1":
                    self.test_search_modes()
                elif choice == "2":
                    self.test_ai_evaluation()
                elif choice == "3":
                    self.test_quality_metrics()
                elif choice == "4":
                    self.test_memory_system()
                elif choice == "5":
                    self.check_system_status()
                elif choice == "6":
                    if self.load_and_index_documents():
                        print("OK: Documents reloaded successfully")
                    else:
                        print("ERROR: Document reload failed")
                elif choice == "7":
                    print("\nThank you for exploring HireFlow!")
                    break
                else:
                    print("ERROR: Invalid choice. Please select 1-7.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"ERROR: {e}")

def main():
    """Run prerequisites -> initialise -> index -> demos -> menu.

    Returns:
        Process exit code: 0 = OK / user quit, 1 = a setup step failed.
    """
    demo = HireFlowDemo()
    
    try:
        # Print header
        demo.show_banner()
        
        # Check prerequisites
        if not demo.check_prerequisites():
            print("\nERROR: Prerequisites not met. Please check configuration and data files.")
            return 1
        
        # Initialize system
        if not demo.initialize_system():
            print("\nERROR: System initialization failed.")
            return 1
        
        # Load and index documents
        if not demo.load_and_index_documents():
            print("\nERROR: Document loading failed.")
            return 1
        
        print("\nHireFlow is ready for comprehensive testing!")
        
        # Run automatic demonstrations
        print("\nRunning Automatic Demonstrations...")
        demo.test_search_modes()
        demo.test_ai_evaluation()
        demo.test_memory_system()
        demo.check_system_status()
        
        # Interactive menu
        demo.interactive_menu()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"\nERROR: Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
