"""
core/evaluator.py  -  MEASURING SEARCH QUALITY with RAGAS
=========================================================

Role in the pipeline
    Answers the question "is our search actually good?" with numbers.
    RAGAS is a library that uses an LLM as a judge to score RAG systems.

    query ─> SearchRouter.search() ─> candidates
          ─> prepare_ragas_data()   ─> one row per candidate:
                 question      = the query
                 contexts      = text describing the candidate
                 answer        = a generated sentence about the candidate
                 ground_truth  = "Excellent/Good/Moderate/Poor match: x/y skills"
          ─> ragas.evaluate()       ─> 4 metrics
          ─> weighted overall_score ─> stored in evaluation_history

The four metrics (all 0.0 - 1.0, higher is better)
    answer_relevancy   - does the answer address the question?           (30%)
    context_precision  - are the retrieved contexts relevant / well ranked? (30%)
    faithfulness       - is the answer supported by the context?          (20%)
    answer_correctness - does the answer match the ground truth?          (20%)

Cost warning
    Each metric makes several LLM calls per row, and the free Gemini tier is
    limited to ~5 requests/minute, so an evaluation with top_k=5 can take many
    minutes. run_config() limits RAGAS to 1 worker with long timeouts for
    that reason.

Public API
    RAGEvaluator.evaluate_search_quality(query, expected_skills, ...)
    RAGEvaluator.get_evaluation_summary() / export_evaluations(filename)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass
from core.search_router import SearchRouter
from core.re_ranker import ReRanker
from utils.schemas import JobDescription, Resume
from utils.utils import get_logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, context_precision, faithfulness, answer_correctness
)

logger = get_logger(__name__)

@dataclass
class RAGEvaluationMetrics:
    """Result of one evaluation: the four RAGAS metrics plus a weighted overall score."""
    answer_relevancy: float     # How relevant the answer is to the question
    context_precision: float    # Precision of retrieved context
    faithfulness: float         # Factual consistency with context
    answer_correctness: float   # Overall answer quality
    overall_score: float        # Weighted average of all metrics
    # Fix for HW21: True when RAGAS could not score at least one metric for at least one
    # candidate (NaN from a quota error, timeout, or unparseable judge reply), or when
    # there was nothing to evaluate at all. A caller should treat a low score with
    # judge_failed=True as "we don't know", not "the search is bad".
    judge_failed: bool = False
    
    def to_dict(self) -> Dict[str, float]:
        """Return the metrics as a plain dict (used for history / CSV export)."""
        return {
            'answer_relevancy': self.answer_relevancy,
            'context_precision': self.context_precision,
            'faithfulness': self.faithfulness,
            'answer_correctness': self.answer_correctness,
            'overall_score': self.overall_score,
            'judge_failed': self.judge_failed
        }

class RAGEvaluator:
    """Runs searches, scores them with RAGAS and keeps a history of the scores."""
    
    def __init__(self, search_router: SearchRouter = None, reranker: ReRanker = None):
        """Initialize RAGAS metrics and evaluation tracking.

        Pass in the app's SearchRouter. Building a new one here gives an
        empty index, so every search returns nothing and every score is 0.

        Args:
            reranker: the ReRanker whose real AI assessment gets judged by RAGAS (fix
                for HW20, see prepare_ragas_data). Built lazily if not supplied.
        """
        self.search_router = search_router
        self.reranker = reranker
        self.ragas_metrics = [
            answer_relevancy, context_precision, faithfulness, answer_correctness
        ]
        self.evaluation_history = []  # Track all evaluations

    @staticmethod
    def estimate_cost(top_k: int, requests_per_minute: int = 5) -> Dict[str, Any]:
        """Fix for HW22: count the cost before you run. Each of the 4 RAGAS metrics makes
        roughly 2 LLM calls per candidate, PLUS one ReRanker call per candidate (fix for
        HW20 - RAGAS now judges a real AI assessment, which itself costs a call).
        Show this to the user BEFORE running, instead of finding out by waiting.

        Returns: {'estimated_llm_calls', 'estimated_minutes'}
        """
        ragas_calls = len([answer_relevancy, context_precision, faithfulness, answer_correctness]) * 2
        reranker_calls = 1
        total_calls = (ragas_calls + reranker_calls) * max(top_k, 1)
        minutes = total_calls / max(requests_per_minute, 1)
        return {"estimated_llm_calls": total_calls, "estimated_minutes": round(minutes, 1)}

    def evaluate_search_quality(self, query: str, expected_skills: List[str], 
                              search_mode: str = "deep", top_k: int = 5) -> RAGEvaluationMetrics:
        """Run one search and score its results.

        Args:
            query:           the search text to test.
            expected_skills: skills a good candidate should have; used to build
                             the "ground truth" labels.
            search_mode:     "shallow", "deep" or "auto".
            top_k:           how many results to evaluate (more = slower).

        Returns:
            RAGEvaluationMetrics. judge_failed=True (fix for HW21) if the search
            returned nothing, an error occurred, or RAGAS itself could not score
            every row - so a caller can tell "the search is bad" from "we don't know".
        """
        estimate = self.estimate_cost(top_k)
        logger.warning(
            f"RAGAS evaluation of top_k={top_k}: ~{estimate['estimated_llm_calls']} LLM calls, "
            f"~{estimate['estimated_minutes']} min at the free tier's 5 requests/minute."
        )
        
        try:
            search_router = self.search_router or SearchRouter()
            search_result = search_router.search(query, top_k, search_mode=search_mode)
            
            if not search_result.get('results'):
                return self.create_default_metrics()
            
            candidates = search_result['results']
            return self.evaluate_with_ragas(query, candidates, expected_skills)
                
        except Exception as e:
            logger.error(f"Search quality evaluation failed: {e}")
            return self.create_default_metrics()
    
    def create_default_metrics(self) -> RAGEvaluationMetrics:
        """Build a metrics object with every value 0.0 (used on failure / empty results).

        Fix for HW21: judge_failed=True flags that this 0.0 is NOT a real quality score -
        there was nothing to evaluate, or evaluation itself broke - so it should never be
        read as "the search returned bad candidates".
        """
        return RAGEvaluationMetrics(
            answer_relevancy=0.0,
            context_precision=0.0,
            faithfulness=0.0,
            answer_correctness=0.0,
            overall_score=0.0,
            judge_failed=True
        )
    
    def evaluate_with_ragas(self, query: str, candidates: List[Dict], 
                            expected_skills: List[str]) -> RAGEvaluationMetrics:
        """Score the candidates with RAGAS and record the outcome.

        Builds the dataset, calls ragas.evaluate() with Gemini as the judge
        (LLM + embeddings), averages each metric over all rows, computes the
        weighted overall score and appends the result to evaluation_history.
        """
        
        evaluation_data = self.prepare_ragas_data(query, candidates, expected_skills)
        results = evaluate(
            dataset=self.to_ragas_dataset(evaluation_data),
            metrics=self.ragas_metrics,
            llm=self.judge_llm(),                # without this, RAGAS tries OpenAI
            embeddings=self.judge_embeddings(),
            run_config=self.run_config(),
        )
        table = results.to_pandas()

        # Fix for HW21: each column_mean() call also reports whether the judge failed
        # to score that metric at all (NaN/missing), instead of silently becoming 0.0.
        relevancy, relevancy_failed = self.column_mean(table, 'answer_relevancy')
        precision, precision_failed = self.column_mean(table, 'context_precision')
        faith, faith_failed = self.column_mean(table, 'faithfulness')
        correctness, correctness_failed = self.column_mean(table, 'answer_correctness')

        metrics = RAGEvaluationMetrics(
            answer_relevancy=relevancy,
            context_precision=precision,
            faithfulness=faith,
            answer_correctness=correctness,
            overall_score=0.0,
            judge_failed=any([relevancy_failed, precision_failed, faith_failed, correctness_failed])
        )
        
        metrics.overall_score = self.calculate_overall_score(metrics)
        self.store_evaluation(query, metrics)
        
        return metrics
    
    def prepare_ragas_data(self, query: str, candidates: List[Dict], 
                           expected_skills: List[str]) -> pd.DataFrame:
        """Turn search results into a DataFrame with the four RAGAS columns.

        Columns: question, contexts (list of strings), ground_truth, answer.
        One row is created per candidate.

        Fix for HW20: the old version built 'context' from metadata that was usually
        empty (fixed now by HW1/HW9, but still just a few keywords), and 'answer' was a
        hand-written template sentence like "{name} with skills in {skills}" - never
        produced by an AI. RAGAS was therefore judging its own template against itself,
        so "faithfulness" and "answer_relevancy" measured nothing real.

        This version judges something real:
          question -> the actual search query
          context  -> the candidate's real resume text
          answer   -> the ReRanker's real AI-generated fit assessment (summary +
                      strengths) for that resume against this query, treated as a job
        This costs one extra ReRanker call per candidate (see estimate_cost / HW22).
        """
        reranker = self.reranker or ReRanker()
        # A minimal ad-hoc JobDescription built from the query itself - this evaluation
        # has no separate job posting, so the query IS the job requirement being judged.
        jd = JobDescription(jd_id="ragas_eval", title=query, text=query)

        data = []
        for candidate in candidates:
            metadata = candidate.get('metadata', {})
            resume_text = candidate.get('text', '') or candidate.get('page_content', '')
            skills = candidate.get('skills') or metadata.get('skills', [])
            experience = candidate.get('experience') or metadata.get('experience', 0)

            # Schema-required fields with no bearing on what gets judged (the prompt
            # only ever sends jd.text and resume.text to the LLM - see ReRanker).
            resume = Resume(
                candidate_id=candidate.get('candidate_id', 'unknown'),
                name=candidate.get('name', metadata.get('name', 'Unknown')),
                email="unknown@example.com",
                phone="Not provided",
                experience=experience or 0,
                skills=skills or [],
                education="Not specified",
                text=resume_text,
            )
            evaluation = reranker.evaluate_candidate(resume, jd)
            real_answer = f"{evaluation.summary} Strengths: {', '.join(evaluation.strengths)}".strip()

            ground_truth = self.create_ground_truth(skills, expected_skills)
            
            data.append({
                'question': query,
                'contexts': [resume_text or "No resume text available"],
                'ground_truth': ground_truth,
                'answer': real_answer or "No assessment generated"
            })
        
        return pd.DataFrame(data)
    
    # ---- RAGAS 0.4 plumbing ------------------------------------------------

    @staticmethod
    def to_ragas_dataset(frame: pd.DataFrame):
        """RAGAS 0.4 needs an EvaluationDataset with these field names, not a DataFrame."""
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        samples = []
        for _, row in frame.iterrows():
            samples.append(SingleTurnSample(
                user_input=row['question'],
                retrieved_contexts=list(row['contexts']),
                response=row['answer'],
                reference=row['ground_truth'],
            ))
        return EvaluationDataset(samples=samples)

    @staticmethod
    def judge_llm():
        """Gemini wrapped for RAGAS to use as the "judge" model (shares the app's rate limiter)."""
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI
        from utils.config import GOOGLE_API_KEY, LLM_MODEL
        from utils.utils import get_rate_limiter
        return LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model=LLM_MODEL, google_api_key=GOOGLE_API_KEY,
            rate_limiter=get_rate_limiter()))   # shares the app's 5/minute budget

    @staticmethod
    def judge_embeddings():
        """Google embeddings wrapped for RAGAS (needed by the answer_relevancy metric)."""
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from utils.utils import get_embeddings
        return LangchainEmbeddingsWrapper(get_embeddings())

    @staticmethod
    def run_config():
        """RAGAS defaults to 16 parallel workers and a 180s timeout. At 5
        requests per minute that means rate-limit errors, then timeouts,
        then every score comes back as NaN."""
        from ragas.run_config import RunConfig
        return RunConfig(max_workers=1, timeout=900, max_retries=10, max_wait=60)

    @staticmethod
    def column_mean(table: pd.DataFrame, column: str) -> tuple:
        """Average of one metric column, ignoring NaN - and whether the judge failed.

        Fix for HW21: RAGAS returns NaN when the judge cannot decide (quota, timeout,
        bad JSON), and the old code turned that into 0.0, indistinguishable from a
        genuinely bad score. This now returns (mean, failed): failed is True when the
        column is missing entirely, or when EVERY row for it came back NaN - so the
        caller can tell "the search is bad" (a real low score) apart from "the judge
        failed" (no usable score at all).

        Returns:
            (float mean [0.0 if nothing usable], bool failed)
        """
        if column not in table.columns:
            return 0.0, True
        values = table[column].dropna()
        if len(values) == 0:
            return 0.0, True
        return float(values.mean()), False

    def create_ground_truth(self, candidate_skills: List[str], expected_skills: List[str]) -> str:
        """Create the expected-answer text from how many expected skills the candidate has.

        >=80% -> "Excellent match", >=60% -> "Good", >=40% -> "Moderate",
        otherwise "Poor", each with a "x/y skills" count.
        """
        if not expected_skills:
            return "No skills specified"
         
        matched_skills = [skill for skill in expected_skills 
                         if skill.lower() in [s.lower() for s in candidate_skills]]
        match_percentage = len(matched_skills) / len(expected_skills)
        
        if match_percentage >= 0.8:
            return f"Excellent match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.6:
            return f"Good match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.4:
            return f"Moderate match: {len(matched_skills)}/{len(expected_skills)} skills"
        else:
            return f"Poor match: {len(matched_skills)}/{len(expected_skills)} skills"
    
    def calculate_overall_score(self, metrics: RAGEvaluationMetrics) -> float:
        """Weighted average: relevancy 30%, precision 30%, faithfulness 20%, correctness 20%."""
        weights = {
            'answer_relevancy': 0.30,
            'context_precision': 0.30,
            'faithfulness': 0.20,
            'answer_correctness': 0.20
        }
        
        overall_score = (
            metrics.answer_relevancy * weights['answer_relevancy'] +
            metrics.context_precision * weights['context_precision'] +
            metrics.faithfulness * weights['faithfulness'] +
            metrics.answer_correctness * weights['answer_correctness']
        )
        
        return overall_score
    
    def store_evaluation(self, query: str, metrics: RAGEvaluationMetrics):
        """Append {query, timestamp, metrics} to the in-memory evaluation history."""
        evaluation_record = {
            'query': query,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics.to_dict()
        }
        
        self.evaluation_history.append(evaluation_record)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Return the number of evaluations, the average of every metric and the 5 most recent records."""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['answer_relevancy', 'context_precision', 'faithfulness', 
                      'answer_correctness', 'overall_score']:
            values = [record['metrics'][metric] for record in self.evaluation_history]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_metrics': avg_metrics,
            'recent_evaluations': self.evaluation_history[-5:]
        }
    
    def export_evaluations(self, filename: str = "rag_evaluations.csv") -> bool:
        """Write the evaluation history to a CSV file (one row per evaluation). Returns True on success."""
        if not self.evaluation_history:
            return False
        
        try:
            export_data = []
            for i, record in enumerate(self.evaluation_history):
                row = {
                    'evaluation_id': i + 1,
                    'query': record['query'],
                    'timestamp': record['timestamp'],
                    **record['metrics']
                }
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            return True
            
        except Exception as e:
            logger.error(f"Failed to export evaluations: {e}")
            return False
