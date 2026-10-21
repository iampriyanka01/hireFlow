"""
core/re_ranker.py  -  AI CANDIDATE EVALUATION
=============================================

Role in the pipeline
    Search returns candidates that LOOK relevant. The ReRanker reads the
    resume against the job description and produces a human-style verdict:

        (Resume, JobDescription) ──> Gemini LLM ──> CandidateEvaluation
                                                     fit_score  (0-100)
                                                     strengths  (list)
                                                     gaps       (list)
                                                     risks      (list)
                                                     summary    (text)

Two modes
    LLM mode  : evaluate_candidate() prompts Gemini and parses the answer.
    Fallback  : simple_evaluation() compares skills / years of experience with
                plain rules when no API key is set or the LLM call fails.

Public API
    evaluate_candidate(resume, jd)      -> CandidateEvaluation (one candidate)
    re_rank_candidates(candidates, jd)  -> list[CandidateEvaluation], best first
    is_available()                      -> True if the LLM client exists

Data models (Resume, JobDescription, CandidateEvaluation) live in
utils/schemas.py.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import Resume, JobDescription, CandidateEvaluation
from utils.utils import get_logger, get_rate_limiter

logger = get_logger(__name__)


class _FitAssessment(BaseModel):
    """Fix for HW16/HW17: ask the LLM to output the assessment as structured JSON,
    instead of free text that this code then has to guess the shape of.

    This replaces the old fit_score formula (50 + 15*len(strengths) - 10*len(gaps)),
    which - because the prompt always asked for exactly 3 strengths and 3 improvement
    areas - gave nearly every candidate the same score (HW16). It also replaces the
    fragile extract_section()/extract_summary() text-scraping, which is why 'gaps' was
    always empty (HW17): the prompt asked for "areas for improvement" but the code
    searched the reply for the word "gaps", a heading the LLM was never asked to write.
    """
    fit_score: int = Field(ge=0, le=100, description="Overall fit for the role, 0-100, "
                           "based on skills, experience and role alignment - not just counts")
    strengths: List[str] = Field(default_factory=list, description="Up to 3 concise strengths")
    gaps: List[str] = Field(default_factory=list, description="Up to 3 concise gaps or missing requirements")
    risks: List[str] = Field(default_factory=list, description="Any hiring risks; empty list if none")
    summary: str = Field(description="One or two sentence overall summary")

class ReRanker:
    """Scores and explains how well a candidate fits a job."""
    
    def __init__(self):
        """Create the Gemini client if an API key exists; otherwise leave `self.llm` as None.

        With `self.llm` None every evaluation transparently uses the
        rule-based simple_evaluation().
        """
        self.llm = None
        # Fix for HW16/HW17: parse the LLM's assessment as structured JSON (same
        # PydanticOutputParser technique parsing.py uses), instead of scraping free text.
        self.output_parser = PydanticOutputParser(pydantic_object=_FitAssessment)
        self.prompt_template = PromptTemplate(
            template="""You are an expert recruiter. Analyze this candidate for the job below and
give an honest, differentiated fit_score - not the same score for every candidate.

JOB: {job_text}
CANDIDATE: {resume_text}

{format_instructions}""",
            input_variables=["job_text", "resume_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        try:
            if not GOOGLE_API_KEY:
                return
                
            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GOOGLE_API_KEY,
                rate_limiter=get_rate_limiter(),  # max 5 requests/minute on the free tier
            )
        except Exception as e:
            self.llm = None

    def evaluate_candidate(self, resume: Resume, jd: JobDescription) -> CandidateEvaluation:
        """Evaluate ONE candidate against ONE job.

        Steps:
          1. no LLM available -> return simple_evaluation()
          2. prompt the LLM for strengths, improvement areas, risks, summary
          3. cut the reply into sections with extract_section()/extract_summary()
          4. compute fit_score from the number of strengths and gaps
          5. return a CandidateEvaluation (falls back to simple_evaluation()
             on any exception)
        """
        if not self.llm:
            return self.simple_evaluation(resume, jd)
        
        try:
            # Fix for HW16/HW17: ask the LLM for the score directly, as structured JSON,
            # instead of counting bullet points under headings that may not match what
            # the prompt asked for. See _FitAssessment above for what this replaces.
            chain = self.prompt_template | self.llm | self.output_parser
            assessment = chain.invoke({"job_text": jd.text, "resume_text": resume.text})
            
            return CandidateEvaluation(
                candidate_id=resume.candidate_id,
                fit_score=assessment.fit_score,
                strengths=assessment.strengths[:3],
                gaps=assessment.gaps[:3],
                risks=assessment.risks[:5],
                summary=assessment.summary
            )
            
        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            return self.simple_evaluation(resume, jd)

    def extract_section(self, text: str, section_name: str, max_items: int) -> List[str]:
        """Superseded by the structured-output fix for HW16/HW17 (evaluate_candidate() no
        longer calls this) - kept only in case other code still depends on it.

        Pull the bullet points under one heading out of the LLM's free-text reply.

        Walks the reply line by line: starts collecting after a line containing
        `section_name`, stops at the next known heading, and keeps up to
        `max_items` bullet lines ("-", "•", "*", "1." ...) with markers removed.
        Note: lines are lowercased while parsing.
        """
        items = []
        lines = text.split('\n')
        in_section = False
        
        for line in lines:
            line = line.strip().lower()
            
            if section_name in line:
                in_section = True
                continue
                
            if in_section and any(s in line for s in ['strengths:', 'gaps:', 'risks:', 'summary:']):
                break
                
            if in_section and line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                clean_item = line.lstrip('-•*1234567890. ').strip()
                if clean_item and len(items) < max_items:
                    items.append(clean_item)
        
        return items

    def extract_summary(self, text: str) -> str:
        """Superseded by the structured-output fix for HW16/HW17 - kept for compatibility.

        Return the first few non-bullet lines after the "summary" heading (joined into one string).

        Falls back to "Evaluation completed" if no summary is found.
        """
        lines = text.split('\n')
        summary_lines = []
        
        for line in lines:
            line = line.strip()
            if 'summary' in line.lower():
                start_idx = lines.index(line) + 1
                for i in range(start_idx, min(start_idx + 3, len(lines))):
                    if lines[i].strip() and not lines[i].strip().startswith(('-', '•', '*')):
                        summary_lines.append(lines[i].strip())
                break
        
        return ' '.join(summary_lines) if summary_lines else "Evaluation completed"

    def simple_evaluation(self, resume: Resume, jd: JobDescription) -> CandidateEvaluation:
        """Rule-based evaluation used when the LLM is unavailable or fails.

        Rules:
          - experience >= job's min_experience  -> strength, else gap
          - any resume skill in job's required_skills -> strength, else gap
        Score = 50 + 20 per strength - 15 per gap, clamped to 0-100.
        """
        strengths = []
        gaps = []
        
        if hasattr(resume, 'experience') and resume.experience:
            if hasattr(jd, 'min_experience') and jd.min_experience:
                if resume.experience >= jd.min_experience:
                    strengths.append(f"Meets experience requirement")
                else:
                    gaps.append(f"Below experience requirement")
        
        if hasattr(resume, 'skills') and resume.skills:
            if hasattr(jd, 'required_skills') and jd.required_skills:
                matching = [s for s in resume.skills if s.lower() in [r.lower() for r in jd.required_skills]]
                if matching:
                    strengths.append(f"Has required skills: {', '.join(matching)}")
                else:
                    gaps.append("Missing required skills")
        
        fit_score = 50 + len(strengths) * 20 - len(gaps) * 15
        fit_score = max(0, min(100, fit_score))
        
        return CandidateEvaluation(
            candidate_id=resume.candidate_id,
            fit_score=fit_score,
            strengths=strengths[:3],
            gaps=gaps[:3],
            risks=[],
            summary=f"Basic evaluation: {len(strengths)} strengths, {len(gaps)} gaps"
        )

    def re_rank_candidates(self, candidates: List[Dict[str, Any]], jd: JobDescription) -> List[CandidateEvaluation]:
        """Evaluate every candidate and return the evaluations sorted best-first.

        Accepts search-result dicts (or Resume objects). Dicts are converted to
        Resume models first (text is read from 'page_content' or 'text').
        Costs ONE LLM request per candidate, so keep the list short.
        """
        if not candidates:
            return []
        
        resume_objects = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                resume_data = {
                    "candidate_id": candidate.get("candidate_id", "unknown"),
                    "name": candidate.get("name", "Unknown"),
                    "text": candidate.get("page_content", candidate.get("text", "")),
                    "skills": candidate.get("skills", []),
                    "experience": candidate.get("experience", None)
                }
                resume_objects.append(Resume(**resume_data))
            elif isinstance(candidate, Resume):
                resume_objects.append(candidate)
        
        evaluations = []
        for resume in resume_objects:
            evaluation = self.evaluate_candidate(resume, jd)
            if evaluation:
                evaluations.append(evaluation)
        
        # Sort by score (highest first)
        evaluations.sort(key=lambda x: x.fit_score, reverse=True)
        
        return evaluations

    def is_available(self) -> bool:
        """True when the Gemini client was created (LLM mode), False in fallback mode."""
        return self.llm is not None
