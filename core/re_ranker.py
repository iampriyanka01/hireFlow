"""
AI-powered candidate evaluation and ranking system.
Uses LLM to assess candidate fit and provide detailed feedback.
"""

from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import Resume, JobDescription, CandidateEvaluation
from utils.utils import get_logger, get_rate_limiter

logger = get_logger(__name__)

class ReRanker:
    """LLM-powered candidate evaluation with detailed fit analysis"""
    
    def __init__(self):
        """Initialize LLM for candidate evaluation or fall back to rule-based"""
        self.llm = None
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
        """Analyze candidate fit using LLM to generate detailed evaluation"""
        if not self.llm:
            return self.simple_evaluation(resume, jd)
        
        try:
            prompt = f"""
            Analyze this candidate for the job:

            JOB: {jd.text}
            CANDIDATE: {resume.text}

            Give me:
            1. 3 key strengths
            2. 3 areas for improvement  
            3. Any risks
            4. A brief summary

            Format with clear sections.
            """
            
            response = self.llm.invoke(prompt)
            eval_text = response.content
            
            strengths = self.extract_section(eval_text, "strengths", 3)
            # TODO: HW17 [Easy] - Gaps are always empty
            # This searches the reply for a heading containing 'gaps'. What heading did
            # the prompt above ask the LLM to use? Also: extract_section() lowercases
            # every line it returns - check how strengths look in the UI.
            gaps = self.extract_section(eval_text, "gaps", 3)
            risks = self.extract_section(eval_text, "risks", 5)
            summary = self.extract_summary(eval_text)
            
            # TODO: HW16 [Medium] - Every candidate gets the same fit score
            # The prompt asks for exactly 3 strengths and 3 improvements. Work out what
            # this formula returns for every candidate. Does the ranking mean anything?
            # Ask the LLM for the score itself, as structured output - parsing.py already
            # uses PydanticOutputParser; the same technique works here.
            # Hint: Session 6 - Part 5, output parsers.
            fit_score = min(100, max(0, 50 + len(strengths) * 15 - len(gaps) * 10))
            
            return CandidateEvaluation(
                candidate_id=resume.candidate_id,
                fit_score=fit_score,
                strengths=strengths,
                gaps=gaps,
                risks=risks,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            return self.simple_evaluation(resume, jd)

    def extract_section(self, text: str, section_name: str, max_items: int) -> List[str]:
        """Parse LLM response to extract specific sections (strengths, gaps, risks)"""
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
        """Extract summary section from LLM evaluation response"""
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
        """Fallback rule-based evaluation when LLM is unavailable"""
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
        """Evaluate and rank all candidates by job fit score"""
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
        """Check if LLM is available for evaluation"""
        return self.llm is not None
