"""
core/parsing.py  -  LLM-based STRUCTURED EXTRACTION
===================================================

Role in the pipeline
    Turns messy free text (a resume or job posting) into a clean, typed
    Python dict that follows the Pydantic schemas in utils/schemas.py.

        raw text ─> clean_text ─> truncate (8000 chars) ─> prompt
                 ─> Gemini LLM ─> PydanticOutputParser ─> dict

The LangChain "chain" used by both parsers
        prompt_template | llm | output_parser
    - prompt_template : injects the text + JSON format instructions
    - llm             : Gemini (ChatGoogleGenerativeAI), rate limited
    - output_parser   : validates the JSON against the Pydantic model

Classes
    ResumeParser  -> parse_resume(text, candidate_id)   -> dict
    JobParser     -> parse_job_description(text, jd_id) -> dict

Used by
    main.py (AI evaluation demo parses the job description with JobParser).
    Parsing costs one LLM request per document, and the free Gemini tier
    allows ~5 requests/minute, so parse selectively.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from utils.schemas import JobDescription
from utils.utils import get_logger, clean_text, truncate_text, get_rate_limiter

logger = get_logger(__name__)


class _ResumeExtraction(BaseModel):
    """Fix for HW18: a narrow schema with ONLY the fields the LLM should invent.

    The old code used PydanticOutputParser(pydantic_object=Resume) directly, which
    forced the LLM to also output candidate_id (overwritten by this code anyway) and the
    entire resume text (wasteful - risks hitting output token limits, and a stray quote
    inside the echoed text can break the JSON). It also required `email: EmailStr`, so
    one malformed address failed the WHOLE parse. Using a small extraction-only model:
      - never asks the LLM to invent an id or copy the resume back,
      - keeps email as a plain Optional[str], so a malformed address just parses as
        text instead of throwing a validation error and losing every other field.
    """
    name: str = Field(description="The candidate's full name")
    email: Optional[str] = Field(default=None, description="Email address, if present")
    phone: Optional[str] = Field(default=None, description="Phone number, if present")
    skills: List[str] = Field(default_factory=list, description="Technical skills and tools")
    experience: Optional[int] = Field(default=None, description="Total years of professional experience")
    education: Optional[str] = Field(default=None, description="Highest degree and school")

class ResumeParser:
    """Uses the Gemini LLM to extract structured fields from a resume.

    Output is a dict shaped like the `Resume` schema (name, skills, experience, ...),
    built from the narrow `_ResumeExtraction` model (fix for HW18) plus candidate_id/text
    added by this code - not parsed directly against the full `Resume` schema.
    """
    
    def __init__(self):
        """Create the LLM client, the Pydantic output parser and the prompt.

        Raises:
            ValueError: if GOOGLE_API_KEY is not configured.
        """
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for LangChain parsing")
        
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            rate_limiter=get_rate_limiter(),  # max 5 requests/minute on the free tier
        )
        
        # Fix for HW18: parse into the narrow _ResumeExtraction schema (see above), not
        # the full Resume model - the LLM is never asked to echo the resume text or
        # invent a candidate_id.
        self.output_parser = PydanticOutputParser(pydantic_object=_ResumeExtraction)
        
        self.prompt_template = PromptTemplate(
            template="""You are an expert resume parser. Given the resume text below, extract ONLY
the requested fields as JSON. Do NOT include the original resume text in your answer.

Resume Text:
{resume_text}

{format_instructions}""",
            input_variables=["resume_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def parse_resume(self, text: str, candidate_id: str) -> Dict[str, Any]:
        """Extract structured resume data (name, skills, experience) from text.

        Steps:
          1. clean_text()    - normalise whitespace / odd characters
          2. truncate_text() - cap at 8000 chars to stay inside token limits
          3. run the chain   - prompt | LLM | Pydantic parser
          4. add bookkeeping - candidate_id, original raw_text, parsing_method

        Returns:
            dict version of the parsed Resume model.
        """
        
        cleaned_text = clean_text(text)
        cleaned_text = truncate_text(cleaned_text, 8000)
        
        # LangChain pipe syntax: output of each step feeds the next one.
        chain = self.prompt_template | self.llm | self.output_parser
        
        parsed_resume = chain.invoke({"resume_text": cleaned_text})
        
        # model_dump() converts the validated Pydantic object into a plain dict.
        # Fix for HW18: candidate_id and the real resume text are added here, by this
        # code - never requested from (or trusted from) the LLM.
        result = parsed_resume.model_dump()
        result.update({
            "candidate_id": candidate_id,
            "text": text,
            "raw_text": text,
            "parsing_method": "langchain_structured"
        })
        
        return result
    
class JobParser:
    """Uses the Gemini LLM to extract structured fields from a job description.

    Output follows the `JobDescription` schema (title, required skills,
    minimum experience, ...). Mirror image of ResumeParser.
    """
    
    def __init__(self):
        """Create the LLM client, the Pydantic output parser and the JD prompt.

        Raises:
            ValueError: if GOOGLE_API_KEY is not configured.
        """
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for LangChain parsing")
        
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            rate_limiter=get_rate_limiter(),  # max 5 requests/minute on the free tier
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=JobDescription)
        
        self.prompt_template = PromptTemplate(
            template="""You are an expert job description parser. Given the following job text, extract structured information.

Job Description Text:
{job_text}

{format_instructions}

Return a valid JSON object that matches the JobDescription schema exactly.""",
            input_variables=["job_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def parse_job_description(self, text: str, jd_id: str) -> Dict[str, Any]:
        """Extract structured job data (title, skills, requirements) from text.

        Same steps as ResumeParser.parse_resume(): clean -> truncate (8000
        chars) -> prompt | LLM | parser -> add jd_id / raw_text / parsing_method.

        Returns:
            dict version of the parsed JobDescription model.
        """
        
        cleaned_text = clean_text(text)
        cleaned_text = truncate_text(cleaned_text, 8000)
        
        chain = self.prompt_template | self.llm | self.output_parser
        
        parsed_jd = chain.invoke({"job_text": cleaned_text})
        
        result = parsed_jd.model_dump()
        result.update({
            "jd_id": jd_id,
            "raw_text": text,
            "parsing_method": "langchain_structured"
        })
        
        return result
   
