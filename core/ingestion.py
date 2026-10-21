"""
core/ingestion.py  -  STAGE 1: PDF files  ->  LangChain Documents
==================================================================

Role in the pipeline
    This is the entry point for data. Everything downstream (indexing,
    searching, ranking) works on the `Document` objects created here.

    data/resumes/*.pdf ─┐
                        ├─>  load_pdf()  ─>  Document(page_content, metadata)
    data/jds/*.pdf ─────┘

What is a Document?
    A LangChain `Document` is a tiny container with two parts:
      - page_content : the full text extracted from the PDF
      - metadata     : a dict of facts about it (id, name, source file ...)
    The metadata travels with the document into Pinecone, so search results
    can later be traced back to a candidate / job.

Public API
    load_resumes(directory)          -> list[Document]   (one per resume PDF)
    load_job_descriptions(directory) -> list[Document]   (one per JD PDF)
    DocumentProcessor                -> legacy helper class (kept for compatibility)

Used by
    main.py (load_and_index_documents) and the Streamlit upload flow.
"""

import os
import re
from pathlib import Path
from langchain_core.documents import Document
from utils.utils import get_logger, load_pdf, split_text

logger = get_logger(__name__)

# Headers that mark the start of a new section in every sample resume. Used to find
# where the "TECHNICAL SKILLS" list ends (fix for HW1, below).
_RESUME_SECTION_HEADERS = ["PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "SOFTWARE PROFICIENCY",
                          "PROFESSIONAL EXPERIENCE", "EDUCATION", "CERTIFICATIONS",
                          "ACHIEVEMENTS", "REFERENCES"]


def _extract_resume_facts(text: str) -> dict:
    """Fix for HW1: pull real skills/location/experience out of the resume text with
    simple rules, instead of leaving every resume's metadata empty. No LLM call - every
    sample resume follows a fixed template ('Location:' line, a 'TECHNICAL SKILLS'
    section), so regex and section-splitting recover real facts for free.

    This is intentionally a lightweight, best-effort parse (not the full ResumeParser
    LLM extraction in parsing.py) - good enough to power search filters and to stop the
    re-ranker and UI from seeing empty skills/location/experience for every candidate.

    Returns:
        {'skills': List[str], 'location': str, 'experience': int}
    """
    location_match = re.search(r"Location:\s*(.+)", text)
    location = location_match.group(1).strip() if location_match else "Unknown"

    years_match = re.search(r"(\d+)\+?\s+years?", text, re.IGNORECASE)
    experience = int(years_match.group(1)) if years_match else 0

    # Both "TECHNICAL SKILLS" (e.g. Tax Preparation, Variance Analysis) and "SOFTWARE
    # PROFICIENCY" (e.g. QuickBooks, Excel) feed the one 'skills' list, since search
    # filters and the re-ranker only look at a single 'skills' field - a candidate
    # filtered on "QuickBooks" should match even though it's a tool, not a skill.
    skills = []
    for header in ("TECHNICAL SKILLS", "SOFTWARE PROFICIENCY"):
        others = "|".join(re.escape(h) for h in _RESUME_SECTION_HEADERS if h != header)
        section = re.search(rf"{re.escape(header)}\s*\n(.*?)(?:\n(?:{others})\b|\Z)", text, re.DOTALL)
        if section:
            for line in section.group(1).splitlines():
                # Bullet glyphs vary by PDF font (seen so far: '\u2022', '-', '*', '\x7f').
                item = line.strip().lstrip("\u2022\x7f-* ").strip()
                if item:
                    skills.append(item)

    return {"skills": skills, "location": location, "experience": experience}

def load_resumes(directory: str) -> list:
    """Load every resume PDF in `directory` and return one Document per file.

    Args:
        directory: folder containing resume PDFs (e.g. "data/resumes").

    Returns:
        A list of Documents. Each has:
          page_content : the full resume text
          metadata     : source (path), filename, candidate_id, name
        An empty list is returned if the folder does not exist.
        PDFs whose text could not be extracted are silently skipped.
    """
    resumes = []
    
    if not os.path.exists(directory):
        return resumes
    
    # Only files ending in lowercase '.pdf' are picked up ('.PDF' would be skipped).
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        # load_pdf() (utils.utils) extracts the text of the PDF. A falsy result
        # (empty / unreadable PDF) makes the `if text:` check skip the file.
        text = load_pdf(file_path)
        
        if text:
            # Metadata is derived from the FILE NAME (id/name) plus the resume TEXT
            # (skills/location/experience - fix for HW1, see _extract_resume_facts above):
            #   candidate_id -> stable id, e.g. "c_Kevin_Torres_Resume_36"
            #                   (also used to de-duplicate vector results)
            #   name         -> "Kevin Torres Resume 36" (underscores -> spaces)
            facts = _extract_resume_facts(text)
            doc = Document(
                page_content=text,
                metadata={
                    'source': file_path,
                    'filename': pdf_file,
                    'candidate_id': f"c_{Path(pdf_file).stem}",
                    'name': Path(pdf_file).stem.replace("_", " ").title(),
                    'skills': facts['skills'],
                    'location': facts['location'],
                    'experience': facts['experience'],
                }
            )
            resumes.append(doc)
    
    return resumes

def load_job_descriptions(directory: str) -> list:
    """Load every job-description PDF in `directory` and return one Document per file.

    Same idea as load_resumes(), but the metadata describes a job instead of a
    person: source, filename, jd_id (e.g. "jd_Senior_Accountant_Position") and
    a human-readable title built from the file name.
    """
    jobs = []
    
    if not os.path.exists(directory):
        return jobs
    
    # Only files ending in lowercase '.pdf' are picked up ('.PDF' would be skipped).
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        # load_pdf() (utils.utils) extracts the text of the PDF. A falsy result
        # (empty / unreadable PDF) makes the `if text:` check skip the file.
        text = load_pdf(file_path)
        
        if text:
            doc = Document(
                page_content=text,
                metadata={
                    'source': file_path,
                    'filename': pdf_file,
                    'jd_id': f"jd_{Path(pdf_file).stem}",
                    'title': Path(pdf_file).stem.replace("_", " ").title()
                }
            )
            jobs.append(doc)
    
    return jobs

class DocumentProcessor:
    """Legacy chunking helper - kept for backward compatibility.

    The current pipeline indexes each resume as ONE vector (no chunking), so
    this class is not used by main.py. It wraps load_pdf() and split_text()
    with a configurable chunk size / overlap for anyone who wants chunk-level
    indexing later.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize with text chunking parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path: str):
        """Load PDF using utility function"""
        return load_pdf(file_path)
    
    def split_text(self, text: str):
        """Split text using utility function"""
        return split_text(text, self.chunk_size, self.chunk_overlap)
    
    def process_resume_pdf(self, file_path: str):
        """Process single resume PDF into Document object"""
        text = self.load_pdf(file_path)
        if text:
            return Document(
                page_content=text,
                metadata={'source': file_path}
            )
        return None

