"""
Document loading and processing for resumes and job descriptions.
Converts PDF files to LangChain Document objects with metadata.
"""

import os
from pathlib import Path
from langchain_core.documents import Document
from utils.utils import get_logger, load_pdf, split_text

logger = get_logger(__name__)

def load_resumes(directory: str) -> list:
    """Load all resume PDFs from directory and convert to Document objects"""
    resumes = []
    
    if not os.path.exists(directory):
        return resumes
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        text = load_pdf(file_path)
        
        if text:
            # TODO: HW1 [Medium] - Resumes have no skills, experience or location
            # Each resume Document only records the filename-based name and id. Nothing
            # stores skills, years of experience or location - yet the search display,
            # the filters in utils.py and the re-ranker all try to read those fields.
            # Also look at what 'name' becomes for Kevin_Torres_Resume_36.pdf.
            # Extract real metadata here. Two options: the ResumeParser in parsing.py
            # (an LLM call per resume - count the cost at 5 requests/minute), or rules,
            # since every sample resume has 'Location:' and a 'TECHNICAL SKILLS' section.
            # Hint: Session 6 - Document metadata travels with every chunk.
            doc = Document(
                page_content=text,
                metadata={
                    'source': file_path,
                    'filename': pdf_file,
                    'candidate_id': f"c_{Path(pdf_file).stem}",
                    'name': Path(pdf_file).stem.replace("_", " ").title()
                }
            )
            resumes.append(doc)
    
    return resumes

def load_job_descriptions(directory: str) -> list:
    """Load all job description PDFs from directory and convert to Document objects"""
    jobs = []
    
    if not os.path.exists(directory):
        return jobs
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
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
    """Legacy document processor class - kept for backward compatibility"""
    
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

