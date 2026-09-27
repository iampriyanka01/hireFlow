"""Data schemas for HireFlow project."""

from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional

class Resume(BaseModel):
    candidate_id: str = Field(..., description="Unique ID for candidate")
    name: str = Field(..., description="Candidate full name")
    email: Optional[EmailStr] = Field(None, description="Candidate email")
    phone: Optional[str] = Field(None, description="Candidate phone")
    location: Optional[str] = Field(None, description="Candidate location")

    text: str = Field(..., description="Raw text of resume")
    skills: List[str] = Field(default_factory=list, description="skills")
    experience: Optional[int] = Field(None, description="Total years of experience")

class JobDescription(BaseModel):
    jd_id: str = Field(..., description="Unique ID for job posting")
    title: str = Field(..., description="Job title")
    text: str = Field(..., description="Full job description text")

    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    optional_skills: List[str] = Field(default_factory=list, description="Optional skills")
    min_experience: Optional[int] = Field(None, description="Minimum years of experience required")
    location: Optional[str] = Field(None, description="Job location")

class CandidateEvaluation(BaseModel):
    candidate_id: str
    fit_score: int = Field(..., ge=0, le=100, description="Fit score (0-100)")
    strengths: List[str]
    gaps: List[str]
    risks: List[str] = Field(default_factory=list)
    summary: str

    evidence: Optional[dict] = Field(default_factory=dict)

