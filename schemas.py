"""Pydantic models shared by every module (lenient parsing of LLM JSON)."""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_NULLS = {"", "null", "none", "n/a", "na", "unknown", "not stated", "not specified", "nil"}


def clean_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _NULLS else s


def to_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = v.replace(";", ",").split(",")
        return [p.strip() for p in parts if p.strip() and p.strip().lower() not in _NULLS]
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip() and str(x).strip().lower() not in _NULLS]
    return [str(v)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CandidateProfile(_Base):
    name: Optional[str] = None
    highest_degree: Optional[str] = None
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    gpa: Optional[str] = None
    english_test: Optional[str] = None
    publications: Optional[str] = None
    research_experience: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    research_interests: List[str] = Field(default_factory=list)
    target_domain: Optional[str] = None
    target_level: Optional[str] = None
    countries: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    @field_validator(
        "name", "highest_degree", "field_of_study", "institution", "gpa",
        "english_test", "publications", "target_domain", "target_level", mode="before",
    )
    @classmethod
    def _opt_str(cls, v):
        return clean_str(v)

    @field_validator("research_experience", "skills", "research_interests", "countries", "keywords", mode="before")
    @classmethod
    def _lists(cls, v):
        return to_list(v)

    @field_validator("keywords")
    @classmethod
    def _dedupe_keywords(cls, v):
        seen, out = set(), []
        for k in v:
            if k.lower() not in seen:
                seen.add(k.lower())
                out.append(k)
        return out[:12]


class RawResult(_Base):
    title: str = ""
    url: str
    snippet: str = ""
    query: str = ""
    page_text: str = ""
    trust: int = 1


class ScholarshipRecord(_Base):
    id: Optional[int] = None
    country: str = "Unknown"
    scholarship_name: str
    provider: str = "Unknown"
    level: str = ""
    deadline: Optional[str] = None
    requirements: str = ""
    funding: str = ""
    official_link: str = ""
    confidence: str = "low"
    fit_score: int = 0
    top_gaps: List[str] = Field(default_factory=list)
    status: str = "Remaining"
    notes: str = ""

    @field_validator("deadline", mode="before")
    @classmethod
    def _deadline(cls, v):
        return clean_str(v)

    @field_validator("country", "provider", mode="before")
    @classmethod
    def _unknown_default(cls, v):
        return clean_str(v) or "Unknown"

    @field_validator("level", "funding", "official_link", "notes", mode="before")
    @classmethod
    def _text(cls, v):
        return clean_str(v) or ""

    @field_validator("requirements", mode="before")
    @classmethod
    def _reqs(cls, v):
        return ", ".join(to_list(v))

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        s = (clean_str(v) or "low").lower()
        return s if s in {"high", "medium", "low"} else "low"

    @field_validator("fit_score", mode="before")
    @classmethod
    def _fit(cls, v):
        try:
            return max(0, min(100, int(float(v))))
        except Exception:
            return 0

    @model_validator(mode="after")
    def _rule_unknown_deadline_low(self):
        # Rule: never guess a deadline - unknown deadline always means low confidence.
        if not self.deadline:
            self.confidence = "low"
        return self


class GapItem(_Base):
    url: str = ""
    fit_score: int = 0
    top_gaps: List[str] = Field(default_factory=list)

    @field_validator("fit_score", mode="before")
    @classmethod
    def _fit(cls, v):
        try:
            return max(0, min(100, int(float(v))))
        except Exception:
            return 0

    @field_validator("top_gaps", mode="before")
    @classmethod
    def _gaps(cls, v):
        return to_list(v)[:3]


class GapReport(_Base):
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    per_item: List[GapItem] = Field(default_factory=list)

    @field_validator("strengths", "gaps", "recommendations", mode="before")
    @classmethod
    def _lists(cls, v):
        return to_list(v)
