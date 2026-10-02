"""Agents, tasks and orchestration (CrewAI + Groq)."""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional, Tuple

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402

import tools  # noqa: E402
import tracker  # noqa: E402
from schemas import CandidateProfile, GapReport, RawResult, ScholarshipRecord  # noqa: E402

MODEL = "groq/openai/gpt-oss-120b"  # the only model used by every agent
SEED_PATH = Path(__file__).parent / "data" / "seed_scholarships.json"


# ----------------------------------------------------------------------------
# LLM + helpers
# ----------------------------------------------------------------------------
def get_llm(api_key: str, temperature: float = 0.1) -> LLM:
    os.environ["GROQ_API_KEY"] = api_key
    return LLM(model=MODEL, api_key=api_key, temperature=temperature, max_tokens=6000)


def _balanced(text: str, start: int) -> Optional[str]:
    open_c = text[start]
    close_c = "}" if open_c == "{" else "]"
    depth, in_str, esc = 0, False, False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None


def extract_json(text):
    """Tolerant JSON extraction from an LLM answer (handles code fences and chatter)."""
    if text is None:
        raise ValueError("Empty model output")
    t = str(text).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    tried = 0
    for i, ch in enumerate(t):
        if ch in "{[":
            chunk = _balanced(t, i)
            if chunk:
                try:
                    return json.loads(chunk)
                except Exception:
                    tried += 1
                    if tried > 5:
                        break
    raise ValueError("Could not find valid JSON in the model output")


NO_TOOLS = "\n\nDo not call any tools or functions. Reply with plain text only."


def _run(agent_factory: Callable[[], Agent], description: str, expected: str, inputs: dict, attempts: int = 4) -> str:
    """Run one task in its own single-agent crew; back off on Groq rate limits."""
    last: Optional[Exception] = None
    for attempt in range(attempts):
        agent = agent_factory()
        task = Task(description=description + NO_TOOLS, expected_output=expected, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        try:
            out = crew.kickoff(inputs={k: str(v) for k, v in inputs.items()})
            return getattr(out, "raw", None) or str(out)
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            if attempt < attempts - 1 and any(k in msg for k in ("tool choice is none", "tool_use_fail", "called a tool")):
                time.sleep(2)  # gpt-oss sometimes emits a stray tool call; a retry normally succeeds
                continue
            if attempt < attempts - 1 and any(k in msg for k in ("rate limit", "rate_limit", "429", "tokens per minute", "tpm", "overloaded", "timeout")):
                time.sleep(20 * (attempt + 1))
                continue
            raise
    raise last  # type: ignore[misc]


def _run_json(agent_factory, description, expected, inputs):
    raw = _run(agent_factory, description, expected, inputs)
    try:
        return extract_json(raw)
    except ValueError:
        retry_desc = description + "\n\nIMPORTANT: your previous answer was not valid JSON. Return ONLY valid JSON, no prose, no code fences."
        return extract_json(_run(agent_factory, retry_desc, expected, inputs))


# ----------------------------------------------------------------------------
# Agent 1 - Profile Analyst (no tools)
# ----------------------------------------------------------------------------
def analyze_profile(cv_text: str, interests: str, domain: str, countries: List[str], level: str, llm: LLM) -> CandidateProfile:
    def factory() -> Agent:
        return Agent(
            role="Academic Profile Analyst",
            goal="Convert the CV and inputs into a precise, search-ready candidate profile.",
            backstory="You are a meticulous admissions analyst. You extract only what is stated and mark everything else as null.",
            llm=llm, tools=[], allow_delegation=False, verbose=False, max_iter=3,
        )

    description = (
        "Convert the candidate's CV and inputs into a precise, search-ready profile.\n"
        "STRICT RULES: extract only what is explicitly stated; every unknown field must be null "
        "(or an empty list for list fields). Never invent degrees, GPA, publications or test scores.\n"
        "Target level: {level}\nResearch interests: {interests}\nTarget domain: {domain}\nPreferred countries: {countries}\n\n"
        "CV TEXT:\n{cv_text}\n\n"
        "Return ONLY one JSON object with keys: name, highest_degree, field_of_study, institution, gpa, "
        "english_test, publications, research_experience (list of short strings), skills (list), "
        "research_interests (list), target_domain, target_level, countries (list), "
        "keywords (list of 8 to 12 short search-ready keywords mixing field, sub-topics and level)."
    )
    data = _run_json(
        factory, description, "A single valid JSON object describing the candidate profile.",
        dict(level=level, interests=interests or "not provided", domain=domain or "not provided",
             countries=", ".join(countries), cv_text=(cv_text or "not provided")[:6000]),
    )
    if isinstance(data, list):
        data = data[0] if data else {}
    profile = CandidateProfile(**data)

    # user-provided inputs are authoritative
    profile.countries = list(countries)
    profile.target_level = level
    if domain:
        profile.target_domain = domain
    if interests and not profile.research_interests:
        profile.research_interests = [x.strip() for x in re.split(r"[,;\n]", interests) if x.strip()]

    # make sure there are 8-12 keywords, only from stated information
    kws = list(profile.keywords)
    for extra in [*profile.research_interests, profile.field_of_study, profile.target_domain, f"{level} scholarship", "fully funded"]:
        if extra and len(kws) < 8 and extra.lower() not in [k.lower() for k in kws]:
            kws.append(extra)
    profile.keywords = kws[:12]
    return profile


# ----------------------------------------------------------------------------
# Agent 2 - Opportunity Scout (ddg_search, hard cap)
# ----------------------------------------------------------------------------
def _fallback_queries(profile: CandidateProfile, level: str) -> List[str]:
    year = date.today().year
    base = profile.target_domain or (profile.research_interests[0] if profile.research_interests else (profile.field_of_study or ""))
    qs = [f"{level} fully funded scholarship {c} {base} {year}".replace("  ", " ") for c in profile.countries]
    qs.append(f"{level} scholarship {' '.join(profile.keywords[:3])} {year}")
    qs.append(f"fully funded {level} fellowship {base} apply {year}".replace("  ", " "))
    return qs


def scout_opportunities(profile: CandidateProfile, level: str, max_searches: int, llm: LLM) -> Tuple[List[RawResult], List[str], str]:
    """Returns (raw_results deduplicated by URL, queries used, warning).

    gpt-oss-120b on Groq cannot drive CrewAI's text-based tool loop (it emits native tool calls that Groq rejects),
    so the Scout agent decides WHAT to search and the capped ddg_search implementation executes the queries.
    """
    max_searches = max(1, min(int(max_searches), 5))
    tools.reset_budget(max_searches)

    def factory() -> Agent:
        return Agent(
            role="Scholarship Intelligence Researcher",
            goal="Find currently open scholarships, fellowships and funded positions that match the candidate.",
            backstory="You are an expert funding researcher who prefers official sources (.edu, .gov, .ac. and official programme domains) over blogs.",
            llm=llm, tools=[], allow_delegation=False, verbose=False, max_iter=3,
        )

    description = (
        "Plan web searches that will find currently open scholarships, fellowships and funded positions "
        "(year {year}) for this candidate.\n"
        "Keywords: {keywords}\nCountries: {countries}\nLevel: {level}\n"
        "Write exactly {max_searches} different, specific search queries (include the level, a country, 'fully funded' "
        "and the year {year}; aim at official programme, university and government pages). "
        "Return ONLY a JSON array of {max_searches} query strings."
    )
    queries: List[str] = []
    warning = ""
    try:
        data = _run_json(factory, description, "A JSON array of search query strings.",
                         dict(year=date.today().year, keywords=", ".join(profile.keywords),
                              countries=", ".join(profile.countries), level=level, max_searches=max_searches))
        if isinstance(data, dict):
            data = data.get("queries") or next((v for v in data.values() if isinstance(v, list)), [])
        queries = [str(q).strip() for q in data if str(q).strip()]
    except Exception as exc:
        warning = f"Scout could not plan queries ({str(exc)[:100]}); built-in queries were used."

    for q in _fallback_queries(profile, level):  # top up if the agent returned fewer than needed
        if len(queries) >= max_searches:
            break
        if q not in queries:
            queries.append(q)

    for q in queries[:max_searches]:
        tools.run_search(q)  # hard cap enforced inside run_search
        time.sleep(1)

    # Results are harvested in code from the tool -> deduplicated, ranked by source trust.
    seen, raw = set(), []
    for item in tools.collected_results():
        url = item.get("url", "")
        key = tools.norm_url(url)
        if not url.startswith("http") or key in seen or tools.is_blocked(url):
            continue
        seen.add(key)
        raw.append(RawResult(title=item.get("title", ""), url=url, snippet=item.get("snippet", ""),
                             query=item.get("query", ""), trust=tools.trust_score(url)))
    raw = sorted(raw, key=lambda r: -r.trust)[:14]

    # optional enrichment: first lines of the top pages (helps find deadlines)
    pages = tools.fetch_pages([r.url for r in raw[:8]], max_chars=700)
    for r in raw:
        r.page_text = pages.get(r.url, "")
    return raw, tools.queries_used(), warning


# ----------------------------------------------------------------------------
# Seed fallback
# ----------------------------------------------------------------------------
def _load_seed() -> list:
    try:
        return json.loads(SEED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def seed_records(level: str, countries: List[str]) -> List[ScholarshipRecord]:
    out = []
    wanted = {c.lower() for c in countries}
    for s in _load_seed():
        if level and level.lower() not in s.get("level", "").lower():
            continue
        rec = ScholarshipRecord(**{**s, "deadline": None, "confidence": "low",
                                   "notes": "Seed entry - deadline unknown, verify on the official site."})
        if wanted and s.get("country", "").lower() in wanted:
            rec.notes = "Matches a selected country. " + rec.notes
        out.append(rec)
    # selected countries first
    out.sort(key=lambda r: 0 if r.country.lower() in wanted else 1)
    return out[:15]


def seed_as_raw(records: List[ScholarshipRecord]) -> List[RawResult]:
    return [RawResult(title=r.scholarship_name, url=r.official_link,
                      snippet=f"{r.provider}; {r.country}; {r.level}; funding: {r.funding}; requirements: {r.requirements}",
                      trust=3) for r in records if r.official_link]


# ----------------------------------------------------------------------------
# Agent 3 - Database Architect & Gap Mentor (Task 3a || Task 3b)
# ----------------------------------------------------------------------------
def _architect(llm: LLM) -> Agent:
    return Agent(
        role="Data Architect and Admissions Mentor",
        goal="Build a clean scholarship database and give honest gap advice.",
        backstory=("You structure messy web data into reliable records and you coach applicants honestly. "
                   "You never guess a deadline: unknown stays null with low confidence."),
        llm=llm, tools=[], allow_delegation=False, verbose=False, max_iter=3,
    )


def _task_3a(raw: List[RawResult], level: str, llm: LLM) -> List[ScholarshipRecord]:
    payload = [{"title": r.title[:120], "url": r.url, "snippet": r.snippet[:280], "page": r.page_text[:500]} for r in raw]
    description = (
        "You are given web search results about scholarships. Build a clean database.\n"
        "Create one record per DISTINCT real scholarship, fellowship or funded position (skip news, listicles and "
        "duplicates unless a page clearly describes one programme).\n"
        "RULES: use only information present in the results. NEVER guess a deadline: if no explicit date for the "
        "current cycle is shown, set deadline to null and confidence to \"low\". Use \"high\" only if the date and "
        "requirements are explicit on an official page, \"medium\" if the page is reliable but partly unclear. "
        "Use the result URL as official_link. Level is MS, PhD, Postdoc or a combination like MS/PhD. "
        "Deadline format: YYYY-MM-DD when a full date is known, else null. "
        "requirements: comma-separated subset of IELTS, GRE, SOP, Proposal, CV, Referees (others only if explicit).\n"
        "Target level: {level}\nRESULTS (JSON):\n{results_json}\n\n"
        "Return ONLY a JSON array; each item has keys: country, scholarship_name, provider, level, deadline, "
        "requirements, funding, official_link, confidence."
    )
    data = _run_json(lambda: _architect(llm), description, "A valid JSON array of scholarship records.",
                     dict(level=level, results_json=json.dumps(payload, ensure_ascii=False)))
    if isinstance(data, dict):
        data = data.get("records") or data.get("scholarships") or next((v for v in data.values() if isinstance(v, list)), [])
    records, seen = [], set()
    for item in data:
        try:
            rec = ScholarshipRecord(**item)
        except Exception:
            continue
        key = tools.norm_url(rec.official_link) or rec.scholarship_name.lower()
        if key in seen or not rec.scholarship_name:
            continue
        seen.add(key)
        records.append(rec)
    return records


def _task_3b(profile: CandidateProfile, raw: List[RawResult], llm: LLM) -> GapReport:
    payload = [{"i": i, "title": r.title[:100], "url": r.url, "snippet": r.snippet[:200]} for i, r in enumerate(raw)]
    description = (
        "Candidate profile (JSON): {profile_json}\n"
        "Scholarship candidates (JSON): {results_json}\n\n"
        "Act as an honest admissions mentor. Produce: strengths (3-6 concrete items from the profile), "
        "gaps (3-6 honest gaps versus typical requirements, e.g. no English test, no publications, no research proposal), "
        "recommendations (3-6 prioritised actions), and per_item: for every candidate its url, fit_score "
        "(integer 0-100, be strict and honest) and top_gaps (at most 3 short strings).\n"
        "Reason only from facts in the profile; a null field means information is missing, do not assume it.\n"
        "Return ONLY a JSON object with keys strengths, gaps, recommendations, per_item."
    )
    data = _run_json(lambda: _architect(llm), description, "A valid JSON object with the gap report.",
                     dict(profile_json=profile.model_dump_json(), results_json=json.dumps(payload, ensure_ascii=False)))
    if isinstance(data, list):
        data = {"per_item": data}
    return GapReport(**data)


def heuristic_fit(profile: CandidateProfile, text: str, level: str) -> int:
    t = text.lower()
    hits = 0
    for kw in profile.keywords:
        words = [w for w in re.findall(r"[a-z]{4,}", kw.lower())]
        if words and any(w in t for w in words):
            hits += 1
    score = 30 + 7 * hits + (8 if level and level.lower() in t else 0)
    return max(0, min(100, score))


def build_database_and_gaps(profile: CandidateProfile, raw: List[RawResult], level: str, llm: LLM,
                            parallel: bool = True) -> Tuple[List[ScholarshipRecord], GapReport, str]:
    """Run Task 3a and Task 3b (in parallel), then merge fit_score + top gaps in code."""
    warning = ""
    from_seed = False
    if not raw:
        records_seed = seed_records(level, profile.countries)
        raw = seed_as_raw(records_seed)
        from_seed = True
        warning = "No live search results were found; showing the built-in seed list. Verify every deadline on the official site."

    if from_seed:
        gap = _task_3b(profile, raw, llm)
        records = records_seed
    elif parallel:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_a = ex.submit(_task_3a, raw, level, llm)
            f_b = ex.submit(_task_3b, profile, raw, llm)
            records = f_a.result()
            try:
                gap = f_b.result()
            except Exception as exc:  # gap advice failing must not lose the database
                gap = GapReport()
                warning += f" Gap analysis failed: {str(exc)[:120]}"
    else:
        records = _task_3a(raw, level, llm)
        try:
            gap = _task_3b(profile, raw, llm)
        except Exception as exc:
            gap = GapReport()
            warning += f" Gap analysis failed: {str(exc)[:120]}"

    # merge in code
    by_url = {tools.norm_url(i.url): i for i in gap.per_item if i.url}
    for rec in records:
        item = by_url.get(tools.norm_url(rec.official_link))
        if item and item.fit_score:
            rec.fit_score, rec.top_gaps = item.fit_score, item.top_gaps
        else:
            rec.fit_score = heuristic_fit(profile, f"{rec.scholarship_name} {rec.provider} {rec.funding} {rec.country}", level)
            rec.top_gaps = item.top_gaps if item else []
    return records, gap, warning.strip()


# ----------------------------------------------------------------------------
# Agent 4 - Progress Tracker (logic in tracker.py; one short LLM call for the summary)
# ----------------------------------------------------------------------------
def tracker_summary(df, llm: LLM) -> str:
    counts = tracker.status_counts(df)
    urgent = tracker.urgent_list(df).head(8)
    urgent_txt = "\n".join(
        f"- {r['scholarship_name']} ({r['country']}), deadline {r['deadline']}, {int(r['days_left'])} days left, status {r['status']}"
        for _, r in urgent.iterrows()
    ) or "none"
    no_deadline = int((df["urgency"] == "Verify").sum()) if len(df) else 0

    def factory() -> Agent:
        return Agent(
            role="Application Progress Manager",
            goal="Show applied vs remaining scholarships and surface urgent deadlines.",
            backstory="You are a concise, practical application coach.",
            llm=llm, tools=[], allow_delegation=False, verbose=False, max_iter=2,
        )

    description = (
        "Write a weekly priorities summary (max 110 words, 3-5 short bullet lines) for a scholarship applicant.\n"
        "Facts: {stats}\nUrgent items:\n{urgent}\nItems without a confirmed deadline: {no_deadline}\n"
        "Use only these facts. Never invent dates. Remind them to verify deadlines on official sites."
    )
    try:
        return _run(factory, description, "3-5 short bullet lines.",
                    dict(stats=json.dumps(counts), urgent=urgent_txt, no_deadline=no_deadline), attempts=2).strip()
    except Exception:
        return tracker.plain_summary(df)
