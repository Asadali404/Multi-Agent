"""Tools: capped DuckDuckGo search (CrewAI tool), page fetcher and CV parser."""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from crewai.tools import tool

try:  # the package was renamed from duckduckgo_search to ddgs
    from ddgs import DDGS
except Exception:  # pragma: no cover
    from duckduckgo_search import DDGS

# ----------------------------------------------------------------------------
# Search budget (hard cap on ddg_search calls, shared with the orchestrator)
# ----------------------------------------------------------------------------
class _Budget:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.max_calls = 5
        self.used = 0
        self.results: List[Dict[str, str]] = []
        self.queries: List[str] = []


BUDGET = _Budget()


def reset_budget(max_calls: int = 5) -> None:
    with BUDGET.lock:
        BUDGET.max_calls = max(1, min(int(max_calls), 5))
        BUDGET.used = 0
        BUDGET.results = []
        BUDGET.queries = []


def collected_results() -> List[Dict[str, str]]:
    with BUDGET.lock:
        return list(BUDGET.results)


def queries_used() -> List[str]:
    with BUDGET.lock:
        return list(BUDGET.queries)


def run_search(query: str) -> str:
    """Capped DuckDuckGo search; shared by the agent tool and the code fallback."""
    with BUDGET.lock:
        if BUDGET.used >= BUDGET.max_calls:
            return "SEARCH LIMIT REACHED. Do not call ddg_search again. Give your final answer now."
        BUDGET.used += 1
        BUDGET.queries.append(query)
        call_no = BUDGET.used

    hits = []
    last_err = ""
    for _ in range(2):
        try:
            hits = list(DDGS().text(query, max_results=8) or [])
            if hits:
                break
        except Exception as exc:  # rate limits etc.
            last_err = str(exc)
    if not hits:
        return f"No results (search {call_no}/{BUDGET.max_calls}). {last_err[:120]}"

    lines = []
    with BUDGET.lock:
        for h in hits[:6]:
            url = h.get("href") or h.get("url") or ""
            title = (h.get("title") or "").strip()
            snippet = (h.get("body") or h.get("snippet") or "").strip()
            if not url:
                continue
            BUDGET.results.append({"title": title, "url": url, "snippet": snippet, "query": query})
            lines.append(f"- {title} | {url} | {snippet[:180]}")
    return f"Search {call_no}/{BUDGET.max_calls} results:\n" + "\n".join(lines)


@tool("ddg_search")
def ddg_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input: one short, specific search query string about scholarships,
    fellowships or funded positions. Returns up to 6 results (title, url, snippet). The number of calls
    per run is strictly limited; once the limit is reached the tool returns a stop message."""
    return run_search(query)


# ----------------------------------------------------------------------------
# Domain trust
# ----------------------------------------------------------------------------
_AGGREGATORS = (
    "scholars4dev", "scholarshipportal", "scholarship-positions", "opportunitiesforyouth",
    "scholarshipsads", "youthop", "opportunitydesk", "findaphd", "fastweb", "scholarshipdb",
    "studyportals", "topuniversities", "scholarshipscorner", "afterschoolafrica",
)
_BLOCKED = ("facebook.", "youtube.", "youtu.be", "linkedin.", "reddit.", "quora.", "twitter.", "x.com", "instagram.", "tiktok.", "pinterest.")


def is_blocked(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(b in host for b in _BLOCKED)


def trust_score(url: str) -> int:
    """3 = .edu/.gov/.ac.* ; 2 = .org / official-looking ; 1 = other ; 0 = aggregator."""
    host = urlparse(url).netloc.lower()
    if any(a in host for a in _AGGREGATORS):
        return 0
    if host.endswith(".edu") or ".edu." in host or ".ac." in host or host.endswith(".gov") or ".gov." in host or host.endswith(".int"):
        return 3
    if host.endswith(".org") or any(k in host for k in ("scholarship", "fellowship", "daad", "chevening", "fulbright")):
        return 2
    return 1


def norm_url(url: str) -> str:
    p = urlparse((url or "").strip().lower())
    host = p.netloc[4:] if p.netloc.startswith("www.") else p.netloc
    return f"{host}{p.path.rstrip('/')}"


# ----------------------------------------------------------------------------
# Page fetching (optional enrichment)
# ----------------------------------------------------------------------------
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ScholarHunter/1.0)"}


def fetch_page(url: str, max_chars: int = 700) -> str:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=6)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "").lower():
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
            t.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return text[:max_chars]
    except Exception:
        return ""


def fetch_pages(urls: List[str], max_chars: int = 700) -> Dict[str, str]:
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        texts = list(ex.map(lambda u: fetch_page(u, max_chars), urls))
    return dict(zip(urls, texts))


# ----------------------------------------------------------------------------
# CV pre-processing (pypdf -> clean text). Done in code, never by an agent.
# ----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A1-\uFFFF]", "", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_cv(file_obj, filename: str = "", max_chars: int = 6000) -> str:
    if filename.lower().endswith(".txt"):
        raw = file_obj.read()
        raw = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        return clean_text(raw)[:max_chars]
    from pypdf import PdfReader

    reader = PdfReader(file_obj)
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("The PDF is password protected.")
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    text = clean_text("\n".join(pages))
    if not text:
        raise ValueError("No text found in the PDF (it may be a scanned image).")
    return text[:max_chars]
