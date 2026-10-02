"""Pure-Python tracker logic: deadlines, urgency, status counts, persistence."""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from schemas import ScholarshipRecord

COLUMNS = [
    "id", "country", "scholarship_name", "provider", "level", "deadline",
    "requirements", "funding", "official_link", "confidence", "fit_score",
    "status", "days_left", "urgency", "notes",
]
STATUSES = ["Remaining", "Pending", "Applied"]
STATE_PATH = Path(__file__).parent / "data" / "tracker_state.json"

_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y")


def parse_deadline(value) -> Optional[date]:
    """Return a date or None. Never guesses: anything unclear is None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "n/a", "unknown", "rolling", "tbd"}:
        return None
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    if m:
        s = m.group(0)
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        from dateutil import parser as dparser
        d = dparser.parse(s, fuzzy=False, dayfirst=False).date()
        return d if d.year >= 2000 else None
    except Exception:
        return None


def days_left(deadline, today: Optional[date] = None) -> Optional[int]:
    d = parse_deadline(deadline)
    if d is None:
        return None
    return (d - (today or date.today())).days


def urgency(days: Optional[int]) -> str:
    """Critical <=14d, Soon <=45d, OK later, Verify if unknown or already past."""
    if days is None or pd.isna(days) or days < 0:
        return "Verify"
    if days <= 14:
        return "Critical"
    if days <= 45:
        return "Soon"
    return "OK"


def records_to_df(records: Iterable[ScholarshipRecord]) -> pd.DataFrame:
    rows = []
    for r in sorted(records, key=lambda x: -x.fit_score):
        d = r.model_dump()
        d["top_gaps"] = "; ".join(r.top_gaps)
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=COLUMNS + ["top_gaps"])
    df["id"] = range(1, len(df) + 1)
    for c in COLUMNS + ["top_gaps"]:
        if c not in df.columns:
            df[c] = None
    return recompute(df)


def recompute(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute days_left and urgency; normalise deadline strings and status."""
    df = df.copy()
    if df.empty:
        return df
    parsed = df["deadline"].apply(parse_deadline)
    df["deadline"] = parsed.apply(lambda d: d.isoformat() if d else None)
    df["days_left"] = pd.array([days_left(d) for d in parsed], dtype="Int64")
    df["urgency"] = df["days_left"].apply(lambda x: urgency(None if pd.isna(x) else int(x)))
    df["status"] = df["status"].where(df["status"].isin(STATUSES), "Remaining")
    df["notes"] = df["notes"].fillna("")
    df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce").fillna(0).astype(int)
    return df


def status_counts(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return dict(total=0, applied=0, pending=0, remaining=0, critical=0, soon=0, verify=0)
    open_ = df[df["status"] != "Applied"]
    return dict(
        total=len(df),
        applied=int((df["status"] == "Applied").sum()),
        pending=int((df["status"] == "Pending").sum()),
        remaining=int((df["status"] == "Remaining").sum()),
        critical=int((open_["urgency"] == "Critical").sum()),
        soon=int((open_["urgency"] == "Soon").sum()),
        verify=int((open_["urgency"] == "Verify").sum()),
    )


def urgent_list(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    m = df[(df["status"] != "Applied") & (df["urgency"].isin(["Critical", "Soon"]))]
    return m.sort_values("days_left")


def plain_summary(df: pd.DataFrame) -> str:
    c = status_counts(df)
    lines = [
        f"- {c['applied']} applied, {c['pending']} pending, {c['remaining']} remaining out of {c['total']}.",
    ]
    for _, r in urgent_list(df).head(5).iterrows():
        lines.append(f"- {r['scholarship_name']} ({r['country']}): {int(r['days_left'])} days left ({r['urgency']}).")
    if c["verify"]:
        lines.append(f"- {c['verify']} item(s) have no confirmed deadline; verify them on the official sites.")
    return "\n".join(lines)


# ---------- persistence (status + notes + manual deadlines survive new runs) ----------
def _key(row) -> str:
    return (str(row.get("official_link") or "") or str(row.get("scholarship_name") or "")).strip().lower().rstrip("/")


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(df: pd.DataFrame) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {}
        for _, r in df.iterrows():
            state[_key(r)] = {"status": r["status"], "notes": r["notes"] or "", "deadline": r["deadline"]}
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass  # read-only filesystem on some hosts: tracker state is best-effort


def apply_state(df: pd.DataFrame) -> pd.DataFrame:
    state = load_state()
    if df.empty or not state:
        return df
    df = df.copy()
    for i, r in df.iterrows():
        s = state.get(_key(r))
        if s:
            df.at[i, "status"] = s.get("status", r["status"])
            df.at[i, "notes"] = s.get("notes", "")
            if s.get("deadline") and not r["deadline"]:
                df.at[i, "deadline"] = s["deadline"]
    return recompute(df)
