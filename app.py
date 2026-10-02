"""ScholarHunter Agents - Streamlit UI."""
# --- sqlite fix for Streamlit Community Cloud (CrewAI -> chromadb needs sqlite >= 3.35) ---
import os
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import crew as sh_crew  # noqa: E402
import export  # noqa: E402
import tracker  # noqa: E402
from tools import parse_cv  # noqa: E402

st.set_page_config(page_title="ScholarHunter Agents", page_icon="🎓", layout="wide")

COUNTRIES = [
    "United Kingdom", "United States", "Germany", "Canada", "Australia", "China", "Turkey", "Hungary",
    "Japan", "South Korea", "Sweden", "Netherlands", "France", "Italy", "Malaysia", "Norway", "Finland",
]
STEPS = ["Profile", "Scout", "Architect", "Tracker"]
ICON = {"wait": "⬜", "run": "🔄", "done": "✅", "fail": "❌"}

for key, default in {
    "profile": None, "raw": [], "queries": [], "base_df": None, "work_df": None, "gap": None,
    "summary": "", "run_id": 0, "warnings": [],
}.items():
    st.session_state.setdefault(key, default)


def get_api_key() -> str:
    """Key comes from Streamlit secrets (or a .env / environment variable), never from the UI."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip().strip('"').strip("'")
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()


# ------------------------------------------------------------------ header + sidebar
st.title("🎓 ScholarHunter Agents")
st.caption("Find. Prepare. Track. Never miss a scholarship.")

with st.sidebar:
    st.header("Candidate inputs")
    cv_file = st.file_uploader("Upload CV (PDF or TXT)", type=["pdf", "txt"])
    interests = st.text_area("Research interests", height=100,
                             placeholder="e.g. federated learning, medical image analysis, low-resource NLP")
    domain = st.text_input("Target domain (optional)", placeholder="e.g. Artificial Intelligence")
    level = st.selectbox("Level", ["MS", "PhD", "Postdoc"])
    selected = st.multiselect("Countries", COUNTRIES, default=["United Kingdom", "Germany", "Turkey"])
    custom = st.text_input("Other countries (comma-separated)")
    max_searches = st.slider("Max searches", 1, 5, 5)
    with st.expander("Advanced"):
        model_id = st.text_input("Groq model ID", value=os.getenv("GROQ_MODEL", sh_crew.DEFAULT_MODEL),
                                 help="Confirm the exact ID and free-tier limits in the Groq console.")
        parallel = st.checkbox("Run Tasks 3a/3b in parallel", value=True,
                               help="Turn off if you hit Groq free-tier rate limits.")
    run_clicked = st.button("🚀 Run Agents", type="primary", use_container_width=True)
    if st.button("🔌 Test Groq connection", use_container_width=True):
        import requests

        _key = get_api_key()
        if not _key:
            st.error("GROQ_API_KEY not found in Secrets.")
        else:
            try:
                _r = requests.get("https://api.groq.com/openai/v1/models",
                                  headers={"Authorization": f"Bearer {_key}"}, timeout=15)
                st.write(f"HTTP {_r.status_code}")
                st.code(_r.text[:300])
            except Exception as exc:
                st.error(str(exc)[:300])

tab_discover, tab_gap, tab_tracker, tab_export = st.tabs(["Discover", "Gap Analysis", "Tracker", "Export"])


# ------------------------------------------------------------------ pipeline
def execute():
    api_key = get_api_key()
    countries = list(dict.fromkeys(selected + [c.strip() for c in custom.split(",") if c.strip()]))
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it under App settings > Secrets.")
        return
    if not cv_file and not interests.strip():
        st.error("Upload a CV or enter research interests.")
        return
    if not countries:
        st.error("Select at least one country.")
        return

    cv_text = ""
    if cv_file is not None:
        try:
            cv_text = parse_cv(cv_file, cv_file.name)
        except Exception as exc:
            st.error(f"Could not read the CV: {exc}")
            return

    states = {s: "wait" for s in STEPS}
    status = st.status("Running agents...", expanded=True)
    with status:
        board = st.empty()

        def paint():
            board.markdown("  →  ".join(f"{ICON[states[s]]} **{s}**" for s in STEPS))

        def step(name, state):
            states[name] = state
            paint()

        paint()
        warnings = []
        try:
            llm = sh_crew.get_llm(api_key, model_id)

            step("Profile", "run")
            profile = sh_crew.analyze_profile(cv_text, interests, domain, countries, level, llm)
            step("Profile", "done")

            step("Scout", "run")
            raw, queries, warn = sh_crew.scout_opportunities(profile, level, max_searches, llm)
            if warn:
                warnings.append(warn)
            step("Scout", "done")

            step("Architect", "run")
            records, gap, warn = sh_crew.build_database_and_gaps(profile, raw, level, llm, parallel)
            if warn:
                warnings.append(warn)
            if not records:
                warnings.append("The database agent returned no records; showing the seed list instead.")
                records = sh_crew.seed_records(level, countries)
            df = tracker.apply_state(tracker.records_to_df(records))
            step("Architect", "done")

            step("Tracker", "run")
            summary = sh_crew.tracker_summary(df, llm)
            step("Tracker", "done")
        except Exception as exc:
            for s in STEPS:
                if states[s] == "run":
                    states[s] = "fail"
            paint()
            status.update(label="Run failed", state="error")
            st.error(f"{type(exc).__name__}: {str(exc)[:400]}")
            if "access denied" in str(exc).lower():
                st.info("Groq blocked this request before it reached your account (network/IP level). "
                        "Use 'Test Groq connection' in the sidebar, then reboot the app to get a different host.")
            else:
                st.info("Tip: on rate-limit errors, untick 'Run Tasks 3a/3b in parallel' or retry in a minute.")
            return
        status.update(label="All agents finished", state="complete", expanded=False)

    st.session_state.update(
        profile=profile, raw=raw, queries=queries, gap=gap, summary=summary, warnings=warnings,
        base_df=df, work_df=df, run_id=st.session_state.run_id + 1,
    )


# ------------------------------------------------------------------ Discover
with tab_discover:
    if run_clicked:
        execute()

    base = st.session_state.base_df
    if base is None:
        st.info("Fill in the sidebar and press **Run Agents**.")
    else:
        for w in st.session_state.warnings:
            st.warning(w)
        with st.expander("Extracted profile", expanded=False):
            st.json(st.session_state.profile.model_dump())
        if st.session_state.queries:
            st.caption("Searches run: " + " | ".join(st.session_state.queries))
        st.subheader(f"Scholarships ({len(base)})")
        st.caption("Always verify deadlines on the official site. You can edit Deadline, Status and Notes.")

        editable = ["deadline", "status", "notes"]
        edited = st.data_editor(
            base,
            key=f"editor_{st.session_state.run_id}",
            hide_index=True,
            use_container_width=True,
            column_order=["id", "country", "scholarship_name", "provider", "level", "deadline", "days_left",
                          "urgency", "fit_score", "requirements", "funding", "confidence", "official_link",
                          "status", "notes"],
            disabled=[c for c in base.columns if c not in editable],
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "scholarship_name": st.column_config.TextColumn("Name", width="large"),
                "deadline": st.column_config.TextColumn("Deadline", help="YYYY-MM-DD; leave empty if unknown"),
                "days_left": st.column_config.NumberColumn("Days left"),
                "fit_score": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%d"),
                "official_link": st.column_config.LinkColumn("Link", display_text="Open"),
                "status": st.column_config.SelectboxColumn("Status", options=tracker.STATUSES, required=True),
            },
        )
        work = tracker.recompute(pd.DataFrame(edited))
        st.session_state.work_df = work
        tracker.save_state(work)


# ------------------------------------------------------------------ Gap analysis
with tab_gap:
    gap, work = st.session_state.gap, st.session_state.work_df
    if gap is None or work is None:
        st.info("Run the agents to see your gap analysis.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Strengths")
            for s in gap.strengths or ["Not enough information in the CV."]:
                st.markdown(f"- {s}")
        with c2:
            st.subheader("⚠️ Gaps")
            for g in gap.gaps or ["No gaps reported."]:
                st.markdown(f"- {g}")
        if gap.recommendations:
            st.subheader("🎯 Recommended actions")
            for r in gap.recommendations:
                st.markdown(f"- {r}")
        st.subheader("Per-scholarship fit")
        for _, r in work.sort_values("fit_score", ascending=False).iterrows():
            with st.expander(f"{r['scholarship_name']} - {r['country']}  |  fit {int(r['fit_score'])}/100"):
                st.progress(int(r["fit_score"]) / 100)
                st.markdown(f"**Provider:** {r['provider']}  \n**Deadline:** {r['deadline'] or 'unknown - verify'}  \n"
                            f"**Requirements:** {r['requirements'] or 'not listed'}")
                if r.get("top_gaps"):
                    st.markdown("**Top gaps:** " + str(r["top_gaps"]))
                if r["official_link"]:
                    st.markdown(f"[Official page]({r['official_link']})")


# ------------------------------------------------------------------ Tracker
with tab_tracker:
    work = st.session_state.work_df
    if work is None or work.empty:
        st.info("Run the agents to start tracking.")
    else:
        c = tracker.status_counts(work)
        m = st.columns(5)
        m[0].metric("Total", c["total"])
        m[1].metric("Applied", c["applied"])
        m[2].metric("Pending", c["pending"])
        m[3].metric("Remaining", c["remaining"])
        m[4].metric("Critical", c["critical"])
        st.progress(c["applied"] / c["total"] if c["total"] else 0.0,
                    text=f"{c['applied']} of {c['total']} applied")

        st.subheader("🗓️ Weekly priorities")
        st.markdown(st.session_state.summary or tracker.plain_summary(work))
        if st.button("Regenerate summary"):
            key = get_api_key()
            if key:
                with st.spinner("Writing summary..."):
                    st.session_state.summary = sh_crew.tracker_summary(work, sh_crew.get_llm(key, model_id))
                st.rerun()
            else:
                st.error("GROQ_API_KEY not found in Secrets.")

        st.subheader("🔥 Urgent deadlines")
        urgent = tracker.urgent_list(work)
        if urgent.empty:
            st.success("No critical or soon deadlines among open items.")
        else:
            st.dataframe(urgent[["scholarship_name", "country", "deadline", "days_left", "urgency", "status"]],
                         hide_index=True, use_container_width=True)
        verify = work[(work["urgency"] == "Verify") & (work["status"] != "Applied")]
        if not verify.empty:
            st.caption(f"{len(verify)} open item(s) have no confirmed deadline - check the official sites.")


# ------------------------------------------------------------------ Export
with tab_export:
    work = st.session_state.work_df
    if work is None or work.empty:
        st.info("Nothing to export yet.")
    else:
        st.write("Excel workbook with the **Scholarships** sheet (clickable links, urgency colours) and a **Gap Report** sheet.")
        st.download_button(
            "⬇️ Download Excel",
            data=export.to_excel_bytes(work, st.session_state.gap),
            file_name="scholarhunter_scholarships.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.download_button("⬇️ Download CSV", data=work[tracker.COLUMNS].to_csv(index=False).encode("utf-8"),
                           file_name="scholarhunter_scholarships.csv", mime="text/csv")
