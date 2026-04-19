"""
DDQ Platform Plugins — 8 market-gap plugins built on top of the RAG system.

1.  Cross-DDQ Consistency Auditor
2.  LP Question Pattern + Document Gap Report
3.  Investor-Specific Memory Layer
4.  Staleness Monitor
5.  SEC/EDGAR Regulatory Filing Sync
6.  ESG Auto-Population
7.  Jurisdiction + Regulatory Language Mapping
8.  Proactive Staleness Update Orchestrator
"""
import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone

import database as db
from database import get_db, put_db, search_fund_documents

FUND_NAME = os.getenv("FUND_NAME", "the Fund")


# ── LLM helper ────────────────────────────────────────────────────────────────

def _get_client():
    """Return (OpenAI client, model) using the same key resolution as agent.py."""
    from openai import OpenAI
    keys = db.get_active_llm_keys()
    for key in keys:
        if key.get("provider") == "openai" and key.get("api_key_enc"):
            from llm_crypto import decrypt_key
            return OpenAI(api_key=decrypt_key(key["api_key_enc"])), key.get("model", "gpt-4o")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "")), "gpt-4o"


def _llm(system: str, user: str, json_mode: bool = False) -> str:
    """Blocking LLM call; returns raw text."""
    client, model = _get_client()
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=2500,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def _llm_json(system: str, user: str) -> dict:
    """LLM call that always returns a dict. Always injects 'json' into system
    prompt as required by OpenAI response_format=json_object."""
    system = "Respond only with valid JSON. No prose outside the JSON object.\n\n" + system
    try:
        return json.loads(_llm(system, user, json_mode=True))
    except Exception as e:
        return {"error": str(e)}


# ── Persistence helper ────────────────────────────────────────────────────────

def _save(plugin_name: str, session_id, result: dict):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO plugin_results (plugin_name, investor_session_id, result_json) VALUES (?,?,?)",
            (plugin_name, session_id, json.dumps(result)),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        put_db(conn)


def get_plugin_history(plugin_name: str, session_id=None, limit: int = 5) -> list:
    conn = get_db()
    try:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM plugin_results WHERE plugin_name=? AND investor_session_id=? ORDER BY created_at DESC LIMIT ?",
                (plugin_name, session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM plugin_results WHERE plugin_name=? ORDER BY created_at DESC LIMIT ?",
                (plugin_name, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        put_db(conn)


# ── Plugin 1: Cross-DDQ Consistency Auditor ───────────────────────────────────

def run_consistency_audit(session_id: int) -> dict:
    """
    Semantically compare current LP session's answers against all other sessions.
    Falls back to intra-session consistency check when only one session exists.
    """
    conn = get_db()
    try:
        # Current session answers with their questions
        current = conn.execute("""
            SELECT m.content AS answer,
                   (SELECT m2.content FROM messages m2
                    WHERE m2.conversation_id = m.conversation_id AND m2.id < m.id
                    AND m2.role = 'user' ORDER BY m2.id DESC LIMIT 1) AS question,
                   s.investor_name
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
            WHERE c.investor_session_id = ? AND m.role = 'assistant'
            ORDER BY m.created_at DESC LIMIT 25
        """, (session_id,)).fetchall()

        # All OTHER answers — other investor sessions AND general (non-investor) conversations
        others = conn.execute("""
            SELECT m.content AS answer, s.investor_name,
                   (SELECT m2.content FROM messages m2
                    WHERE m2.conversation_id = m.conversation_id AND m2.id < m.id
                    AND m2.role = 'user' ORDER BY m2.id DESC LIMIT 1) AS question
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
            WHERE (c.investor_session_id != ? OR c.investor_session_id IS NULL)
            AND m.role = 'assistant'
            ORDER BY m.created_at DESC LIMIT 60
        """, (session_id,)).fetchall()

        # Count all sessions for context
        session_count = conn.execute(
            "SELECT COUNT(*) FROM investor_sessions"
        ).fetchone()
    finally:
        put_db(conn)

    total_sessions = list(session_count.values())[0] if session_count else 0

    if not current:
        return {
            "contradictions": [],
            "summary": "No answered questions found for this session yet. Run some Q&A sessions first.",
            "mode": "cross_session",
        }

    curr_text = "\n---\n".join(
        f"Q: {(m['question'] or 'Unknown question')[:250]}\nA: {m['answer'][:500]}"
        for m in current[:12]
    )

    # ── Cross-session mode ────────────────────────────────────────────────────
    if others:
        other_text = "\n---\n".join(
            f"LP: {m['investor_name'] or 'General session'}\nQ: {(m['question'] or '')[:200]}\nA: {m['answer'][:400]}"
            for m in others[:18]
        )
        result = _llm_json(
            system="""You are a DDQ consistency auditor for a fund manager.
Compare answers given to the CURRENT LP session against answers given in OTHER sessions.
Flag genuine factual contradictions — strategy, AUM, team size, hold period, fee structure, fund metrics.
Ignore minor wording differences or reasonable variations in emphasis. Only flag real inconsistencies.
Return: {
  "contradictions": [{"topic":"","current_claim":"","other_claim":"","lp":"","severity":"high|medium|low","explanation":""}],
  "summary": "one sentence summary",
  "risk_score": 0-100,
  "mode": "cross_session"
}""",
            user=f"CURRENT SESSION ANSWERS:\n{curr_text}\n\nOTHER LP / GENERAL SESSION ANSWERS:\n{other_text}",
        )
        result["mode"] = "cross_session"
        result["sessions_compared"] = len(others)

    # ── Intra-session fallback: check this session's own answers for inconsistency ──
    else:
        result = _llm_json(
            system="""You are a DDQ consistency auditor for a fund manager.
Only one investor session exists, so check this session's own answers for internal consistency.
Flag any answers that contradict each other within this session — e.g. if AUM was stated differently in two answers, or strategy described inconsistently.
Return: {
  "contradictions": [{"topic":"","current_claim":"","other_claim":"","lp":"Within same session","severity":"high|medium|low","explanation":""}],
  "summary": "one sentence summary",
  "risk_score": 0-100,
  "mode": "intra_session"
}""",
            user=f"ALL ANSWERS IN THIS SESSION:\n{curr_text}",
        )
        result["mode"] = "intra_session"
        result["note"] = (
            f"Only {total_sessions} investor session(s) found. "
            "Checked this session's own answers for internal inconsistencies. "
            "Add more investor sessions to enable cross-LP comparison."
        )

    _save("consistency_audit", session_id, result)
    return result


# ── Plugin 2: LP Question Pattern + Document Gap Report ───────────────────────

def run_gap_report() -> dict:
    """
    Cluster all investor questions by topic, score documentation coverage,
    and generate a prioritised document production backlog.
    """
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT m.content AS question,
                   m2.sources,
                   s.investor_name
            FROM messages m
            LEFT JOIN messages m2 ON (
                m2.conversation_id = m.conversation_id AND m2.role = 'assistant'
                AND m2.id = (SELECT MIN(id) FROM messages
                             WHERE conversation_id = m.conversation_id
                             AND role = 'assistant' AND id > m.id)
            )
            LEFT JOIN conversations c ON m.conversation_id = c.id
            LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
            WHERE m.role = 'user'
            ORDER BY m.created_at DESC LIMIT 150
        """).fetchall()

        docs = conn.execute(
            "SELECT name, ai_doc_type, doc_type FROM fund_documents WHERE status='active'"
        ).fetchall()
    finally:
        put_db(conn)

    if not rows:
        return {"clusters": [], "gaps": [], "backlog": [], "summary": "No investor questions found yet."}

    q_lines = "\n".join(
        f"- {r['question'][:200]} [sources:{len(json.loads(r['sources'] or '[]'))}] LP:{r['investor_name'] or 'General'}"
        for r in rows if r["question"]
    )
    doc_lines = "\n".join(f"- {d['name']} ({d['ai_doc_type'] or d['doc_type']})" for d in docs)

    result = _llm_json(
        system="""You are a DDQ intelligence analyst. Analyse investor questions to:
1. Cluster them by topic
2. Score documentation coverage (good/partial/missing) per topic
3. Generate a prioritised document production backlog.
Return: {
  "clusters": [{"topic":"","count":N,"example_questions":[],"coverage":"good|partial|missing","risk":"high|medium|low"}],
  "gaps": [{"topic":"","description":"","question_count":N,"impact":""}],
  "backlog": [{"priority":N,"document_title":"","document_type":"","covers_n_questions":N,"why":""}],
  "summary":"","total_questions":N,"gap_coverage_percent":N
}""",
        user=f"INVESTOR QUESTIONS (source count=0 means no doc coverage):\n{q_lines}\n\nFUND DOCUMENTS:\n{doc_lines}",
    )
    _save("gap_report", None, result)
    return result


# ── Plugin 3: Investor-Specific Memory Layer ──────────────────────────────────

def build_investor_memory(session_id: int) -> dict:
    """
    Analyse all past interactions for an LP and build a preference + behaviour profile.
    Saves to investor_sessions.profile_text.
    """
    conn = get_db()
    try:
        inv = conn.execute("SELECT * FROM investor_sessions WHERE id=?", (session_id,)).fetchone()
        if not inv:
            return {"error": "Investor session not found"}

        msgs = conn.execute("""
            SELECT m.role, m.content, m.created_at
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.investor_session_id = ?
            ORDER BY m.created_at ASC LIMIT 120
        """, (session_id,)).fetchall()
    finally:
        put_db(conn)

    if not msgs:
        return {"error": "No conversation history for this investor yet — start a Q&A session first."}

    qa_text, last_q = "", ""
    for m in msgs:
        if m["role"] == "user":
            last_q = m["content"][:300]
        elif m["role"] == "assistant" and last_q:
            qa_text += f"Q: {last_q}\nA: {m['content'][:400]}\n\n"
            last_q = ""

    profile = _llm_json(
        system="""You are an LP relationship intelligence system.
Analyse conversation history to build a detailed investor preference profile.
Return: {
  "focus_areas": [],
  "response_style_preference": "brief|detailed|quantitative|qualitative|mixed",
  "sensitivity_areas": [],
  "follow_up_patterns": [],
  "jurisdiction": "",
  "sophistication_level": "sophisticated|institutional|retail",
  "key_concerns": [],
  "recommended_tone": "",
  "preferred_formats": [],
  "summary": ""
}""",
        user=f"Investor: {inv['investor_name']} ({inv.get('investor_entity','Unknown entity')})\nNotes: {inv.get('notes','')}\n\nCONVERSATION HISTORY:\n{qa_text[:4000]}",
    )

    if "error" not in profile:
        conn2 = get_db()
        try:
            conn2.execute(
                "UPDATE investor_sessions SET profile_text=?, profile_generated_at=? WHERE id=?",
                (json.dumps(profile), datetime.now(timezone.utc).isoformat(), session_id),
            )
            conn2.commit()
        finally:
            put_db(conn2)

    _save("investor_memory", session_id, profile)
    return profile


def get_investor_memory_context(session_id: int) -> str:
    """
    Return a formatted string of the LP's profile for injection into RAG context.
    Called by agent.py when generating answers for this LP.
    """
    conn = get_db()
    try:
        inv = conn.execute(
            "SELECT profile_text FROM investor_sessions WHERE id=?", (session_id,)
        ).fetchone()
    finally:
        put_db(conn)

    if not inv or not inv.get("profile_text"):
        return ""
    try:
        p = json.loads(inv["profile_text"])
        lines = []
        if p.get("focus_areas"):
            lines.append(f"LP Focus Areas: {', '.join(p['focus_areas'])}")
        if p.get("response_style_preference"):
            lines.append(f"Preferred Style: {p['response_style_preference']}")
        if p.get("sensitivity_areas"):
            lines.append(f"Sensitivity Areas: {', '.join(p['sensitivity_areas'])}")
        if p.get("recommended_tone"):
            lines.append(f"Tone: {p['recommended_tone']}")
        return "\n".join(lines)
    except Exception:
        return ""


# ── Plugin 4: Staleness Monitor ───────────────────────────────────────────────

def run_staleness_monitor() -> dict:
    """
    Find DDQ answers that are likely outdated based on age and content type.
    """
    conn = get_db()
    try:
        answers = conn.execute("""
            SELECT m.id, m.content, m.sources, m.created_at,
                   c.investor_session_id, s.investor_name, c.title
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
            WHERE m.role = 'assistant'
            ORDER BY m.created_at ASC LIMIT 200
        """).fetchall()
    finally:
        put_db(conn)

    stale_keywords = {
        "high": ["aum", "assets under management", "total raised", "fund size",
                 "team size", "headcount", "employees", "key person", "portfolio companies",
                 "irr", "tvpi", "dpi", "moic", "nav", "current valuation"],
        "medium": ["strategy", "target sectors", "investment thesis", "target return",
                  "management fee", "carry", "hurdle", "fund term"],
    }

    alerts = []
    now = datetime.now(timezone.utc)

    for ans in answers:
        try:
            created = datetime.fromisoformat(
                str(ans["created_at"]).replace("Z", "+00:00").replace(" ", "T")
            )
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (now - created).days
        except Exception:
            continue

        if age_days < 60:
            continue

        content_lower = ans["content"].lower()
        severity = None
        matched_kw = []
        for sev, kwlist in stale_keywords.items():
            hits = [kw for kw in kwlist if kw in content_lower]
            if hits:
                severity = sev
                matched_kw = hits
                break

        if severity:
            alerts.append({
                "message_id": ans["id"],
                "investor": ans["investor_name"] or "General",
                "conversation": ans["title"] or "Untitled",
                "age_days": age_days,
                "answer_preview": ans["content"][:250],
                "severity": severity,
                "stale_content_type": matched_kw[:3],
                "reason": f"{age_days}-day-old answer contains time-sensitive data: {', '.join(matched_kw[:3])}",
            })

    alerts.sort(key=lambda a: (a["severity"] == "high", a["age_days"]), reverse=True)
    result = {
        "alerts": alerts,
        "total_reviewed": len(answers),
        "stale_count": len(alerts),
        "summary": f"Reviewed {len(answers)} answers — {len(alerts)} may be stale.",
    }
    _save("staleness_monitor", None, result)
    return result


# ── Plugin 5: SEC/EDGAR Regulatory Filing Sync ────────────────────────────────

def run_edgar_sync(fund_name_query: str = "") -> dict:
    """
    Search public SEC EDGAR for filings matching the fund name.
    Flag potential DDQ ↔ public filing conflicts.
    """
    query = (fund_name_query or FUND_NAME).strip()
    encoded = urllib.parse.quote(query)

    # Use EDGAR full-text search (no auth required)
    url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{encoded}%22"
        f"&forms=ADV,D,PF&dateRange=custom&startdt=2018-01-01&enddt=2026-12-31"
    )
    headers = {"User-Agent": "DDQPlatform contact@fund.com"}

    filings = []
    edgar_error = None
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
        for hit in data.get("hits", {}).get("hits", [])[:10]:
            src = hit.get("_source", {})
            cik = src.get("cik", "")
            form = src.get("form_type", "")
            filings.append({
                "form_type": form,
                "file_date": src.get("file_date", ""),
                "entity_name": src.get("entity_name", ""),
                "cik": cik,
                "description": src.get("file_description", ""),
                "edgar_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form}&dateb=&owner=include&count=5",
            })
    except Exception as e:
        edgar_error = str(e)

    # Get recent DDQ answers for conflict check
    conn = get_db()
    try:
        recent = conn.execute(
            "SELECT content FROM messages WHERE role='assistant' ORDER BY created_at DESC LIMIT 15"
        ).fetchall()
    finally:
        put_db(conn)

    conflicts = []
    if filings and recent:
        answers_text = "\n---\n".join(r["content"][:300] for r in recent[:6])
        filings_text = json.dumps(filings[:5], indent=2)
        cr = _llm_json(
            system="""You are a compliance auditor. Given SEC filing metadata and DDQ answer excerpts,
flag potential inconsistencies (entity names, AUM, strategy, fund type, registration status).
Return: {"conflicts":[{"topic":"","filing_says":"","ddq_says":"","risk":"high|medium|low","explanation":""}],"summary":""}""",
            user=f"SEC FILINGS:\n{filings_text}\n\nDDQ ANSWERS:\n{answers_text}",
        )
        conflicts = cr.get("conflicts", [])

    result = {
        "filings": filings,
        "conflicts": conflicts,
        "edgar_error": edgar_error,
        "searched_for": query,
        "summary": (
            f"Found {len(filings)} EDGAR filings for '{query}'. "
            f"{len(conflicts)} potential conflict(s) detected."
            + (f" Note: {edgar_error}" if edgar_error else "")
        ),
    }
    _save("edgar_sync", None, result)
    return result


# ── Plugin 6: ESG Auto-Population ─────────────────────────────────────────────

def run_esg_autopop() -> dict:
    """
    Search fund documents for ESG content and auto-populate ILPA ESG DDQ sections.
    """
    queries = [
        "ESG environmental social governance policy",
        "responsible investment sustainability",
        "climate risk carbon emissions TCFD",
        "diversity equity inclusion DEI",
        "PRI UNPRI signatory responsible investment",
        "exclusion list prohibited sectors weapons tobacco",
        "SFDR Article 8 9 sustainable finance",
        "ESG monitoring portfolio company reporting",
    ]

    seen, chunks = set(), []
    for q in queries:
        for c in search_fund_documents(q, limit=3):
            key = (c.get("document_id"), c.get("chunk_text", "")[:80])
            if key not in seen:
                seen.add(key)
                chunks.append(c)

    if not chunks:
        return {
            "esg_policy": {"has_policy": False, "description": "Not documented"},
            "responsible_investment": {"framework": "none", "signatory": False},
            "esg_integration": {"how_integrated": "Not documented"},
            "exclusions": {"has_exclusion_list": False, "excluded_sectors": []},
            "climate": {"has_climate_policy": False, "tcfd_aligned": False},
            "diversity": {"dei_policy": "Not documented"},
            "reporting": {"esg_reporting_frequency": "Not documented"},
            "coverage_score": 0,
            "missing_sections": ["All ILPA ESG sections — no ESG documentation found in fund documents"],
            "summary": "No ESG content found. Upload your ESG policy, responsible investment framework, or PRI Transparency Report.",
        }

    context = "\n\n".join(
        f"[{c.get('doc_name','doc')} p.{c.get('page_ref','')}]: {c['chunk_text'][:400]}"
        for c in chunks[:20]
    )

    result = _llm_json(
        system="""You are an ESG specialist completing the ILPA DDQ ESG section.
Use the fund document extracts to populate as many fields as possible.
For missing fields write "Not documented".
Return: {
  "esg_policy":{"has_policy":bool,"description":"","source_refs":[]},
  "responsible_investment":{"framework":"PRI|UNPRI|TCFD|SFDR|other|none","signatory":bool,"details":""},
  "esg_integration":{"how_integrated":"","pre_investment":"","post_investment":""},
  "exclusions":{"has_exclusion_list":bool,"excluded_sectors":[],"source_refs":[]},
  "climate":{"has_climate_policy":bool,"tcfd_aligned":bool,"details":""},
  "diversity":{"dei_policy":"","team_diversity_data":""},
  "reporting":{"esg_reporting_frequency":"","format":""},
  "coverage_score":0-100,
  "missing_sections":[],
  "summary":""
}""",
        user=f"FUND DOCUMENT EXTRACTS:\n{context}\n\nPopulate the ILPA ESG DDQ sections.",
    )
    _save("esg_autopop", None, result)
    return result


# ── Plugin 7: Jurisdiction + Regulatory Language Mapping ──────────────────────

def run_jurisdiction_mapping(session_id: int) -> dict:
    """
    Detect LP jurisdiction from investor documents and entity name.
    Provide regulatory-compliant language adaptations for key DDQ answers.
    """
    conn = get_db()
    try:
        inv = conn.execute("SELECT * FROM investor_sessions WHERE id=?", (session_id,)).fetchone()
        if not inv:
            return {"error": "Investor session not found"}

        inv_docs = conn.execute(
            "SELECT content FROM investor_documents WHERE investor_session_id=? AND content IS NOT NULL LIMIT 3",
            (session_id,),
        ).fetchall()

        answers = conn.execute("""
            SELECT m.content FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.investor_session_id=? AND m.role='assistant'
            ORDER BY m.created_at DESC LIMIT 8
        """, (session_id,)).fetchall()
    finally:
        put_db(conn)

    doc_text = "\n\n".join(d["content"][:600] for d in inv_docs if d["content"])
    answers_text = "\n---\n".join(a["content"][:300] for a in answers)

    result = _llm_json(
        system="""You are a cross-border regulatory language specialist for fund raising.
Detect the LP's jurisdiction and provide DDQ language adaptations for local regulatory compliance.
Return: {
  "detected_jurisdiction":"",
  "confidence":"high|medium|low",
  "regulatory_regime":"AIFMD|FSA-Japan|SFC-HK|MAS-Singapore|SEC-US|DIFC|other",
  "required_disclosures":[],
  "language_adaptations":[{"topic":"","standard_language":"","adapted_language":"","regulatory_basis":""}],
  "flags":[],
  "summary":""
}""",
        user=(
            f"INVESTOR: {inv['investor_name']}\nENTITY: {inv.get('investor_entity','')}\n"
            f"NOTES: {inv.get('notes','')}\n\n"
            f"INVESTOR DOCUMENTS:\n{doc_text[:1500] or 'None uploaded'}\n\n"
            f"RECENT DDQ ANSWERS TO ADAPT:\n{answers_text[:1500] or 'None yet'}"
        ),
    )
    _save("jurisdiction_map", session_id, result)
    return result


# ── Plugin 8: Proactive Staleness Update Orchestrator ────────────────────────

def run_staleness_orchestrator() -> dict:
    """
    Identify LPs with stale answers and draft proactive investor update emails.
    """
    monitor = run_staleness_monitor()
    alerts = monitor.get("alerts", [])

    if not alerts:
        return {
            "updates_needed": [],
            "summary": "No stale answers requiring proactive LP updates detected.",
        }

    # Group by investor
    by_investor: dict = {}
    for alert in alerts:
        by_investor.setdefault(alert["investor"], []).append(alert)

    updates = []
    for investor_name, inv_alerts in list(by_investor.items())[:6]:
        previews = "\n".join(f"- {a['answer_preview'][:180]}" for a in inv_alerts[:3])
        draft = _llm_json(
            system=f"""You are an investor relations professional at {FUND_NAME}.
Draft a concise, professional proactive update email to an LP whose DDQ answers may be outdated.
The email should: acknowledge proactive outreach, note information may have evolved, offer updated materials.
Be warm but brief. Return: {{"subject":"","body":"","urgency":"high|medium|low"}}""",
            user=f"LP: {investor_name}\nPotentially outdated information:\n{previews}\n\nDraft the email.",
        )
        if "error" in draft or not draft.get("subject"):
            draft = {
                "subject": f"Proactive Update — {FUND_NAME}",
                "body": (
                    f"Dear {investor_name} team,\n\nWe are reaching out proactively to ensure the information "
                    f"we shared with you previously remains accurate and current. Given recent developments, "
                    f"we would welcome the opportunity to provide an updated briefing.\n\n"
                    f"Please let us know a convenient time.\n\nBest regards,\n{FUND_NAME} Investor Relations"
                ),
                "urgency": "medium",
            }
        updates.append({
            "investor": investor_name,
            "stale_answer_count": len(inv_alerts),
            "email_draft": draft,
            "alerts": inv_alerts,
        })

    # Sort by urgency
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    updates.sort(key=lambda u: urgency_order.get(u["email_draft"].get("urgency", "low"), 2))

    result = {
        "updates_needed": updates,
        "total_lps_affected": len(updates),
        "summary": f"{len(updates)} LP(s) may need proactive updates covering {len(alerts)} stale answer(s).",
    }
    _save("staleness_orchestrator", None, result)
    return result
