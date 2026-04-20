"""
Company Research Engine — AI agent that gathers web, social media,
review, and government filing data to produce a scored investment report.
"""
import json, os, time, threading
from database import get_db, put_db

# ── Helpers (shared pattern with fund_research) ───────────────────────────────

def _get_client():
    from openai import OpenAI
    import database as _db
    from llm_crypto import decrypt_key
    for k in _db.get_active_llm_keys():
        if k.get("provider") == "openai" and k.get("api_key_enc"):
            return OpenAI(api_key=decrypt_key(k["api_key_enc"])), k.get("model", "gpt-4o")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "")), "gpt-4o"


def _web_search(query: str, max_results: int = 6) -> list:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return [
                {"title": r.get("title",""), "url": r.get("href",""), "snippet": r.get("body","")[:500]}
                for r in ddgs.text(query, max_results=max_results)
            ]
    except Exception as e:
        return [{"error": str(e)}]


def _browse(url: str, char_limit: int = 4000) -> str:
    try:
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (compatible; CompanyResearch/1.0)"})
            page.goto(url, timeout=18000, wait_until="domcontentloaded")
            html = page.content()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","form"]):
            tag.decompose()
        text = "\n".join(l for l in soup.get_text(separator="\n", strip=True).splitlines() if l.strip())
        return text[:char_limit]
    except Exception as e:
        return f"[Browse failed: {e}]"


def _llm_json(system: str, user: str) -> dict:
    client, model = _get_client()
    system = "Respond only with valid JSON. No prose outside the JSON object.\n\n" + system
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=4000,
            response_format={"type":"json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        return {"error": str(e)}


# ── Social / sentiment agent loop ─────────────────────────────────────────────

def _social_agent_loop(company_name: str, ticker: str, sector: str,
                        current_year: int, current_month: str, log=None) -> list:
    client, model = _get_client()
    MAX_ROUNDS = 5
    evidence, search_history = [], []

    def _log(p, t):
        if log: log(p, t)

    system = f"""You are a social intelligence analyst researching {company_name} ({ticker or sector}).

Find:
- Reddit discussions (r/investing, r/stocks, sector subreddits) — bull/bear sentiment
- Twitter/X trending mentions, hashtags, influencer takes
- Trustpilot / Google Reviews / App Store ratings — customer experience
- Glassdoor / Indeed — employee sentiment, culture, management score
- YouTube / podcast mentions — media coverage tone
- Negative press: lawsuits, scandals, product failures, layoffs
- Positive press: awards, partnerships, product launches, growth news

Each step respond with JSON:
{{"reasoning":"","search_query":"","query_type":"reddit|twitter|reviews|glassdoor|news|other","done":false}}
When done: {{"reasoning":"","search_query":"","query_type":"other","done":true}}"""

    messages = [
        {"role":"system","content":system},
        {"role":"user","content":f"Start social/sentiment research for {company_name} as of {current_month}."}
    ]
    _log("phase", f"Social & sentiment agent starting for {company_name}…")

    for round_num in range(MAX_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=400,
                response_format={"type":"json_object"}
            )
            raw = resp.choices[0].message.content or "{}"
            action = json.loads(raw)
        except Exception:
            break

        messages.append({"role":"assistant","content":raw})
        if action.get("reasoning"):
            _log("think", action["reasoning"])
        if action.get("done"):
            _log("done", "Social research complete.")
            break

        query = action.get("search_query","").strip()
        if not query or query in search_history:
            break
        search_history.append(query)
        _log("search", query)

        results = _web_search(query, max_results=5)
        snippets = []
        for r in results:
            snippets.append(f"[{r.get('title','')}] {r.get('snippet','')}\nURL: {r.get('url','')}")

        top_url = next((r.get("url","") for r in results if r.get("url","").startswith("http") and "google" not in r.get("url","")), "")
        if top_url:
            _log("browse", f"Reading: {top_url}")
            page = _browse(top_url, char_limit=2000)
            if page and not page.startswith("[Browse failed"):
                evidence.append(f"[SOCIAL] SOURCE: {top_url}\n{page[:1600]}")
            time.sleep(0.3)

        observation = f"RESULTS: {query}\n\n" + "\n\n".join(snippets[:5])
        observation += "\n\nWhat should you search next, or are you done?"
        messages.append({"role":"user","content":observation})

    return evidence


# ── Main research agent loop ──────────────────────────────────────────────────

def _research_agent_loop(company_name: str, ticker: str, sector: str,
                          country: str, current_year: int, current_month: str,
                          log=None) -> dict:
    client, model = _get_client()
    label = f"{company_name} ({ticker})" if ticker else company_name
    MAX_ROUNDS = 12

    def _log(p, t):
        if log: log(p, t)

    CHECKLIST = f"""
REQUIRED DATA:
[ ] Revenue (TTM or last fiscal year)
[ ] Revenue growth YoY %
[ ] Net income / profit margin
[ ] EBITDA
[ ] Debt-to-equity ratio
[ ] P/E ratio and EPS
[ ] Market capitalisation
[ ] Stock price YTD and 1-year return
[ ] 52-week high / low
[ ] Dividend yield (if any)
[ ] ROE / ROA
[ ] Number of employees
[ ] CEO / leadership
[ ] Main products / services
[ ] Key competitors and market position
[ ] Recent earnings report highlights
[ ] Annual revenue for {current_year-1}, {current_year-2}, {current_year-3}
[ ] Analyst ratings / price targets
[ ] Latest SEC/govt filing highlights (10-K, annual report)
[ ] Recent news (5+ items)
[ ] ESG / sustainability score if available
"""

    system = f"""You are a senior equity research analyst collecting data for {label} ({sector}, {country}).

SOURCE STRATEGY:
1. First search: "best financial data sites for {country} {sector} company research"
2. Use discovered sites for all subsequent queries — do NOT default to US-only sites for non-US companies
3. For govt/regulatory filings search: SEC EDGAR (US), Companies House (UK), MCA (India), SEDAR (Canada), etc.
4. For social data: Reddit, Twitter/X, Glassdoor, Trustpilot, Google Reviews
5. Report new sites in "new_sites_found" each round

{CHECKLIST}

Each step respond with JSON:
{{
  "reasoning": "what found, what still needed",
  "found_items": ["confirmed checklist items"],
  "new_sites_found": ["newly discovered domains"],
  "search_query": "exact next query",
  "query_type": "financials|stock|filings|news|competitors|management|other",
  "done": false
}}
When done: {{"reasoning":"","found_items":[],"new_sites_found":[],"search_query":"","query_type":"other","done":true}}"""

    messages = [
        {"role":"system","content":system},
        {"role":"user","content":f"Start research for {label} ({country}) as of {current_month}. First identify the best data sources, then gather all checklist items."}
    ]

    all_results, news_results, browsed_pages, search_history, discovered_sites = [], [], [], [], []
    _log("phase", f"Research agent starting for {label}…")

    for round_num in range(1, MAX_ROUNDS+1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=600,
                response_format={"type":"json_object"}
            )
            raw = resp.choices[0].message.content or "{}"
            action = json.loads(raw)
        except Exception:
            break

        messages.append({"role":"assistant","content":raw})
        if action.get("reasoning"):
            _log("think", action["reasoning"])

        for s in action.get("new_sites_found", []):
            s = s.strip().lower().replace("https://","").replace("http://","").split("/")[0]
            if s and s not in discovered_sites:
                discovered_sites.append(s)
                _log("think", f"Discovered source: {s}")

        if action.get("done"):
            _log("done", f"Research complete after {round_num} rounds.")
            break

        query = action.get("search_query","").strip()
        if not query or query in search_history:
            break
        search_history.append(query)

        q_type = action.get("query_type","other")
        _log("search", f"[{round_num}/{MAX_ROUNDS}] {query}")

        results = _web_search(query, max_results=5)
        for r in results:
            try:
                from urllib.parse import urlparse
                d = urlparse(r.get("url","")).netloc.lower()
                if d and d not in discovered_sites and "google" not in d:
                    discovered_sites.append(d)
            except Exception:
                pass
            if q_type in ("news",):
                r["_news"] = True; news_results.append(r)
            else:
                all_results.append(r)

        preferred = next((r.get("url","") for r in results if any(s in r.get("url","") for s in discovered_sites) and r.get("url","").startswith("http") and "google" not in r.get("url","")), "")
        top_url = preferred or next((r.get("url","") for r in results if r.get("url","").startswith("http") and "google" not in r.get("url","")), "")
        page_text = ""
        if top_url:
            _log("browse", f"Reading: {top_url}")
            page_text = _browse(top_url, char_limit=3000 if q_type=="financials" else 2000)
            if page_text and not page_text.startswith("[Browse failed"):
                browsed_pages.append(f"[RESEARCH] SOURCE: {top_url}\n{page_text[:2500]}")
            time.sleep(0.3)

        snippets = [f"[{r.get('title','')}] {r.get('snippet','')}\nURL: {r.get('url','')}" for r in results]
        obs = f"SEARCH RESULTS: {query}\n\n" + "\n\n".join(snippets[:5])
        if page_text and not page_text.startswith("[Browse failed"):
            obs += f"\n\nFULL PAGE ({top_url}):\n{page_text[:1200]}"
        if discovered_sites:
            obs += f"\n\nDISCOVERED SOURCES: {', '.join(discovered_sites[-10:])}"
        obs += "\n\nWhat is confirmed found? What sites discovered? What to search next?"
        messages.append({"role":"user","content":obs})

    # Save learnings
    if discovered_sites:
        messages.append({"role":"system","content":f"[META] country={country} sector={sector} discovered_sites={discovered_sites}"})
    threading.Thread(target=_save_memories, args=(sector, ticker, search_history, browsed_pages, messages), daemon=True).start()

    return {"search_results": all_results, "news_results": news_results, "browsed_pages": browsed_pages}


def _save_memories(sector: str, ticker: str, search_history: list, browsed_pages: list, messages: list):
    if not search_history:
        return
    domains = []
    for page in browsed_pages:
        for line in page.split("\n"):
            if "SOURCE:" in line:
                url = line.split("SOURCE:")[-1].strip()
                try:
                    from urllib.parse import urlparse
                    d = urlparse(url).netloc
                    if d and d not in domains: domains.append(d)
                except Exception: pass

    session_summary = f"Sector: {sector} | Ticker: {ticker}\nQueries: {json.dumps(search_history)}\nDomains: {json.dumps(domains)}"
    agent_turns = [m for m in messages if m.get("role")=="assistant"][-4:]
    session_summary += "\n" + "\n".join(t.get("content","")[:300] for t in agent_turns)

    try:
        result = _llm_json(
            "Extract reusable company research learnings.",
            f"""{session_summary}

Return JSON:
{{"memories":[
  {{"type":"query","query_template":"...","what_it_finds":"...","quality":"high|medium|low"}},
  {{"type":"source","domain":"...","reliable_for":["..."]}},
  {{"type":"strategy","insight":"..."}}
]}}"""
        )
        mems = result.get("memories", [])
        if not mems: return
        conn = get_db()
        try:
            for mem in mems[:6]:
                mtype = mem.get("type","strategy")
                content = json.dumps(mem)
                existing = None
                if mtype == "query":
                    existing = conn.execute("SELECT id FROM company_research_memory WHERE sector=%s AND memory_type='query' AND content->>'query_template'=%s", (sector, mem.get("query_template",""))).fetchone()
                elif mtype == "source":
                    existing = conn.execute("SELECT id FROM company_research_memory WHERE sector=%s AND memory_type='source' AND content->>'domain'=%s", (sector, mem.get("domain",""))).fetchone()
                if existing:
                    conn.execute("UPDATE company_research_memory SET hits=hits+1, updated_at=NOW() WHERE id=%s", (existing["id"],))
                else:
                    conn.execute("INSERT INTO company_research_memory (sector, memory_type, content) VALUES (%s,%s,%s)", (sector, mtype, content))
            conn.commit()
        finally:
            put_db(conn)
    except Exception:
        pass


# ── Gather all evidence ───────────────────────────────────────────────────────

def _gather_evidence(company_name: str, ticker: str, sector: str, country: str, log=None) -> dict:
    import datetime
    current_year = datetime.date.today().year
    current_month = datetime.date.today().strftime("%B %Y")

    def _log(p, t):
        if log: log(p, t)

    evidence = _research_agent_loop(company_name, ticker, sector, country, current_year, current_month, log=log)
    _log("phase", "Starting social media & sentiment research…")
    social = _social_agent_loop(company_name, ticker, sector, current_year, current_month, log=log)

    return {
        "search_results": evidence["search_results"],
        "news_results": evidence["news_results"],
        "browsed_pages": evidence["browsed_pages"] + social,
    }


def _build_evidence_text(ev: dict) -> str:
    lines = []
    seen = set()
    for r in ev.get("search_results", []):
        u = r.get("url","")
        if u and u not in seen:
            seen.add(u)
            lines.append(f"[{r.get('title','')}] {r.get('snippet','')}\nURL: {u}")
    lines.append("\n── NEWS ──")
    for r in ev.get("news_results", []):
        u = r.get("url","")
        if u and u not in seen:
            seen.add(u)
            lines.append(f"[NEWS] {r.get('title','')} — {r.get('snippet','')}\nURL: {u}")
    lines.append("\n── FULL PAGES ──")
    for page in ev.get("browsed_pages", [])[:10]:
        lines.append(page[:1800])
    return "\n\n".join(lines)[:24000]


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(company_id: int, log=None) -> dict:
    def _log(p, t):
        if log: log(p, t)

    conn = get_db()
    try:
        co = conn.execute("SELECT * FROM company_watchlist WHERE id=%s", (company_id,)).fetchone()
    finally:
        put_db(conn)

    if not co:
        return {"error": "Company not found"}

    company_name = co["company_name"]
    ticker  = co.get("ticker") or ""
    sector  = co.get("sector") or ""
    country = co.get("country") or ""

    _log("phase", f"Starting research for {company_name}…")
    ev = _gather_evidence(company_name, ticker, sector, country, log=log)
    evidence_text = _build_evidence_text(ev)

    _log("phase", "All evidence gathered. Building company report with AI…")
    import datetime
    current_year = datetime.date.today().year

    report = _llm_json(
        system=f"""You are a senior equity research analyst. Using all web evidence provided,
produce a comprehensive company research report for {company_name} ({ticker or 'no ticker'}).

CRITICAL RULES:
1. Use "" for truly unknown fields — never write "Unavailable", "N/A", "Unknown".
2. Score fields are integers 0-100. Overall company_score is a weighted composite.
3. sentiment_score: 0=extremely negative, 50=neutral, 100=extremely positive.
4. recent_news MUST have AT LEAST 5 items. social_posts MUST have AT LEAST 3 items.
5. customer_reviews positive_themes and negative_themes must each have 3-5 real points.
6. financial score weights: revenue_growth 25%, profitability 25%, debt_health 20%, efficiency 15%, valuation 15%.
7. investment_verdict: "Strong Buy|Buy|Hold|Sell|Avoid" based on overall_score (80+=Strong Buy, 65+=Buy, 45+=Hold, 30+=Sell, <30=Avoid).

Return this exact JSON:
{{
  "company_overview": {{
    "full_name": "", "ticker": "", "sector": "", "industry": "",
    "founded": "", "headquarters": "", "employees": "", "ceo": "",
    "website": "", "exchange": "", "market_cap": "", "description": ""
  }},
  "financial_performance": {{
    "revenue_ttm": "", "revenue_growth_yoy": "", "net_income": "",
    "profit_margin": "", "ebitda": "", "debt_to_equity": "",
    "current_ratio": "", "pe_ratio": "", "eps": "",
    "dividend_yield": "", "roe": "", "roa": "",
    "annual_revenue": [
      {{"year": {current_year-1}, "revenue": "", "growth": ""}},
      {{"year": {current_year-2}, "revenue": "", "growth": ""}},
      {{"year": {current_year-3}, "revenue": "", "growth": ""}}
    ]
  }},
  "stock_performance": {{
    "current_price": "", "ytd": "", "1_year": "", "3_year": "",
    "52w_high": "", "52w_low": "", "beta": "",
    "vs_sector": "", "vs_sp500": ""
  }},
  "social_sentiment": {{
    "overall_sentiment": "Positive|Mixed|Negative",
    "sentiment_score": 50,
    "reddit_sentiment": "", "twitter_sentiment": "",
    "news_sentiment": "",
    "trending_topics": [],
    "viral_posts": [
      {{"platform":"","content":"","reaction":"positive|negative|neutral","engagement":""}}
    ]
  }},
  "customer_reviews": {{
    "overall_rating": "",
    "trustpilot_rating": "", "google_rating": "",
    "glassdoor_rating": "", "app_store_rating": "",
    "positive_themes": [],
    "negative_themes": [],
    "review_volume": "",
    "nps_estimate": ""
  }},
  "regulatory_filings": {{
    "latest_annual_report_year": "",
    "key_filing_highlights": [],
    "legal_issues": [],
    "govt_contracts": [],
    "regulatory_actions": []
  }},
  "competitive_position": {{
    "market_share": "",
    "main_competitors": [
      {{"name":"","ticker":"","vs_this_company":""}}
    ],
    "competitive_moat": "",
    "strengths": [],
    "weaknesses": []
  }},
  "growth_outlook": {{
    "growth_drivers": [],
    "expansion_plans": "",
    "product_pipeline": "",
    "addressable_market": "",
    "1_year_outlook": "",
    "3_year_outlook": ""
  }},
  "risk_assessment": {{
    "overall_risk": "Low|Medium|High|Very High",
    "key_risks": [
      {{"risk":"","severity":"High|Medium|Low","detail":""}}
    ],
    "esg_score": "",
    "esg_concerns": []
  }},
  "recent_news": [
    {{"headline":"","date":"","summary":"","sentiment":"Positive|Neutral|Negative","source_url":""}},
    {{"headline":"","date":"","summary":"","sentiment":"Positive|Neutral|Negative","source_url":""}},
    {{"headline":"","date":"","summary":"","sentiment":"Positive|Neutral|Negative","source_url":""}},
    {{"headline":"","date":"","summary":"","sentiment":"Positive|Neutral|Negative","source_url":""}},
    {{"headline":"","date":"","summary":"","sentiment":"Positive|Neutral|Negative","source_url":""}}
  ],
  "social_posts": [
    {{"platform":"Reddit|Twitter|LinkedIn|Other","content":"","sentiment":"Positive|Neutral|Negative","url":""}},
    {{"platform":"Reddit|Twitter|LinkedIn|Other","content":"","sentiment":"Positive|Neutral|Negative","url":""}},
    {{"platform":"Reddit|Twitter|LinkedIn|Other","content":"","sentiment":"Positive|Neutral|Negative","url":""}}
  ],
  "company_score": {{
    "overall_score": 0,
    "financial_score": 0,
    "sentiment_score": 0,
    "growth_score": 0,
    "risk_score": 0,
    "customer_score": 0,
    "management_score": 0,
    "investment_verdict": "Hold",
    "verdict_summary": "",
    "biggest_opportunity": "",
    "biggest_threat": "",
    "suitable_for": "",
    "time_horizon": ""
  }},
  "data_quality": {{
    "sources_found": 0,
    "completeness": "High|Medium|Low",
    "caveats": []
  }}
}}""",
        user=f"COMPANY: {company_name} ({ticker})\nSECTOR: {sector}\nCOUNTRY: {country}\n\nEVIDENCE:\n{evidence_text}"
    )

    if "error" in report:
        return report

    import datetime as _dt
    report["generated_at"] = _dt.datetime.now().isoformat()
    report["company_name"] = company_name
    report["ticker"] = ticker

    conn2 = get_db()
    try:
        conn2.execute(
            """INSERT INTO company_research_reports (company_id, report_json, generated_at)
               VALUES (%s, %s, NOW())
               ON CONFLICT (company_id) DO UPDATE SET report_json=EXCLUDED.report_json, generated_at=NOW()""",
            (company_id, json.dumps(report))
        )
        conn2.commit()
    finally:
        put_db(conn2)

    return report


def save_thoughts(company_id: int, thoughts: list) -> None:
    conn = get_db()
    try:
        row = conn.execute("SELECT report_json FROM company_research_reports WHERE company_id=%s", (company_id,)).fetchone()
        if not row: return
        try:
            rj = json.loads(row["report_json"])
        except Exception:
            rj = {}
        rj["_agent_thoughts"] = thoughts
        conn.execute("UPDATE company_research_reports SET report_json=%s WHERE company_id=%s", (json.dumps(rj), company_id))
        conn.commit()
    finally:
        put_db(conn)


def get_latest_report(company_id: int):
    conn = get_db()
    try:
        row = conn.execute("SELECT report_json FROM company_research_reports WHERE company_id=%s", (company_id,)).fetchone()
        if not row: return None
        return json.loads(row["report_json"])
    except Exception:
        return None
    finally:
        put_db(conn)
