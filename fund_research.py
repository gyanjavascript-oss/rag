"""
Fund Research Engine — Risk Assessment Agent extension.

For each fund in the watchlist:
1. Runs targeted web searches (history, AUM, strategy, risk, outlook)
2. Browses top URLs for full article text
3. Feeds all evidence into LLM to produce a structured 5-year prediction report
"""
import json
import os
import time

from database import get_db, put_db

FUND_NAME_CTX = os.getenv("FUND_NAME", "the Fund")

# ── LLM + tool helpers ────────────────────────────────────────────────────────

def _get_client():
    from openai import OpenAI
    import database as db
    from llm_crypto import decrypt_key
    keys = db.get_active_llm_keys()
    for k in keys:
        if k.get("provider") == "openai" and k.get("api_key_enc"):
            return OpenAI(api_key=decrypt_key(k["api_key_enc"])), k.get("model", "gpt-4o")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "")), "gpt-4o"


def _web_search(query: str, max_results: int = 6) -> list:
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:500],
                })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def _browse(url: str, char_limit: int = 4000) -> str:
    try:
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (compatible; FundResearch/1.0)"})
            page.goto(url, timeout=18000, wait_until="domcontentloaded")
            html = page.content()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
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
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        return {"error": str(e)}


# ── Research pipeline ─────────────────────────────────────────────────────────

def _geopolitical_agent_loop(fund_name: str, ticker: str, category: str,
                              current_year: int, current_month: str,
                              log=None) -> list:
    """
    LLM-driven agent loop for geopolitical + market sentiment research.
    The LLM decides which searches to run based on the fund type, then
    iterates up to MAX_ROUNDS rounds, browsing pages and refining its queries.
    Returns a list of evidence strings.
    """
    client, model = _get_client()
    MAX_ROUNDS = 4
    evidence = []
    search_history = []

    system_prompt = f"""You are a geopolitical risk analyst researching how global events affect {fund_name} ({ticker or category}).

Your job is to run web searches to find:
- Active geopolitical conflicts and their market impact (wars, sanctions, proxy conflicts)
- Black/grey market activity relevant to the asset class (e.g. crypto used to bypass Iran/Russia sanctions, shadow oil markets)
- Market emotion & sentiment: fear/greed index, investor panic or euphoria, VIX levels
- Macro political risks: US-China tensions, Middle East instability, BRICS de-dollarization, currency wars
- Regulatory crackdowns: OFAC sanctions, SEC actions, FATF rules affecting this asset class
- Specific country-level risks affecting this fund's holdings or strategy

At each step, respond with JSON:
{{
  "reasoning": "why you're searching this",
  "search_query": "the exact query to run",
  "done": false
}}

When you have gathered enough evidence (after reviewing results), respond with:
{{
  "reasoning": "summary of findings",
  "search_query": "",
  "done": true
}}

Focus on events most directly relevant to {fund_name} in {current_month}. Be specific and targeted."""

    def _log(phase, text):
        if log:
            log(phase, text)

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": f"Start researching geopolitical risks for {fund_name} ({ticker or category}) as of {current_month}. What is most important to search first?"})

    _log("think", f"Starting geopolitical agent for {fund_name}…")

    for round_num in range(MAX_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            action = json.loads(raw)
        except Exception:
            break

        messages.append({"role": "assistant", "content": raw})

        reasoning = action.get("reasoning", "")
        if reasoning:
            _log("think", reasoning)

        if action.get("done"):
            _log("done", "Geopolitical research complete.")
            break

        query = action.get("search_query", "").strip()
        if not query or query in search_history:
            break
        search_history.append(query)

        _log("search", f"Searching: {query}")

        # Run search
        results = _web_search(query, max_results=5)
        snippets = []
        for r in results:
            snippets.append(f"[{r.get('title','')}] {r.get('snippet','')}\nURL: {r.get('href') or r.get('url','')}")

        # Browse top result
        top_url = next((r.get("url","") for r in results if r.get("url","").startswith("http")), "")
        page_text = ""
        if top_url:
            _log("browse", f"Reading: {top_url}")
            page_text = _browse(top_url, char_limit=2000)
            if page_text and not page_text.startswith("[Browse failed"):
                evidence.append(f"[GEO] SOURCE: {top_url}\n{page_text[:1800]}")
            time.sleep(0.3)

        observation = "SEARCH RESULTS FOR: " + query + "\n\n" + "\n\n".join(snippets[:5])
        if page_text and not page_text.startswith("[Browse failed"):
            observation += f"\n\nFULL PAGE ({top_url}):\n{page_text[:800]}"

        messages.append({"role": "user", "content": observation + "\n\nWhat should you search next? Or are you done?"})

    return evidence


def _gather_evidence(fund_name: str, ticker: str = "", category: str = "", log=None) -> dict:
    """Run targeted web searches, news searches, and optionally browse top URLs."""
    label = f"{fund_name} {ticker}".strip()
    import datetime
    current_year = datetime.date.today().year
    current_month = datetime.date.today().strftime("%B %Y")

    research_queries = [
        f"{label} annual returns {current_year-1} {current_year-2} {current_year-3} performance",
        f"{label} YTD return {current_year} year to date performance",
        f"{label} 1 year 3 year 5 year annualised return",
        f"{label} AUM assets under management net flows {current_year}",
        f"{label} investment strategy portfolio holdings sector allocation",
        f"{label} risk factors volatility drawdown sharpe ratio",
        f"{label} analyst outlook forecast {current_year} {current_year+1}",
        f"{label} expense ratio fees performance vs benchmark index",
        f"{ticker or fund_name} historical performance data morningstar",
    ]

    news_queries = [
        f"{label} news {current_month}",
        f"{label} latest update {current_year}",
        f"{ticker or fund_name} fund flows inflows outflows {current_year}",
        f"{label} SEC filing regulatory news {current_year}",
        f"{ticker or fund_name} performance news {current_year}",
        f"{label} investor outlook analyst rating {current_year}",
        f"{ticker or fund_name} ETF news today",
    ]

    def _log(phase, text):
        if log:
            log(phase, text)

    all_results = []
    browsed_content = []
    news_results = []

    # Research queries — browse top result each
    for i, q in enumerate(research_queries):
        _log("search", f"[{i+1}/{len(research_queries)}] {q}")
        results = _web_search(q, max_results=4)
        all_results.extend(results)
        for r in results[:1]:
            url = r.get("url", "")
            if url and url.startswith("http") and "google" not in url:
                _log("browse", f"Reading: {url}")
                text = _browse(url, char_limit=2500)
                if text and not text.startswith("[Browse failed"):
                    browsed_content.append(f"SOURCE: {url}\n{text[:2000]}")
                time.sleep(0.3)

    # News queries — collect snippets, browse top result
    _log("phase", "Fetching latest news…")
    for q in news_queries:
        _log("search", q)
        results = _web_search(q, max_results=5)
        for r in results:
            r["_news"] = True
        news_results.extend(results)
        for r in results[:1]:
            url = r.get("url", "")
            if url and url.startswith("http") and "google" not in url:
                _log("browse", f"Reading: {url}")
                text = _browse(url, char_limit=2000)
                if text and not text.startswith("[Browse failed"):
                    browsed_content.append(f"[NEWS] SOURCE: {url}\n{text[:1600]}")
                time.sleep(0.3)

    # Geopolitical agent loop — LLM decides what to search
    _log("phase", "Starting geopolitical & sentiment agent loop…")
    geo_results = _geopolitical_agent_loop(fund_name, ticker, category, current_year, current_month, log=log)

    return {
        "search_results": all_results,
        "news_results": news_results,
        "browsed_pages": browsed_content,
        "geo_results": geo_results,
    }


def _build_evidence_text(evidence: dict) -> str:
    lines = []

    # Research search snippets
    seen_urls = set()
    for r in evidence.get("search_results", []):
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            lines.append(f"[{r.get('title','')}] {r.get('snippet','')}\nURL: {url}")

    # News snippets
    lines.append("\n── RECENT NEWS ──")
    seen_news = set()
    for r in evidence.get("news_results", []):
        url = r.get("url", "")
        if url and url not in seen_news and url not in seen_urls:
            seen_news.add(url)
            lines.append(f"[NEWS] {r.get('title','')} — {r.get('snippet','')}\nURL: {url}")

    lines.append("\n── FULL PAGE EXTRACTS ──")
    for page in evidence.get("browsed_pages", [])[:8]:
        lines.append(page[:1800])

    geo = evidence.get("geo_results", [])
    if geo:
        lines.append("\n── GEOPOLITICAL & MARKET SENTIMENT RESEARCH ──")
        for g in geo[:6]:
            lines.append(g[:1800])

    return "\n\n".join(lines)[:22000]


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(fund_id: int, log=None) -> dict:
    """
    Full research pipeline for a watchlist fund.
    log(phase, text) is called at each step for live progress streaming.
    Saves result to fund_research_reports and returns the report dict.
    """
    def _log(phase, text):
        if log:
            log(phase, text)

    conn = get_db()
    try:
        fund = conn.execute(
            "SELECT * FROM fund_watchlist WHERE id = ?", (fund_id,)
        ).fetchone()
    finally:
        put_db(conn)

    if not fund:
        return {"error": "Fund not found in watchlist"}

    fund_name = fund["fund_name"]
    ticker = fund.get("ticker") or ""
    category = fund.get("category") or "Unknown"

    _log("phase", f"Starting research for {fund_name} ({ticker or category})…")

    # 1. Gather web evidence
    evidence = _gather_evidence(fund_name, ticker, category=category, log=log)
    evidence_text = _build_evidence_text(evidence)

    # 2. Generate structured report
    _log("phase", "All evidence gathered. Building report with AI…")
    import datetime
    current_year = datetime.date.today().year

    report = _llm_json(
        system=f"""You are a senior fund research analyst. Using the web evidence provided,
produce a comprehensive, detailed investment research report for {fund_name} ({ticker or 'no ticker'}).
The report must include factual historical data, current positioning, and a rigorous 5-year performance outlook.
Be specific with numbers, percentages, and dates wherever possible.

CRITICAL RULES:
1. For historical_performance fields (ytd, 1_year, 3_year_annualised, 5_year_annualised, 10_year_annualised): extract exact figures from the evidence. If exact figures are not in the evidence but you know them from your training data for well-known funds (e.g. IBIT, SPY, VOO, QQQ), use those known figures with a "~" prefix (e.g. "~12.5%"). Always include a sign (+ or -).
2. If a data point is truly unknown even from training knowledge, use "" — never write "Unavailable", "N/A", "Unknown", or similar.
3. notable_periods must contain real events (e.g. "2022 Bear Market", "COVID Crash 2020") with actual return figures — do not use placeholder text.
4. top_holdings and sector_allocation must list real holdings/sectors with approximate weights if known.
5. For key_milestones "why" fields: write 2-4 detailed sentences. Do NOT use vague phrases like "economic stabilization" or "market maturity" alone — always explain the specific mechanism: which policy, which sector dynamic, which regulatory change, which macroeconomic force, and why it applies to THIS fund in THAT year specifically.
6. recent_news MUST contain AT LEAST 5 distinct news items. Draw from the news evidence provided AND from your training knowledge of recent events affecting this fund. Each item must have a real or realistic headline, a date (within the last 6 months), a 1-2 sentence summary, sentiment, and a source_url (use "" if unknown). Do NOT leave items blank.

Return this exact JSON structure:
{{
  "fund_overview": {{
    "full_name": "",
    "ticker": "",
    "category": "",
    "asset_class": "",
    "fund_type": "ETF|Mutual Fund|Hedge Fund|Private Fund|Index Fund|other",
    "inception_date": "",
    "aum_usd": "",
    "expense_ratio": "",
    "benchmark": "",
    "manager": "",
    "strategy_summary": ""
  }},
  "historical_performance": {{
    "ytd": "",
    "1_year": "",
    "3_year_annualised": "",
    "5_year_annualised": "",
    "10_year_annualised": "",
    "since_inception": "",
    "max_drawdown": "",
    "sharpe_ratio": "",
    "vs_benchmark": "",
    "notable_periods": [
      {{"period": "", "return": "", "context": ""}}
    ]
  }},
  "portfolio_composition": {{
    "top_holdings": [{{"name": "", "weight": "", "sector": ""}}],
    "sector_allocation": [{{"sector": "", "weight": ""}}],
    "geographic_exposure": [{{"region": "", "weight": ""}}],
    "concentration_risk": ""
  }},
  "risk_assessment": {{
    "overall_risk_rating": "Low|Medium|High|Very High",
    "volatility_profile": "",
    "key_risks": [
      {{"risk": "", "severity": "High|Medium|Low", "explanation": ""}}
    ],
    "liquidity_risk": "",
    "correlation_to_market": ""
  }},
  "market_trends": {{
    "macro_tailwinds": [],
    "macro_headwinds": [],
    "sector_trends": [],
    "competitive_landscape": ""
  }},
  "five_year_outlook": {{
    "base_case": {{
      "scenario": "Base Case",
      "probability": "",
      "predicted_annual_return": "",
      "predicted_5yr_cumulative": "",
      "rationale": "",
      "key_assumptions": []
    }},
    "bull_case": {{
      "scenario": "Bull Case",
      "probability": "",
      "predicted_annual_return": "",
      "predicted_5yr_cumulative": "",
      "rationale": "",
      "key_assumptions": []
    }},
    "bear_case": {{
      "scenario": "Bear Case",
      "probability": "",
      "predicted_annual_return": "",
      "predicted_5yr_cumulative": "",
      "rationale": "",
      "key_assumptions": []
    }},
    "key_milestones": [
      {{"year": {current_year+1}, "expected_development": "", "why": "2-4 sentences: cite specific macro drivers, policy changes, earnings trends, or structural shifts that make this development likely in this specific year. Reference actual data points or known catalysts.", "impact_on_fund": "", "probability": "High|Medium|Low"}},
      {{"year": {current_year+2}, "expected_development": "", "why": "2-4 sentences: cite specific macro drivers, policy changes, earnings trends, or structural shifts that make this development likely in this specific year. Reference actual data points or known catalysts.", "impact_on_fund": "", "probability": "High|Medium|Low"}},
      {{"year": {current_year+3}, "expected_development": "", "why": "2-4 sentences: cite specific macro drivers, policy changes, earnings trends, or structural shifts that make this development likely in this specific year. Reference actual data points or known catalysts.", "impact_on_fund": "", "probability": "High|Medium|Low"}},
      {{"year": {current_year+5}, "expected_development": "", "why": "2-4 sentences: cite specific macro drivers, policy changes, earnings trends, or structural shifts that make this development likely in this specific year. Reference actual data points or known catalysts.", "impact_on_fund": "", "probability": "High|Medium|Low"}}
    ]
  }},
  "recent_news": [
    {{
      "headline": "",
      "date": "",
      "summary": "",
      "sentiment": "Positive|Neutral|Negative",
      "source_url": ""
    }},
    {{
      "headline": "",
      "date": "",
      "summary": "",
      "sentiment": "Positive|Neutral|Negative",
      "source_url": ""
    }},
    {{
      "headline": "",
      "date": "",
      "summary": "",
      "sentiment": "Positive|Neutral|Negative",
      "source_url": ""
    }},
    {{
      "headline": "",
      "date": "",
      "summary": "",
      "sentiment": "Positive|Neutral|Negative",
      "source_url": ""
    }},
    {{
      "headline": "",
      "date": "",
      "summary": "",
      "sentiment": "Positive|Neutral|Negative",
      "source_url": ""
    }}
  ],
  "geopolitical_risks": {{
    "overall_geo_risk": "Low|Moderate|High|Critical",
    "market_emotion": {{
      "current_sentiment": "Extreme Fear|Fear|Neutral|Greed|Extreme Greed",
      "sentiment_score": 0,
      "sentiment_score_note": "0-100 scale: 0=Extreme Fear, 100=Extreme Greed",
      "driver": "",
      "vix_level": "",
      "investor_behaviour": ""
    }},
    "active_conflicts": [
      {{
        "conflict": "",
        "region": "",
        "direct_impact_on_fund": "",
        "severity": "High|Medium|Low"
      }}
    ],
    "sanctions_black_market": [
      {{
        "event": "",
        "countries_involved": [],
        "asset_class_affected": "",
        "impact": "",
        "example": ""
      }}
    ],
    "macro_political_risks": [
      {{
        "risk": "",
        "countries": [],
        "probability": "High|Medium|Low",
        "fund_impact": ""
      }}
    ],
    "regulatory_risks": [
      {{
        "regulation": "",
        "jurisdiction": "",
        "status": "",
        "fund_impact": ""
      }}
    ],
    "geo_outlook_summary": ""
  }},
  "analyst_verdict": {{
    "conviction_rating": "Strong Buy|Buy|Hold|Underperform|Avoid",
    "conviction_score": 0,
    "conviction_score_note": "Integer from 1 to 10 only. 10 = highest conviction. Do NOT use 0-100 scale.",
    "summary": "",
    "why_will_perform_this_way": "",
    "biggest_upside_catalyst": "",
    "biggest_downside_risk": "",
    "suitable_for": "",
    "time_horizon_recommendation": ""
  }},
  "data_quality": {{
    "sources_found": 0,
    "data_completeness": "High|Medium|Low",
    "caveats": []
  }}
}}""",
        user=f"FUND: {fund_name} ({ticker})\nCATEGORY: {category}\n\nWEB RESEARCH EVIDENCE:\n{evidence_text}",
    )

    if "error" in report:
        return report

    report["generated_at"] = datetime.datetime.now().isoformat()
    report["fund_name"] = fund_name
    report["ticker"] = ticker

    # Save to DB
    conn2 = get_db()
    try:
        conn2.execute(
            """INSERT INTO fund_research_reports (fund_id, report_json, generated_at)
               VALUES (?, ?, NOW())
               ON CONFLICT (fund_id) DO UPDATE SET report_json=EXCLUDED.report_json, generated_at=NOW()""",
            (fund_id, json.dumps(report)),
        )
        conn2.execute(
            "UPDATE fund_watchlist SET last_researched_at=NOW() WHERE id=?",
            (fund_id,),
        )
        conn2.commit()
    finally:
        put_db(conn2)

    return report


def save_thoughts(fund_id: int, thoughts: list) -> None:
    """Append agent thoughts log to the saved report JSON."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT report_json FROM fund_research_reports WHERE fund_id=?", (fund_id,)
        ).fetchone()
        if not row:
            return
        try:
            rj = json.loads(row["report_json"])
        except Exception:
            rj = {}
        rj["_agent_thoughts"] = thoughts
        conn.execute(
            "UPDATE fund_research_reports SET report_json=? WHERE fund_id=?",
            (json.dumps(rj), fund_id),
        )
        conn.commit()
    finally:
        put_db(conn)


def get_latest_report(fund_id: int):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT report_json, generated_at FROM fund_research_reports WHERE fund_id=?",
            (fund_id,),
        ).fetchone()
    finally:
        put_db(conn)
    if not row:
        return None
    try:
        r = json.loads(row["report_json"])
        r["_generated_at"] = str(row["generated_at"])
        return r
    except Exception:
        return None
