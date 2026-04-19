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
- BLACK MARKET & SHADOW ECONOMY news: dark web trading, sanctioned nation crypto evasion (Iran/Russia/North Korea using BTC/USDT), shadow oil/commodity markets, underground money flows, hawala networks, illicit asset flows that move prices
- BIDDING & AUCTION MARKET news: treasury bond auction results & bid-to-cover ratios, dark pool block trades, commodity exchange auction outcomes, ETF creation/redemption arbitrage, real estate auction data, art/collectible auction results if relevant, exchange-level order book imbalances
- Market emotion & sentiment: fear/greed index, investor panic or euphoria, VIX levels
- Macro political risks: US-China tensions, Middle East instability, BRICS de-dollarization, currency wars
- Regulatory crackdowns: OFAC sanctions, SEC actions, FATF rules affecting this asset class
- Specific country-level risks affecting this fund's holdings or strategy

Search for black market AND bidding/auction news explicitly — these are required sections.

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

Focus on events most directly relevant to {fund_name} ({category}) in {current_month}.
Use country-specific news sources when available (e.g. for India funds: economictimes.com, livemint.com;
for China: caixin.com, scmp.com; for Middle East: arabnews.com, reuters.com/ME).
Be specific and targeted."""

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


# ── Agent self-training memory ────────────────────────────────────────────────

def _load_memories(category: str) -> str:
    """
    Load past learned strategies for this fund category from the DB.
    Returns a formatted string to inject into the agent's system prompt.
    """
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT memory_type, content, hits FROM fund_research_memory
               WHERE category = %s OR category = ''
               ORDER BY hits DESC, updated_at DESC LIMIT 30""",
            (category,)
        ).fetchall()
    finally:
        put_db(conn)

    if not rows:
        return ""

    lines = ["LEARNED FROM PAST RESEARCH SESSIONS (apply these first):"]
    for r in rows:
        try:
            c = json.loads(r["content"]) if isinstance(r["content"], str) else r["content"]
        except Exception:
            continue
        mtype = r["memory_type"]
        hits = r.get("hits", 1)
        if mtype == "query":
            lines.append(
                f"  ✓ PROVEN QUERY (used {hits}x): \"{c.get('query_template','')}\" "
                f"→ finds: {c.get('what_it_finds','')} "
                f"[quality: {c.get('quality','medium')}]"
            )
        elif mtype == "source":
            lines.append(
                f"  ✓ RELIABLE SOURCE: {c.get('domain','')} "
                f"→ good for: {', '.join(c.get('reliable_for',[]))}"
            )
        elif mtype == "strategy":
            lines.append(f"  ✓ STRATEGY: {c.get('insight','')}")
    return "\n".join(lines)


def _save_memories(category: str, ticker: str, search_history: list,
                   browsed_pages: list, messages: list) -> None:
    """
    After a research loop, ask LLM to extract what worked and save learnings to DB.
    Runs in background — does not block report generation.
    """
    if not search_history:
        return

    # Build a compact session summary for the LLM
    domains_hit = []
    for page in browsed_pages:
        for line in page.split("\n"):
            if line.startswith("[RESEARCH] SOURCE:") or line.startswith("SOURCE:"):
                url = line.split("SOURCE:")[-1].strip()
                try:
                    from urllib.parse import urlparse
                    d = urlparse(url).netloc
                    if d and d not in domains_hit:
                        domains_hit.append(d)
                except Exception:
                    pass

    session_summary = (
        f"Fund category: {category} | Ticker: {ticker}\n"
        f"Queries run ({len(search_history)}): {json.dumps(search_history)}\n"
        f"Domains browsed: {json.dumps(domains_hit)}\n"
    )
    # Include only last few agent turns to infer what found data
    agent_turns = [m for m in messages if m.get("role") == "assistant"][-4:]
    session_summary += "Last agent reasoning:\n" + "\n".join(
        t.get("content", "")[:300] for t in agent_turns
    )

    extraction_prompt = f"""You are reviewing a fund research session to extract reusable learnings.

{session_summary}

Extract up to 6 high-value memories. Each memory must be one of:
- "query": a search query template that reliably finds specific data
- "source": a website domain that had accurate, useful data
- "strategy": a general research insight for this fund category

Return JSON:
{{
  "memories": [
    {{
      "type": "query",
      "query_template": "{{ticker}} annual returns site:morningstar.com",
      "what_it_finds": "precise annual return percentages by year",
      "quality": "high|medium|low"
    }},
    {{
      "type": "source",
      "domain": "etf.com",
      "reliable_for": ["expense ratio", "AUM", "top holdings"]
    }},
    {{
      "type": "strategy",
      "insight": "For crypto ETFs, search CoinGecko for NAV vs price premium/discount"
    }}
  ]
}}

Only include memories that genuinely helped find real data. Skip failed queries."""

    try:
        result = _llm_json("Extract reusable research learnings.", extraction_prompt)
        memories = result.get("memories", [])
        if not memories:
            return

        conn = get_db()
        try:
            for mem in memories[:6]:
                mtype = mem.get("type", "strategy")
                content = json.dumps(mem)
                # Upsert: if similar query/domain exists for this category, increment hits
                if mtype == "query":
                    key = mem.get("query_template", "")
                    existing = conn.execute(
                        "SELECT id FROM fund_research_memory WHERE category=%s AND memory_type='query' AND content->>'query_template'=%s",
                        (category, key)
                    ).fetchone()
                elif mtype == "source":
                    key = mem.get("domain", "")
                    existing = conn.execute(
                        "SELECT id FROM fund_research_memory WHERE category=%s AND memory_type='source' AND content->>'domain'=%s",
                        (category, key)
                    ).fetchone()
                else:
                    existing = None

                if existing:
                    conn.execute(
                        "UPDATE fund_research_memory SET hits=hits+1, updated_at=NOW() WHERE id=%s",
                        (existing["id"],)
                    )
                else:
                    conn.execute(
                        """INSERT INTO fund_research_memory (category, memory_type, content)
                           VALUES (%s, %s, %s)""",
                        (category, mtype, content)
                    )
            conn.commit()
        finally:
            put_db(conn)
    except Exception:
        pass  # Memory saving is best-effort — never block report generation


def _research_agent_loop(fund_name: str, ticker: str, category: str,
                          current_year: int, current_month: str,
                          log=None) -> dict:
    """
    LLM-driven research agent. The LLM decides what to search at every step —
    performance data, holdings, risk metrics, news, sentiment — and iterates
    until it has enough evidence or reaches MAX_ROUNDS.

    Returns dict with keys: search_results, news_results, browsed_pages.
    """
    client, model = _get_client()
    label = f"{fund_name} ({ticker})" if ticker else fund_name
    MAX_ROUNDS = 12

    # What data the LLM must try to find
    CHECKLIST = """
REQUIRED DATA (mark each as found when you have it):
[ ] YTD return {year}
[ ] 1-year return
[ ] 3-year annualised return
[ ] 5-year annualised return
[ ] 10-year annualised return or since-inception
[ ] Annual return {y1} (exact %)
[ ] Annual return {y2} (exact %)
[ ] Annual return {y3} (exact %)
[ ] AUM / assets under management
[ ] Expense ratio / TER
[ ] Top 5-10 holdings with weights
[ ] Sector allocation
[ ] Geographic exposure
[ ] Max drawdown
[ ] Sharpe ratio or risk metrics
[ ] Benchmark comparison
[ ] Analyst outlook / rating for {year}
[ ] Recent news headline (at least 5 distinct items)
[ ] Fund flows / inflows-outflows
[ ] Market sentiment / investor emotion (fear-greed, VIX)
""".format(year=current_year, y1=current_year-1, y2=current_year-2, y3=current_year-3)

    def _log(phase, text):
        if log:
            log(phase, text)

    # ── Step 0: source selection ──────────────────────────────────────
    # Ask the LLM to identify the fund's home country/exchange and pick
    # the best regional data sources BEFORE any searching starts.
    _log("think", f"Identifying best data sources for {label}…")
    source_plan = _llm_json(
        "You are a fund data expert. Given a fund name, ticker and category, "
        "identify its home country/exchange and list the best websites to find "
        "performance data, holdings, risk metrics and news for it. "
        "Different countries have very different financial data sites — be specific.\n\n"
        "Examples:\n"
        "- India ETFs → moneycontrol.com, valueresearchonline.com, nseindia.com, bseindia.com, morningstar.in\n"
        "- Japan ETFs → jpx.co.jp, morningstar.co.jp, tse.or.jp, kabutan.jp\n"
        "- UK funds  → hl.co.uk, morningstar.co.uk, londonstockexchange.com, trustnet.com, citywire.co.uk\n"
        "- Germany   → boerse-frankfurt.de, comdirect.de, finanzen.net, morningstar.de\n"
        "- China     → eastmoney.com, cninfo.com.cn, sse.com.cn, szse.cn, wind.com.cn\n"
        "- Brazil    → b3.com.br, infomoney.com.br, fundamentus.com.br\n"
        "- Saudi     → tadawul.com.sa, argaam.com, mubasher.info\n"
        "- Korea     → krx.co.kr, naver.com/finance, investing.com/kr\n"
        "- Australia → asx.com.au, morningstar.com.au, intelligentinvestor.com.au\n"
        "- Canada    → tmxmoney.com, globeandmail.com, morningstar.ca\n"
        "- US        → etf.com, morningstar.com, finance.yahoo.com, etfdb.com\n"
        "- Global/Mixed → use the most liquid exchange's local site + morningstar global\n\n"
        "Return JSON:\n"
        '{{"home_country":"","home_exchange":"","primary_sites":["site1","site2","site3","site4"],'
        '"news_sites":["site1","site2"],"reasoning":""}}',
        f"Fund: {fund_name}\nTicker: {ticker}\nCategory: {category}"
    )
    home_country  = source_plan.get("home_country", "")
    home_exchange = source_plan.get("home_exchange", "")
    primary_sites = source_plan.get("primary_sites", [])
    news_sites    = source_plan.get("news_sites", [])
    site_reasoning = source_plan.get("reasoning", "")

    if primary_sites:
        sites_str = ", ".join(primary_sites)
        _log("think", f"Using regional sources for {home_country or 'this fund'}: {sites_str}")
    else:
        sites_str = "Morningstar, ETF.com, Yahoo Finance, Bloomberg"

    # ── Load past learnings ───────────────────────────────────────────
    past_memories = _load_memories(category)

    system_prompt = f"""You are a meticulous fund research analyst collecting data for {label} ({category}).
Home country / exchange: {home_country} / {home_exchange}

Your task: run web searches to gather ALL the data points in the checklist below.

REGIONAL DATA SOURCES — use these first, they are the most accurate for this fund:
  Primary: {sites_str}
  News:    {", ".join(news_sites) if news_sites else "local financial news + Reuters"}

Do NOT default to US-only sites (Morningstar.com, ETF.com) for non-US funds.
Instead use the country-specific sources listed above.

{CHECKLIST}

{past_memories}

After EACH search, assess what is still missing and search for that next.
Prioritise exact percentage returns with signs (e.g. +12.4%, -8.1%).
Start with PROVEN QUERIES from past sessions if available — they have already been validated.

At each step respond with JSON:
{{
  "reasoning": "what I found and what I still need",
  "found_items": ["list of checklist items now confirmed found"],
  "search_query": "the exact query to run next",
  "query_type": "performance|holdings|risk|news|sentiment|flows|other",
  "done": false
}}

When ALL critical items are found (or after enough rounds), set done=true:
{{
  "reasoning": "summary of all data collected",
  "found_items": [],
  "search_query": "",
  "query_type": "other",
  "done": true
}}

Be precise. Search for exact numbers. If a well-known fund, you may already know some figures — still verify with a search."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({
        "role": "user",
        "content": (
            f"Start collecting data for {label} ({home_country}) as of {current_month}. "
            f"Use {sites_str} as your primary sources. "
            f"What is the most important thing to search for first to get accurate performance numbers?"
        )
    })

    all_results = []
    news_results = []
    browsed_pages = []
    search_history = []
    round_num = 0

    _log("phase", f"Research agent starting for {label}…")

    while round_num < MAX_ROUNDS:
        round_num += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=600,
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
            _log("done", f"Research agent complete after {round_num} rounds.")
            break

        query = action.get("search_query", "").strip()
        if not query or query in search_history:
            break
        search_history.append(query)

        q_type = action.get("query_type", "other")
        _log("search", f"[{round_num}/{MAX_ROUNDS}] {query}")

        results = _web_search(query, max_results=5)

        is_news = q_type in ("news", "sentiment")
        for r in results:
            if is_news:
                r["_news"] = True
                news_results.append(r)
            else:
                all_results.append(r)

        # Browse top non-Google URL
        top_url = next(
            (r.get("url", "") for r in results
             if r.get("url", "").startswith("http") and "google" not in r.get("url", "")),
            ""
        )
        page_text = ""
        if top_url:
            _log("browse", f"Reading: {top_url}")
            char_lim = 3000 if q_type == "performance" else 2000
            page_text = _browse(top_url, char_limit=char_lim)
            if page_text and not page_text.startswith("[Browse failed"):
                tag = "[NEWS]" if is_news else "[RESEARCH]"
                browsed_pages.append(f"{tag} SOURCE: {top_url}\n{page_text[:2500]}")
            time.sleep(0.3)

        # Feed results back to agent
        snippets = []
        for r in results:
            snippets.append(
                f"[{r.get('title','')}] {r.get('snippet','')}\nURL: {r.get('url','')}"
            )
        observation = f"SEARCH RESULTS FOR: {query}\n\n" + "\n\n".join(snippets[:5])
        if page_text and not page_text.startswith("[Browse failed"):
            observation += f"\n\nFULL PAGE ({top_url}):\n{page_text[:1200]}"

        observation += "\n\nWhat checklist items are now found? What should you search for next?"
        messages.append({"role": "user", "content": observation})

    # Save learnings to memory in background thread
    import threading
    # Inject source plan into messages so _save_memories can extract site learnings
    if primary_sites:
        messages.append({
            "role": "system",
            "content": f"[META] home_country={home_country} primary_sites={primary_sites} news_sites={news_sites}"
        })
    threading.Thread(
        target=_save_memories,
        args=(category, ticker, search_history, browsed_pages, messages),
        daemon=True
    ).start()
    _log("think", f"Saving research learnings to memory for future {category} fund runs…")

    return {
        "search_results": all_results,
        "news_results": news_results,
        "browsed_pages": browsed_pages,
    }


def _gather_evidence(fund_name: str, ticker: str = "", category: str = "", log=None) -> dict:
    """Run LLM-driven research agent + geopolitical agent loop."""
    import datetime
    current_year = datetime.date.today().year
    current_month = datetime.date.today().strftime("%B %Y")

    def _log(phase, text):
        if log:
            log(phase, text)

    # Main research agent — LLM decides all queries adaptively
    evidence = _research_agent_loop(fund_name, ticker, category, current_year, current_month, log=log)

    # Geopolitical + sentiment agent loop — separate focused loop
    _log("phase", "Starting geopolitical & market sentiment agent loop…")
    geo_results = _geopolitical_agent_loop(fund_name, ticker, category, current_year, current_month, log=log)

    return {
        "search_results": evidence["search_results"],
        "news_results": evidence["news_results"],
        "browsed_pages": evidence["browsed_pages"],
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
7. black_market_news MUST contain AT LEAST 3 items — cover sanctioned-nation asset flows (Iran/Russia/North Korea crypto/oil), shadow commodity markets, underground financial networks that affect this fund's sector. Use real known events + training knowledge. Include mechanism and fund_impact for each.
8. bidding_activity MUST contain AT LEAST 3 items — cover recent treasury auctions, dark pool trading, commodity exchange auctions, or ETF arbitrage events relevant to this fund's asset class. Include bid-to-cover ratios or clearing yields where known.

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
    "black_market_news": [
      {{
        "headline": "",
        "region": "",
        "actors": "",
        "asset_or_commodity": "",
        "mechanism": "e.g. crypto tunnelling, hawala, ghost shipping, underground exchanges",
        "estimated_volume": "",
        "fund_impact": "",
        "severity": "High|Medium|Low",
        "date": "",
        "source_url": ""
      }},
      {{
        "headline": "",
        "region": "",
        "actors": "",
        "asset_or_commodity": "",
        "mechanism": "",
        "estimated_volume": "",
        "fund_impact": "",
        "severity": "High|Medium|Low",
        "date": "",
        "source_url": ""
      }},
      {{
        "headline": "",
        "region": "",
        "actors": "",
        "asset_or_commodity": "",
        "mechanism": "",
        "estimated_volume": "",
        "fund_impact": "",
        "severity": "High|Medium|Low",
        "date": "",
        "source_url": ""
      }}
    ],
    "bidding_activity": [
      {{
        "event": "",
        "venue_or_exchange": "",
        "asset": "",
        "outcome": "e.g. bid-to-cover 2.3x, clearing yield 4.25%, heavy buying",
        "sentiment": "Strong|Neutral|Weak",
        "fund_relevance": "",
        "date": "",
        "source_url": ""
      }},
      {{
        "event": "",
        "venue_or_exchange": "",
        "asset": "",
        "outcome": "",
        "sentiment": "Strong|Neutral|Weak",
        "fund_relevance": "",
        "date": "",
        "source_url": ""
      }},
      {{
        "event": "",
        "venue_or_exchange": "",
        "asset": "",
        "outcome": "",
        "sentiment": "Strong|Neutral|Weak",
        "fund_relevance": "",
        "date": "",
        "source_url": ""
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
