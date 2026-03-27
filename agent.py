"""
DDQ Agent: OpenAI GPT-4o with RAG tools for answering investor due diligence questions.

Flow:
1. Search fund documents for relevant context
2. (Optional) Search investor-specific documents
3. Find similar previously answered questions
4. Generate a source-cited answer + professional draft response
5. Extract themes for analytics
"""
import json
import os
from openai import OpenAI
from database import (
    search_fund_documents,
    search_assigned_fund_documents,
    search_investor_documents,
    find_similar_questions,
    search_kb,
    search_session_answers,
    get_active_llm_keys,
    log_llm_usage,
    get_env_key_sentinel_id,
)
from llm_crypto import decrypt_key

FUND_NAME = os.getenv("FUND_NAME", "the Fund")
MODEL = "gpt-4o"


_OPENAI_COMPAT_PROVIDERS = {"openai", "groq", "mistral", "ollama", "openrouter", "custom"}

_PROVIDER_BASE_URLS = {
    "groq":       "https://api.groq.com/openai/v1",
    "mistral":    "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama":     "http://localhost:11434/v1",
}


def _get_clients() -> list:
    """Returns [(key_id, model, provider, client_or_anthropic), ...] sorted by priority. Falls back to env var."""
    keys = get_active_llm_keys()
    result = []
    for k in keys:
        try:
            provider = k.get("provider", "openai").lower()
            model = k["model"]
            raw_key = decrypt_key(k["api_key_enc"])
            key_id = k["id"]
            base_url = k.get("base_url") or _PROVIDER_BASE_URLS.get(provider, "")

            if provider == "anthropic":
                try:
                    import anthropic as _anthropic
                    client = _anthropic.Anthropic(api_key=raw_key)
                    result.append((key_id, model, "anthropic", client))
                except ImportError:
                    pass
            elif provider in _OPENAI_COMPAT_PROVIDERS:
                kwargs = {"api_key": raw_key}
                if base_url:
                    kwargs["base_url"] = base_url
                elif provider == "ollama":
                    kwargs["base_url"] = "http://localhost:11434/v1"
                client = OpenAI(**kwargs)
                result.append((key_id, model, provider, client))
        except Exception:
            continue

    if result:
        return result
    # Fallback to env var — use sentinel ID so usage is still tracked
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return [(get_env_key_sentinel_id(), MODEL, "openai", OpenAI(api_key=env_key))]
    return []


# Keep old name as alias used elsewhere
def _get_openai_clients() -> list:
    """Legacy alias — returns only OpenAI-compatible entries as (key_id, model, client)."""
    return [(kid, mdl, cli) for kid, mdl, prov, cli in _get_clients()
            if prov != "anthropic"]

_SYSTEM_PROMPT_TEMPLATE = """You are an expert DDQ (Due Diligence Questionnaire) assistant for {fund_name}.
You also act as a Research Agent capable of deep web research for external market and company questions.

TODAY'S DATE: {today}
Your training data is outdated — always use web_search to get current figures. When searching for financial data, always include the current year ({year}) in your query (e.g. "Apple market cap {year}").

QUESTION TYPE ROUTING — decide the type first:

TYPE A — Fund questions (about the fund itself, its strategy, team, portfolio, fees, terms, performance):
  1. ALWAYS call search_fund_documents first
  2. If documents don't fully cover external context (benchmarks, sector trends, news) → also call web_search
  3. For investor-specific docs → also call search_investor_documents

TYPE B — External research questions (market caps, company valuations, financial data, news, industry trends, any question about external companies or markets):
  1. Call web_search with a query that includes the current year: e.g. "Apple Inc market cap {year}"
  2. ALWAYS follow with browse_url on the most relevant result URL to get accurate current data
  3. If the first browse doesn't answer it, try another URL
  4. Do NOT search fund documents for external company questions

DETECTING QUESTION TYPE:
- If the question mentions a specific company (Google, Apple, Microsoft, etc.) or asks for market data, valuations, prices, or external news → TYPE B
- If the question uses terms like "market cap", "stock price", "net worth", "valuation" of a company → TYPE B
- If uncertain, check fund documents first (TYPE A)

RESEARCH WORKFLOW FOR TYPE B (web research):
1. web_search: use specific query with current year e.g. "Google Alphabet market cap {year}" (never use past years)
2. browse_url: open the best result (prefer companiesmarketcap.com, finance.yahoo.com, reuters.com, macrotrends.net)
3. If browse returns good data → compose answer with actual numbers and the date they were retrieved
4. If not → try another URL or different web_search query
5. NEVER say "I couldn't find results" after only one search attempt — retry with different terms

RESPONSE FORMAT — you MUST return ONLY valid JSON, no other text:
{{
  "answer": "Detailed factual answer with actual numbers and the date retrieved. Never say you couldn't find results without retrying.",
  "sources": [
    {{"doc_name": "exact document name or URL", "doc_id": 0, "section": "section or page reference", "excerpt": "short verbatim quote"}}
  ],
  "draft_response": "Professional response. Start with 'Thank you for your question...'",
  "themes": ["theme1", "theme2"],
  "confidence": "high|medium|low",
  "gaps": "Describe any gaps, otherwise null",
  "chart_data": {{
    "type": "bar or line or pie (choose best fit)",
    "title": "Chart title",
    "labels": ["label1", "label2", "..."],
    "datasets": [
      {{"label": "Series name", "data": [1.0, 2.0, "..."]}}
    ]
  }}
}}

CHART DATA RULES:
- Include chart_data ONLY when the answer contains 2+ comparable numerical values (market caps, financials, performance, rankings, allocations, historical series, comparisons)
- For a single company's historical market cap → line chart with dates as labels
- For comparing multiple companies → bar chart
- For portfolio allocations or percentages that sum to 100 → pie chart
- Values must be plain numbers (no $ signs or commas) — use billions (e.g. 3000 for $3 trillion, 182 for $182B)
- Add a "unit" key like "unit": "USD Billions" or "unit": "%" so the chart can label axes
- If no chart is appropriate, set chart_data to null

RULES:
- TODAY is {today} — always search for {year} data, never reference 2023 or older years
- For external company/market questions: web_search + browse_url first, NOT fund documents
- Always browse at least one URL after web_search to get actual data (not just snippets)
- For fund questions: always search fund documents first
- Never say "I couldn't find results" without having tried at least 2 different searches
- THEMES must be from: fee structure, performance, portfolio, governance, legal structure,
  investment strategy, ESG, liquidity, reporting, LP rights, key person, risk management,
  compliance, tax, track record, team, fund terms, co-investment, distributions, valuation, other"""


def _get_system_prompt() -> str:
    from datetime import date
    today = date.today()
    return _SYSTEM_PROMPT_TEMPLATE.format(
        fund_name=FUND_NAME,
        today=today.strftime("%B %d, %Y"),
        year=today.year,
    )


# Keep a module-level alias for any code that references SYSTEM_PROMPT directly
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE  # replaced dynamically via _get_system_prompt()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_fund_documents",
            "description": (
                "Search the fund's official documentation (LPA, presentations, policies, memos) "
                "for information relevant to an investor's question. "
                "Use specific terms from the question for best results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — use specific terms related to the question"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 6, max 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_investor_documents",
            "description": (
                "Search investor-specific documents (entity structure, tax forms, etc.) "
                "uploaded for this investor. Use when the question relates to the investor's "
                "specific situation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for investor-specific documents"
                    },
                    "investor_session_id": {
                        "type": "integer",
                        "description": "The investor session ID"
                    }
                },
                "required": ["query", "investor_session_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current market context, sector analysis, industry benchmarks, "
                "macroeconomic data, or news relevant to the investor's question. "
                "Use when the question requires up-to-date information beyond the fund's documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to run on the web"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_url",
            "description": (
                "Open a URL in a headless browser and extract the full page text. "
                "Use after web_search to read the actual content of a page — financial reports, "
                "company pages, news articles, Wikipedia, etc. "
                "Essential for getting complete information beyond search snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to browse (must start with http:// or https://)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_documents",
            "description": (
                "List all documents available to the current investor. "
                "Use this when the investor asks what documents, materials, or information you have, "
                "what is available to them, or any similar question about document availability."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_questions",
            "description": (
                "Find previously answered similar questions for reference and consistency. "
                "Useful to ensure consistent responses across investors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The investor question to find similar past questions for"
                    }
                },
                "required": ["question"]
            }
        }
    }
]


def _execute_tool(tool_name: str, tool_input: dict, investor_session_id: int = None,
                  is_investor: bool = False) -> str:
    if tool_name == "browse_url":
        url = tool_input.get("url", "").strip()
        if not url.startswith("http"):
            return json.dumps({"error": "Invalid URL — must start with http:// or https://"})
        try:
            from playwright.sync_api import sync_playwright
            from bs4 import BeautifulSoup
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (compatible; DDQAgent/1.0)"})
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                html = page.content()
                browser.close()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Collapse blank lines
            lines = [l for l in text.splitlines() if l.strip()]
            text = "\n".join(lines)[:5000]
            return json.dumps({"url": url, "content": text})
        except Exception as e:
            return json.dumps({"error": f"Browse failed: {e}"})

    if tool_name == "web_search":
        query = tool_input.get("query", "")
        max_results = min(tool_input.get("max_results", 5), 10)
        try:
            from ddgs import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:400]
                    })
            if not results:
                return json.dumps({"results": [], "message": "No web results found."})
            return json.dumps({"results": results})
        except Exception as e:
            return json.dumps({"error": f"Web search failed: {e}"})

    if tool_name == "list_available_documents":
        if investor_session_id:
            from database import get_assigned_documents
            docs = get_assigned_documents(investor_session_id)
        else:
            from database import list_fund_documents
            docs = list_fund_documents()
        if not docs:
            return json.dumps({"documents": [], "message": "No documents are currently available."})
        return json.dumps({
            "documents": [
                {"name": d["name"], "type": d.get("doc_type", ""), "summary": (d.get("summary_snippet") or "")[:300]}
                for d in docs
            ]
        })

    if tool_name == "search_fund_documents":
        if is_investor and investor_session_id:
            results = search_assigned_fund_documents(
                tool_input["query"],
                investor_session_id,
                tool_input.get("limit", 6)
            )
        else:
            results = search_fund_documents(
                tool_input["query"],
                tool_input.get("limit", 6)
            )
        if not results:
            return json.dumps({"results": [], "message": "No relevant documents found."})
        return json.dumps({
            "results": [
                {
                    "doc_id": r["document_id"],
                    "doc_name": r["doc_name"],
                    "doc_type": r["doc_type"],
                    "section": r.get("section_ref") or r.get("page_ref") or "N/A",
                    "excerpt": r["chunk_text"][:600]
                }
                for r in results
            ]
        })

    elif tool_name == "search_investor_documents":
        sid = tool_input.get("investor_session_id") or investor_session_id
        if not sid:
            return json.dumps({"results": [], "message": "No investor session active."})
        results = search_investor_documents(tool_input["query"], sid)
        if not results:
            return json.dumps({"results": [], "message": "No investor documents found."})
        return json.dumps({
            "results": [
                {
                    "doc_name": r["doc_name"],
                    "doc_type": r["doc_type"],
                    "excerpt": r["chunk_text"][:600]
                }
                for r in results
            ]
        })

    elif tool_name == "find_similar_questions":
        results = find_similar_questions(tool_input["question"])
        if not results:
            return json.dumps({"results": [], "message": "No similar prior questions found."})
        return json.dumps({
            "results": [
                {
                    "question": r["question"][:300],
                    "answer": r["answer"][:400],
                    "date": r["created_at"]
                }
                for r in results
            ]
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _build_messages(question: str, conversation_history: list, custom_system_prompt: str = None) -> list:
    base = _get_system_prompt()
    if custom_system_prompt:
        base = custom_system_prompt + "\n\n" + base
    messages = [{"role": "system", "content": base}]
    for msg in conversation_history[-6:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"] if msg["role"] == "user" else (
                msg.get("answer") or msg["content"]
            )
        })
    messages.append({"role": "user", "content": question})
    return messages


def _check_knowledge_base(question: str) -> dict | None:
    """Check admin knowledge base for a trained answer. Returns result dict or None."""
    kb_hits = search_kb(question, limit=1)
    if not kb_hits:
        return None
    best = kb_hits[0]
    # The AND query in search_kb already ensures all significant keywords match.
    # Any result here is a genuine match — no score threshold needed.
    return {
        "answer": best["answer"],
        "sources": [{"doc_name": "Knowledge Base", "section": "Admin-trained answer", "excerpt": best["question"]}],
        "draft_response": f"Thank you for your question.\n\n{best['answer']}\n\nPlease let us know if you need any further clarification.",
        "themes": ["other"],
        "confidence": "high",
        "gaps": None,
        "_kb_match": True,
    }


def _check_session_answers(question: str, investor_session_id: int) -> dict | None:
    """Check investor-provided session answers first (highest priority)."""
    if not investor_session_id:
        return None
    hits = search_session_answers(question, investor_session_id, limit=1)
    if not hits:
        return None
    best = hits[0]
    return {
        "answer": best["answer"],
        "sources": [{"doc_name": "Your Provided Answers", "section": "Investor-verified answer", "excerpt": best["question"]}],
        "draft_response": f"Thank you for your question.\n\n{best['answer']}\n\nPlease let us know if you need any further clarification.",
        "themes": ["other"],
        "confidence": "high",
        "gaps": None,
        "_kb_match": True,
    }


def _answer_anthropic(client, model: str, key_id, messages: list,
                      investor_session_id: int, is_investor: bool) -> dict | None:
    """Run tool-use loop with Anthropic client. Returns parsed result dict or None on failure."""
    try:
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), _get_system_prompt())
        msgs = [m for m in messages if m["role"] != "system"]

        anthropic_tools = []
        for t in TOOLS:
            fn = t["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            })

        for _ in range(6):
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_msg,
                messages=msgs,
                tools=anthropic_tools,
                tool_choice={"type": "auto"},
            )
            if resp.stop_reason == "tool_use":
                msgs.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for blk in resp.content:
                    if blk.type == "tool_use":
                        result = _execute_tool(blk.name, blk.input, investor_session_id, is_investor)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": blk.id,
                            "content": result,
                        })
                msgs.append({"role": "user", "content": tool_results})
            else:
                text = "".join(blk.text for blk in resp.content if hasattr(blk, "text"))
                if key_id:
                    log_llm_usage(key_id, "anthropic", model,
                                  resp.usage.input_tokens, resp.usage.output_tokens)
                return _parse_response(text)
    except Exception:
        pass
    return None


def _build_investor_context(messages: list, investor_name: str, investor_session_id: int,
                            agent_memories: list = None):
    """Inject investor context + agent memory into the system prompt."""
    if investor_name:
        messages[0]["content"] += f"\n\nCURRENT INVESTOR: {investor_name}"
        if investor_session_id:
            messages[0]["content"] += f" (investor_session_id: {investor_session_id})"
            from database import get_assigned_documents
            assigned_docs = get_assigned_documents(investor_session_id)
            if assigned_docs:
                doc_names = ", ".join(d["name"] for d in assigned_docs)
                messages[0]["content"] += f"\nDocuments available to this investor: {doc_names}"
            messages[0]["content"] += "\nInvestor-specific documents are available — search them."
    if agent_memories:
        interests = [m["content"] for m in agent_memories if m["memory_type"] == "interest"]
        gaps = [m["content"] for m in agent_memories if m["memory_type"] == "gap"]
        if interests:
            messages[0]["content"] += f"\n\nAGENT MEMORY — topics this investor has asked about: {'; '.join(interests[-5:])}"
        if gaps:
            messages[0]["content"] += f"\nAGENT MEMORY — previous gaps/failures to note: {'; '.join(gaps[-3:])}"
        messages[0]["content"] += "\nUse this memory to give better, more personalised answers. If a past gap is relevant, try a different search approach."


def answer_question(
    question: str,
    conversation_history: list,
    investor_session_id: int = None,
    investor_name: str = None,
    is_investor: bool = False,
    custom_system_prompt: str = None,
) -> dict:
    """Run the DDQ agent. Returns parsed dict with answer, sources, draft_response, themes."""
    session_result = _check_session_answers(question, investor_session_id)
    if session_result:
        return session_result
    kb_result = _check_knowledge_base(question)
    if kb_result:
        return kb_result

    clients = _get_clients()
    if not clients:
        return _fallback()

    messages = _build_messages(question, conversation_history, custom_system_prompt)
    _build_investor_context(messages, investor_name, investor_session_id)

    for key_id, model, provider, client in clients:
        try:
            if provider == "anthropic":
                result = _answer_anthropic(client, model, key_id, messages,
                                           investor_session_id, is_investor)
                if result:
                    return result
                continue

            msgs = [m.copy() for m in messages]
            first_call = True
            retried = False
            for _ in range(8):
                # Investors can trigger list_available_documents, so let model pick
                # Internal staff: force search on first call
                if first_call and not is_investor:
                    tc_choice = {"type": "function", "function": {"name": "search_fund_documents"}}
                else:
                    tc_choice = "auto"
                response = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=TOOLS,
                    tool_choice=tc_choice,
                    response_format={"type": "json_object"},
                )
                first_call = False
                msg = response.choices[0].message

                if msg.tool_calls:
                    msgs.append(msg)
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        res = _execute_tool(tc.function.name, args, investor_session_id, is_investor)
                        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                else:
                    if key_id and response.usage:
                        log_llm_usage(key_id, provider, model,
                                      response.usage.prompt_tokens,
                                      response.usage.completion_tokens)
                    parsed = _parse_response(msg.content or "")
                    if parsed.get("confidence") == "low" and not retried:
                        retried = True
                        msgs.append(msg)
                        msgs.append({"role": "user", "content":
                            f"LOOP RETRY: Low confidence. Gaps: {parsed.get('gaps') or 'unclear'}. "
                            f"Try broader/different search keywords and search again."})
                        continue
                    return parsed
        except Exception:
            continue  # try next key

    return _fallback()


def stream_answer(
    question: str,
    conversation_history: list,
    investor_session_id: int = None,
    investor_name: str = None,
    is_investor: bool = False,
    agent_memories: list = None,
    custom_system_prompt: str = None,
):
    """
    Generator yielding SSE-formatted strings.
    Runs tool-use loop first (non-streaming), then yields the final result.
    Includes retry loop: if confidence is low, retries once with broader query.
    """
    # Check investor session answers first (highest priority), then admin knowledge base
    session_result = _check_session_answers(question, investor_session_id)
    if session_result:
        yield f"data: {json.dumps({'type': 'result', 'data': session_result})}\n\n"
        return
    kb_result = _check_knowledge_base(question)
    if kb_result:
        yield f"data: {json.dumps({'type': 'result', 'data': kb_result})}\n\n"
        return

    clients = _get_clients()
    if not clients:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No LLM keys configured.'})}\n\n"
        return

    messages = _build_messages(question, conversation_history, custom_system_prompt)
    _build_investor_context(messages, investor_name, investor_session_id, agent_memories)

    yield f"data: {json.dumps({'type': 'thinking', 'text': 'Analysing your question…'})}\n\n"

    for key_id, model, provider, client in clients:
        try:
            if provider == "anthropic":
                result = _answer_anthropic(client, model, key_id, messages,
                                           investor_session_id, is_investor)
                if result:
                    answer = result.get("answer", "")
                    if answer:
                        words = answer.split(" ")
                        for i, word in enumerate(words):
                            token = word + (" " if i < len(words) - 1 else "")
                            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                    yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
                    return
                continue

            msgs = [m.copy() for m in messages]
            first_call = True
            retried = False
            step = 0
            for _ in range(10):
                if first_call and not is_investor:
                    tc_choice = {"type": "function", "function": {"name": "search_fund_documents"}}
                else:
                    tc_choice = "auto"
                response = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=TOOLS,
                    tool_choice=tc_choice,
                    response_format={"type": "json_object"},
                )
                first_call = False
                msg = response.choices[0].message

                if msg.tool_calls:
                    msgs.append(msg)
                    for tc in msg.tool_calls:
                        step += 1
                        args = json.loads(tc.function.arguments)
                        tool = tc.function.name

                        # Emit rich thinking event per tool call
                        if tool == "web_search":
                            q = args.get("query", "")
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '🔍', 'text': f'Searching the web for: {q}'})}\n\n"
                        elif tool == "browse_url":
                            url = args.get("url", "")
                            domain = url.split("/")[2] if "//" in url else url[:40]
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '🌐', 'text': f'Reading page: {domain}', 'url': url})}\n\n"
                        elif tool == "search_fund_documents":
                            q = args.get("query", "")
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '📄', 'text': f'Searching fund documents for: {q}'})}\n\n"
                        elif tool == "search_investor_documents":
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '📁', 'text': 'Searching your uploaded documents…'})}\n\n"
                        elif tool == "list_available_documents":
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '📋', 'text': 'Listing available documents…'})}\n\n"
                        elif tool == "find_similar_questions":
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '🔗', 'text': 'Checking previously answered questions…'})}\n\n"

                        res = _execute_tool(tool, args, investor_session_id, is_investor)

                        # Emit what was found
                        try:
                            res_data = json.loads(res)
                            if tool == "web_search":
                                sites = [r.get("url", "") for r in res_data.get("results", [])[:5]]
                                domains = list({s.split("/")[2] for s in sites if "//" in s})[:4]
                                if domains:
                                    domains_str = ", ".join(domains)
                                    yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '✅', 'text': 'Found results from: ' + domains_str, 'sites': sites})}\n\n"
                            elif tool == "browse_url":
                                chars = len(res_data.get("content", ""))
                                yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '✅', 'text': f'Extracted {chars} characters of content'})}\n\n"
                            elif tool in ("search_fund_documents", "search_investor_documents"):
                                count = len(res_data.get("results", []))
                                yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '✅', 'text': f'Found {count} relevant passage(s) in documents'})}\n\n"
                        except Exception:
                            pass

                        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                else:
                    if key_id and response.usage:
                        log_llm_usage(key_id, provider, model,
                                      response.usage.prompt_tokens,
                                      response.usage.completion_tokens)
                    parsed = _parse_response(msg.content or "")

                    # ── Loop agent: retry once on low confidence ──────────────
                    if parsed.get("confidence") == "low" and not retried:
                        retried = True
                        gap_text = str(parsed.get("gaps") or "unclear")[:80]
                        yield f"data: {json.dumps({'type': 'thinking', 'icon': '🔄', 'text': 'Low confidence — retrying with broader search… (gap: ' + gap_text + ')'})}\n\n"
                        msgs.append(msg)
                        msgs.append({"role": "user", "content":
                            f"LOOP RETRY: Low confidence. Gaps: {parsed.get('gaps') or 'unclear'}. "
                            f"Try broader/different keywords. Search again before answering."})
                        continue

                    yield f"data: {json.dumps({'type': 'thinking', 'icon': '✍️', 'text': 'Composing answer…'})}\n\n"
                    answer = parsed.get("answer", "")
                    if answer:
                        words = answer.split(" ")
                        for i, word in enumerate(words):
                            token = word + (" " if i < len(words) - 1 else "")
                            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                    yield f"data: {json.dumps({'type': 'result', 'data': parsed})}\n\n"
                    return
        except Exception:
            continue  # try next key

    yield f"data: {json.dumps({'type': 'error', 'message': 'Max iterations reached'})}\n\n"


_json_decoder = json.JSONDecoder()


def _parse_response(text: str) -> dict:
    if not text:
        return _fallback()
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1] if text.count("```") >= 2 else text
        text = text.lstrip("json").strip().rstrip("`").strip()
    try:
        start = text.find("{")
        if start >= 0:
            # raw_decode only parses the FIRST complete JSON object,
            # ignoring any extra data after it (e.g. concatenated responses)
            data, _ = _json_decoder.raw_decode(text, start)
            return {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "draft_response": data.get("draft_response", ""),
                "themes": data.get("themes", []),
                "confidence": data.get("confidence", "medium"),
                "gaps": data.get("gaps"),
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return {
        "answer": text,
        "sources": [],
        "draft_response": text,
        "themes": [],
        "confidence": "low",
        "gaps": None,
    }


def generate_investor_profile(investor_name: str, entity: str, notes: str, questions: list) -> dict:
    """
    Generate a structured investor profile from their description and question history.
    Returns a dict with keys: investor_type, focus_areas, key_concerns, due_diligence_priorities,
    communication_style, summary.
    """
    clients = _get_clients()
    if not clients:
        return None

    q_text = "\n".join(f"- {q['content']}" for q in questions[:40]) if questions else "No questions asked yet."

    prompt = f"""You are an investor relations analyst. Based on the information below, generate a concise investor profile.

INVESTOR NAME: {investor_name}
ENTITY: {entity or 'Not specified'}
NOTES / DESCRIPTION: {notes or 'None provided'}

QUESTIONS ASKED BY THIS INVESTOR:
{q_text}

Return ONLY valid JSON with this exact structure:
{{
  "investor_type": "e.g. Institutional, Family Office, HNW Individual, Pension Fund, Endowment, etc.",
  "focus_areas": ["area1", "area2", "area3"],
  "key_concerns": ["concern1", "concern2", "concern3"],
  "due_diligence_priorities": ["priority1", "priority2", "priority3"],
  "communication_style": "Brief description of how they communicate (e.g. detail-oriented, high-level, technical)",
  "risk_profile": "Conservative / Moderate / Aggressive — with brief reasoning",
  "summary": "2-3 sentence professional summary of this investor's profile and what they care about most."
}}"""

    for key_id, model, provider, client in clients:
        try:
            if provider == "anthropic":
                resp = client.messages.create(
                    model=model, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                if key_id:
                    log_llm_usage(key_id, "anthropic", model,
                                  resp.usage.input_tokens, resp.usage.output_tokens)
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or ""
                if key_id and resp.usage:
                    log_llm_usage(key_id, provider, model,
                                  resp.usage.prompt_tokens, resp.usage.completion_tokens)

            start = text.find("{")
            if start >= 0:
                data, _ = _json_decoder.raw_decode(text, start)
                return data
        except Exception:
            continue
    return None


def _fallback() -> dict:
    return {
        "answer": "I was unable to generate a response. Please try again.",
        "sources": [],
        "draft_response": "",
        "themes": [],
        "confidence": "low",
        "gaps": None,
    }
