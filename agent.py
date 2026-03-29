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

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str, **kwargs) -> str:
    """
    Load a prompt from prompts/<name>.md.
    - Strips YAML frontmatter (--- ... ---) automatically
    - Substitutes ${variable} and ${variable:default} placeholders
    - Falls back to frontmatter defaults for any variable not passed in kwargs
    """
    import re
    path = os.path.join(_PROMPTS_DIR, name + ".md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Parse YAML frontmatter to extract variable defaults
    defaults = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            # Extract variables block
            in_vars = False
            for line in front.splitlines():
                if line.strip() == "variables:":
                    in_vars = True
                    continue
                if in_vars:
                    if line and not line.startswith(" "):
                        in_vars = False
                        continue
                    m = re.match(r'\s+(\w+):\s*"?(.*?)"?\s*$', line)
                    if m:
                        defaults[m.group(1)] = m.group(2)
            # Strip frontmatter from body
            text = text[end + 4:].lstrip("\n")

    # Merge defaults with provided kwargs (kwargs takes priority)
    values = {**defaults, **{k: str(v) for k, v in kwargs.items()}}

    # Replace ${var:default} — inline default overrides everything if key not in values
    def _replace(m):
        key = m.group(1)
        inline_default = m.group(2) if m.group(2) is not None else ""
        return values.get(key, inline_default)

    text = re.sub(r'\$\{(\w+)(?::([^}]*))?\}', _replace, text)
    return text


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

def _get_system_prompt(is_investor: bool = False) -> str:
    from datetime import date
    today = date.today()
    name = "investor_agent" if is_investor else "ddq_agent"
    return _load_prompt(name,
        fund_name=FUND_NAME,
        today=today.strftime("%B %d, %Y"),
        year=str(today.year),
    )


# Keep a module-level alias for any code that references SYSTEM_PROMPT directly
SYSTEM_PROMPT = _get_system_prompt()  # loaded from prompts/ddq_agent.md


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

# Investor-only tools: no web_search or browse_url
INVESTOR_TOOLS = [t for t in TOOLS if t["function"]["name"] not in ("web_search", "browse_url")]


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


def _build_messages(question: str, conversation_history: list,
                    custom_system_prompt: str = None, is_investor: bool = False) -> list:
    base = _get_system_prompt(is_investor=is_investor)
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


def _check_knowledge_base(question: str) -> dict:
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


def _check_session_answers(question: str, investor_session_id: int) -> dict:
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
                      investor_session_id: int, is_investor: bool,
                      allowed_tools: list = None) -> dict:
    """Run tool-use loop with Anthropic client. Returns parsed result dict or None on failure."""
    try:
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), _get_system_prompt())
        msgs = [m for m in messages if m["role"] != "system"]

        # Respect investor tool restrictions and allowed_tools filter
        base = INVESTOR_TOOLS if is_investor else TOOLS
        if allowed_tools:
            base = [t for t in base if t["function"]["name"] in allowed_tools] or base

        anthropic_tools = []
        for t in base:
            fn = t["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            })

        doc_searches_done = 0
        retry_count = 0
        MIN_DOC_SEARCHES = 3 if is_investor else 1
        MAX_RETRIES = 3 if is_investor else 1

        for _ in range(14):
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
                        if blk.name == "search_fund_documents":
                            doc_searches_done += 1
                        result = _execute_tool(blk.name, blk.input, investor_session_id, is_investor)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": blk.id,
                            "content": result,
                        })
                msgs.append({"role": "user", "content": tool_results})
            else:
                text = "".join(blk.text for blk in resp.content if hasattr(blk, "text"))
                # Enforce minimum search rounds for investors
                if is_investor and doc_searches_done < MIN_DOC_SEARCHES and retry_count < MAX_RETRIES:
                    retry_count += 1
                    msgs.append({"role": "assistant", "content": resp.content})
                    msgs.append({"role": "user", "content": [{
                        "type": "text",
                        "text": (
                            f"SEARCH AGENT: You have only called search_fund_documents {doc_searches_done} time(s). "
                            f"You MUST search at least {MIN_DOC_SEARCHES} times with DIFFERENT query angles. "
                            f"Search round {doc_searches_done + 1}: use synonyms, related section names, or broader concepts. "
                            f"Do NOT answer yet — search more first."
                        )
                    }]})
                    continue
                parsed = _parse_response(text)
                if parsed.get("confidence") == "low" and retry_count < MAX_RETRIES:
                    retry_count += 1
                    msgs.append({"role": "assistant", "content": resp.content})
                    msgs.append({"role": "user", "content": [{
                        "type": "text",
                        "text": (
                            f"SEARCH RETRY {retry_count}: Confidence is low. Gaps: {parsed.get('gaps') or 'unclear'}. "
                            f"Search with completely different terminology — synonyms, section names, related concepts. "
                            f"Search again before answering."
                        )
                    }]})
                    continue
                if key_id:
                    log_llm_usage(key_id, "anthropic", model,
                                  resp.usage.input_tokens, resp.usage.output_tokens)
                return parsed
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
    allowed_tools: list = None,
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

    messages = _build_messages(question, conversation_history, custom_system_prompt, is_investor)
    _build_investor_context(messages, investor_name, investor_session_id)

    base_tools = INVESTOR_TOOLS if is_investor else TOOLS
    if allowed_tools:
        active_tools = [t for t in base_tools if t["function"]["name"] in allowed_tools]
        if not active_tools:
            active_tools = base_tools
    else:
        active_tools = base_tools

    for key_id, model, provider, client in clients:
        try:
            if provider == "anthropic":
                result = _answer_anthropic(client, model, key_id, messages,
                                           investor_session_id, is_investor, allowed_tools)
                if result:
                    return result
                continue

            msgs = [m.copy() for m in messages]
            first_call = True
            retry_count = 0
            doc_searches_done = 0
            MIN_DOC_SEARCHES = 3 if is_investor else 1
            MAX_RETRIES = 3 if is_investor else 1
            for _ in range(14):
                if first_call:
                    tc_choice = {"type": "function", "function": {"name": "search_fund_documents"}}
                else:
                    tc_choice = "auto"
                response = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=active_tools,
                    tool_choice=tc_choice,
                    response_format={"type": "json_object"},
                )
                first_call = False
                msg = response.choices[0].message

                if msg.tool_calls:
                    msgs.append(msg)
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        if tc.function.name == "search_fund_documents":
                            doc_searches_done += 1
                        res = _execute_tool(tc.function.name, args, investor_session_id, is_investor)
                        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                else:
                    # Enforce minimum search rounds for investors
                    if is_investor and doc_searches_done < MIN_DOC_SEARCHES and retry_count < MAX_RETRIES:
                        retry_count += 1
                        msgs.append(msg)
                        msgs.append({"role": "user", "content":
                            f"SEARCH AGENT: You have only called search_fund_documents {doc_searches_done} time(s). "
                            f"You MUST search at least {MIN_DOC_SEARCHES} times with DIFFERENT query angles before answering. "
                            f"Search round {doc_searches_done + 1}: use synonyms, related section names, or broader concepts. "
                            f"Do NOT answer yet — search more first."})
                        continue

                    if key_id and response.usage:
                        log_llm_usage(key_id, provider, model,
                                      response.usage.prompt_tokens,
                                      response.usage.completion_tokens)
                    parsed = _parse_response(msg.content or "")

                    if parsed.get("confidence") == "low" and retry_count < MAX_RETRIES:
                        retry_count += 1
                        msgs.append(msg)
                        msgs.append({"role": "user", "content":
                            f"SEARCH RETRY {retry_count}: Low confidence. Gaps: {parsed.get('gaps') or 'unclear'}. "
                            f"Search with completely different terminology — synonyms, section names, related concepts. "
                            f"Search again before answering."})
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
    allowed_tools: list = None,
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

    messages = _build_messages(question, conversation_history, custom_system_prompt, is_investor)
    _build_investor_context(messages, investor_name, investor_session_id, agent_memories)

    base_tools = INVESTOR_TOOLS if is_investor else TOOLS
    if allowed_tools:
        active_tools = [t for t in base_tools if t["function"]["name"] in allowed_tools]
        if not active_tools:
            active_tools = base_tools
    else:
        active_tools = base_tools

    yield f"data: {json.dumps({'type': 'thinking', 'text': 'Analysing your question…'})}\n\n"

    for key_id, model, provider, client in clients:
        try:
            if provider == "anthropic":
                result = _answer_anthropic(client, model, key_id, messages,
                                           investor_session_id, is_investor, allowed_tools)
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
            retry_count = 0
            doc_searches_done = 0  # track how many search_fund_documents calls made
            doc_registry = {}      # doc_name (lower) → doc_id, built from search results
            # Investor loop agent: require at least MIN_DOC_SEARCHES before accepting answer
            MIN_DOC_SEARCHES = 3 if is_investor else 1
            MAX_RETRIES = 3 if is_investor else 1
            step = 0
            for _ in range(14):
                # Force first call to always search fund documents
                if first_call:
                    tc_choice = {"type": "function", "function": {"name": "search_fund_documents"}}
                else:
                    tc_choice = "auto"
                response = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=active_tools,
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

                        if tool == "search_fund_documents":
                            doc_searches_done += 1
                            q = args.get("query", "")
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '📄', 'text': f'Searching fund documents (round {doc_searches_done}): {q}'})}\n\n"
                        elif tool == "web_search":
                            q = args.get("query", "")
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '🔍', 'text': f'Searching the web for: {q}'})}\n\n"
                        elif tool == "browse_url":
                            url = args.get("url", "")
                            domain = url.split("/")[2] if "//" in url else url[:40]
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '🌐', 'text': f'Reading page: {domain}', 'url': url})}\n\n"
                        elif tool == "search_investor_documents":
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '📁', 'text': 'Searching your uploaded documents…'})}\n\n"
                        elif tool == "list_available_documents":
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '📋', 'text': 'Listing available documents…'})}\n\n"
                        elif tool == "find_similar_questions":
                            yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '🔗', 'text': 'Checking previously answered questions…'})}\n\n"

                        res = _execute_tool(tool, args, investor_session_id, is_investor)

                        try:
                            res_data = json.loads(res)
                            if tool == "web_search":
                                sites = [r.get("url", "") for r in res_data.get("results", [])[:5]]
                                domains = list({s.split("/")[2] for s in sites if "//" in s})[:4]
                                if domains:
                                    yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '✅', 'text': 'Found results from: ' + ', '.join(domains), 'sites': sites})}\n\n"
                            elif tool == "browse_url":
                                chars = len(res_data.get("content", ""))
                                yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '✅', 'text': f'Extracted {chars} characters of content'})}\n\n"
                            elif tool in ("search_fund_documents", "search_investor_documents"):
                                count = len(res_data.get("results", []))
                                # Build doc_name → doc_id registry from search results
                                for r in res_data.get("results", []):
                                    name = (r.get("doc_name") or "").strip()
                                    did = r.get("doc_id") or 0
                                    if name and did:
                                        doc_registry[name.lower()] = did
                                yield f"data: {json.dumps({'type': 'thinking', 'step': step, 'icon': '✅', 'text': f'Found {count} relevant passage(s)'})}\n\n"
                        except Exception:
                            pass

                        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                else:
                    # LLM wants to produce final answer — check if loop agent requires more searches
                    if is_investor and doc_searches_done < MIN_DOC_SEARCHES and retry_count < MAX_RETRIES:
                        retry_count += 1
                        remaining = MIN_DOC_SEARCHES - doc_searches_done
                        yield f"data: {json.dumps({'type': 'thinking', 'icon': '🔄', 'text': f'Search agent: searched {doc_searches_done} time(s) — trying {remaining} more angle(s) for better coverage…'})}\n\n"
                        msgs.append(msg)
                        msgs.append({"role": "user", "content":
                            f"SEARCH AGENT: You have only called search_fund_documents {doc_searches_done} time(s). "
                            f"You MUST search at least {MIN_DOC_SEARCHES} times with DIFFERENT query angles before answering. "
                            f"Search round {doc_searches_done + 1}: use synonyms, related section names, or broader concepts "
                            f"related to the question. Do NOT answer yet — search more first."})
                        continue

                    if key_id and response.usage:
                        log_llm_usage(key_id, provider, model,
                                      response.usage.prompt_tokens,
                                      response.usage.completion_tokens)
                    parsed = _parse_response(msg.content or "")

                    # ── Search agent: retry on low confidence (up to MAX_RETRIES) ──
                    if parsed.get("confidence") == "low" and retry_count < MAX_RETRIES:
                        retry_count += 1
                        gap_text = str(parsed.get("gaps") or "unclear")[:100]
                        yield f"data: {json.dumps({'type': 'thinking', 'icon': '🔄', 'text': f'Low confidence (retry {retry_count}/{MAX_RETRIES}) — searching with different terms… Gap: {gap_text}'})}\n\n"
                        msgs.append(msg)
                        msgs.append({"role": "user", "content":
                            f"SEARCH RETRY {retry_count}: Confidence is low. Gaps: {parsed.get('gaps') or 'unclear'}. "
                            f"Search fund documents again using completely different terminology — "
                            f"try section names, synonyms, or related concepts you haven't tried yet. "
                            f"Search before answering."})
                        continue

                    # Enrich sources with correct doc_id from registry
                    for src in parsed.get("sources", []):
                        if not src.get("doc_id"):
                            name_key = (src.get("doc_name") or "").strip().lower()
                            if name_key in doc_registry:
                                src["doc_id"] = doc_registry[name_key]
                            else:
                                # Fuzzy: find any registry key that starts with / contains the name
                                for reg_name, reg_id in doc_registry.items():
                                    if name_key and (name_key in reg_name or reg_name in name_key):
                                        src["doc_id"] = reg_id
                                        break

                    yield f"data: {json.dumps({'type': 'thinking', 'icon': '✍️', 'text': f'Composing answer from {doc_searches_done} search round(s)…'})}\n\n"
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

    prompt = _load_prompt("investor_profile",
        investor_name=investor_name,
        entity=entity or "Not specified",
        notes=notes or "None provided",
        questions=q_text,
    )

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
