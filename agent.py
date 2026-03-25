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

SYSTEM_PROMPT = f"""You are an expert DDQ (Due Diligence Questionnaire) assistant for {FUND_NAME}.

Your role is to help the investor relations team respond to questions using the fund's official documentation. You MUST always search the documents before answering — never answer from general knowledge.

MANDATORY WORKFLOW — follow this every time:
1. ALWAYS call search_fund_documents first with relevant keywords from the question
2. If the question involves the investor's own documents, also call search_investor_documents
3. Optionally call find_similar_questions for consistency with prior answers
4. After retrieving results, compose your response

RESPONSE FORMAT — you MUST return ONLY valid JSON, no other text:
{{
  "answer": "Detailed factual answer with inline citations [Doc Name, Section]",
  "sources": [
    {{"doc_name": "exact document name", "doc_id": 123, "section": "section or page reference", "excerpt": "short verbatim quote from the document"}}
  ],
  "draft_response": "Professional email-ready response the team can copy and send. Start with 'Thank you for your question...'",
  "themes": ["theme1", "theme2"],
  "confidence": "high|medium|low",
  "gaps": "Describe any gaps if the documents don't fully answer the question, otherwise null"
}}

RULES:
- You MUST call search_fund_documents before composing any answer
- Base your answer ONLY on what was found in the retrieved documents
- If nothing relevant is found, say so clearly in the answer field — do not invent information
- Always populate the sources array with what was found
- THEMES must be from: fee structure, performance, portfolio, governance, legal structure,
  investment strategy, ESG, liquidity, reporting, LP rights, key person, risk management,
  compliance, tax, track record, team, fund terms, co-investment, distributions, valuation, other"""


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


def _build_messages(question: str, conversation_history: list) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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


def _answer_anthropic(client, model: str, key_id, messages: list,
                      investor_session_id: int, is_investor: bool) -> dict | None:
    """Run tool-use loop with Anthropic client. Returns parsed result dict or None on failure."""
    try:
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
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


def answer_question(
    question: str,
    conversation_history: list,
    investor_session_id: int = None,
    investor_name: str = None,
    is_investor: bool = False,
) -> dict:
    """Run the DDQ agent. Returns parsed dict with answer, sources, draft_response, themes."""
    kb_result = _check_knowledge_base(question)
    if kb_result:
        return kb_result

    clients = _get_clients()
    if not clients:
        return _fallback()

    messages = _build_messages(question, conversation_history)
    if investor_name:
        messages[0]["content"] += f"\n\nCURRENT INVESTOR: {investor_name}"
        if investor_session_id:
            messages[0]["content"] += f" (investor_session_id: {investor_session_id})"
            messages[0]["content"] += "\nInvestor-specific documents are available — search them."

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
            for _ in range(6):
                response = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=TOOLS,
                    tool_choice={"type": "function", "function": {"name": "search_fund_documents"}} if first_call else "auto",
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
                    return _parse_response(msg.content or "")
        except Exception:
            continue  # try next key

    return _fallback()


def stream_answer(
    question: str,
    conversation_history: list,
    investor_session_id: int = None,
    investor_name: str = None,
    is_investor: bool = False,
):
    """
    Generator yielding SSE-formatted strings.
    Runs tool-use loop first (non-streaming), then yields the final result.
    """
    # Check admin knowledge base first
    kb_result = _check_knowledge_base(question)
    if kb_result:
        yield f"data: {json.dumps({'type': 'result', 'data': kb_result})}\n\n"
        return

    clients = _get_clients()
    if not clients:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No LLM keys configured.'})}\n\n"
        return

    messages = _build_messages(question, conversation_history)
    if investor_name:
        messages[0]["content"] += f"\n\nCURRENT INVESTOR: {investor_name}"
        if investor_session_id:
            messages[0]["content"] += f" (investor_session_id: {investor_session_id})"
            messages[0]["content"] += "\nInvestor-specific documents are available — search them."

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
            for _ in range(6):
                response = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=TOOLS,
                    tool_choice={"type": "function", "function": {"name": "search_fund_documents"}} if first_call else "auto",
                    response_format={"type": "json_object"},
                )
                first_call = False
                msg = response.choices[0].message

                if msg.tool_calls:
                    msgs.append(msg)
                    for tc in msg.tool_calls:
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc.function.name})}\n\n"
                        args = json.loads(tc.function.arguments)
                        res = _execute_tool(tc.function.name, args, investor_session_id, is_investor)
                        msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                else:
                    if key_id and response.usage:
                        log_llm_usage(key_id, provider, model,
                                      response.usage.prompt_tokens,
                                      response.usage.completion_tokens)
                    parsed = _parse_response(msg.content or "")
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
                import anthropic as _anthropic
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
