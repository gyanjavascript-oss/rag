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
)

FUND_NAME = os.getenv("FUND_NAME", "the Fund")
MODEL = "gpt-4o"

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
    {{"doc_name": "exact document name", "section": "section or page reference", "excerpt": "short verbatim quote from the document"}}
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
    # BM25 scores are negative; any match (score < 0) means FTS found a hit.
    # Admin-curated KB is small and intentionally trained, so accept all hits.
    if best["score"] >= 0:
        return None
    return {
        "answer": best["answer"],
        "sources": [{"doc_name": "Knowledge Base", "section": "Admin-trained answer", "excerpt": best["question"]}],
        "draft_response": f"Thank you for your question.\n\n{best['answer']}\n\nPlease let us know if you need any further clarification.",
        "themes": ["other"],
        "confidence": "high",
        "gaps": None,
        "_kb_match": True,
    }


def answer_question(
    question: str,
    conversation_history: list,
    investor_session_id: int = None,
    investor_name: str = None,
    is_investor: bool = False,
) -> dict:
    """Run the DDQ agent. Returns parsed dict with answer, sources, draft_response, themes."""
    # Check admin knowledge base first
    kb_result = _check_knowledge_base(question)
    if kb_result:
        return kb_result

    client = OpenAI()
    messages = _build_messages(question, conversation_history)

    if investor_name:
        messages[0]["content"] += f"\n\nCURRENT INVESTOR: {investor_name}"
        if investor_session_id:
            messages[0]["content"] += f" (investor_session_id: {investor_session_id})"
            messages[0]["content"] += "\nInvestor-specific documents are available — search them."

    # First call: force a document search
    first_call = True
    for _ in range(6):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "search_fund_documents"}} if first_call else "auto",
            response_format={"type": "json_object"},
        )
        first_call = False
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = _execute_tool(tc.function.name, args, investor_session_id, is_investor)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return _parse_response(msg.content or "")

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

    client = OpenAI()
    messages = _build_messages(question, conversation_history)

    if investor_name:
        messages[0]["content"] += f"\n\nCURRENT INVESTOR: {investor_name}"
        if investor_session_id:
            messages[0]["content"] += f" (investor_session_id: {investor_session_id})"
            messages[0]["content"] += "\nInvestor-specific documents are available — search them."

    first_call = True
    for _ in range(6):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "search_fund_documents"}} if first_call else "auto",
            response_format={"type": "json_object"},
        )
        first_call = False
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc.function.name})}\n\n"
                args = json.loads(tc.function.arguments)
                result = _execute_tool(tc.function.name, args, investor_session_id, is_investor)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            parsed = _parse_response(msg.content or "")
            answer = parsed.get("answer", "")
            # Stream answer word-by-word for typing effect
            if answer:
                words = answer.split(" ")
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
            # Send full result (sources, themes, meta) after tokens
            yield f"data: {json.dumps({'type': 'result', 'data': parsed})}\n\n"
            return

    yield f"data: {json.dumps({'type': 'error', 'message': 'Max iterations reached'})}\n\n"


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
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
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


def _fallback() -> dict:
    return {
        "answer": "I was unable to generate a response. Please try again.",
        "sources": [],
        "draft_response": "",
        "themes": [],
        "confidence": "low",
        "gaps": None,
    }
