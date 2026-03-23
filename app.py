"""
DDQ Platform - Main Flask Application
"""
import os
import json
import uuid
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, Response, stream_with_context, send_file
)
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

import database as db
import document_processor as dp
from agent import stream_answer

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
FUND_DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FUND_DOCS_FOLDER, exist_ok=True)

db.init_db()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def investor_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "investor_user_id" not in session:
            return redirect(url_for("investor_login"))
        return f(*args, **kwargs)
    return decorated


def _current_investor() -> dict:
    return {
        "id": session.get("investor_user_id"),
        "name": session.get("investor_name"),
        "session_id": session.get("investor_session_id"),
    }


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def _current_user() -> dict:
    return {
        "id": session.get("user_id"),
        "name": session.get("user_name"),
        "email": session.get("user_email"),
        "role": session.get("user_role"),
    }


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("investor_login"))


@app.route("/admin")
def admin_index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = db.verify_login(email, password)
        if user:
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]
            session["user_role"] = user["role"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    stats = db.get_dashboard_stats()
    themes = db.get_theme_analytics()
    recent_questions = db.get_recent_questions(10)
    recent_conversations = db.list_conversations(limit=8)
    doc_stats = db.get_document_citation_stats()
    return render_template(
        "dashboard.html",
        user=_current_user(),
        stats=stats,
        themes=themes,
        recent_questions=recent_questions,
        recent_conversations=recent_conversations,
        doc_stats=doc_stats,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


# ── Q&A / Chat ────────────────────────────────────────────────────────────────

@app.route("/chat")
@login_required
def chat_list():
    conversations = db.list_conversations()
    investor_sessions = db.list_investor_sessions()
    return render_template(
        "chat.html",
        user=_current_user(),
        conversations=conversations,
        investor_sessions=investor_sessions,
        active_conv=None,
        messages=[],
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/chat/new", methods=["POST"])
@login_required
def new_conversation():
    investor_session_id = request.form.get("investor_session_id") or None
    if investor_session_id:
        investor_session_id = int(investor_session_id)
    title = request.form.get("title") or None
    user = _current_user()
    conv_id = db.create_conversation(
        created_by=user["id"],
        investor_session_id=investor_session_id,
        title=title,
    )
    return redirect(url_for("chat_view", conv_id=conv_id))


@app.route("/chat/<int:conv_id>")
@login_required
def chat_view(conv_id):
    conv = db.get_conversation(conv_id)
    if not conv:
        flash("Conversation not found.", "error")
        return redirect(url_for("chat_list"))
    messages = db.get_messages(conv_id)
    conversations = db.list_conversations()
    investor_sessions = db.list_investor_sessions()
    return render_template(
        "chat.html",
        user=_current_user(),
        conversations=conversations,
        investor_sessions=investor_sessions,
        active_conv=conv,
        messages=messages,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/chat/<int:conv_id>/stream")
@login_required
def chat_stream(conv_id):
    question = request.args.get("q", "").strip()
    if not question:
        return Response("data: {\"type\":\"error\",\"message\":\"Empty question\"}\n\n",
                        mimetype="text/event-stream")

    conv = db.get_conversation(conv_id)
    if not conv:
        return Response("data: {\"type\":\"error\",\"message\":\"Not found\"}\n\n",
                        mimetype="text/event-stream")

    # Save user message
    db.add_message(conv_id, "user", question)

    # Auto-title first message
    history = db.get_messages(conv_id)
    if len([m for m in history if m["role"] == "user"]) == 1 and not conv["title"]:
        db.update_conversation_title(conv_id, question[:80])

    # Build history for agent (exclude current question)
    agent_history = [
        m for m in history
        if not (m["role"] == "user" and m["content"] == question)
    ]

    investor_session_id = conv.get("investor_session_id")
    investor_name = conv.get("investor_name")

    def generate():
        final_result = None

        for chunk in stream_answer(
            question=question,
            conversation_history=agent_history,
            investor_session_id=investor_session_id,
            investor_name=investor_name,
        ):
            yield chunk
            # Capture the result event
            if '"type": "result"' in chunk or '"type":"result"' in chunk:
                try:
                    data = json.loads(chunk.replace("data: ", "").strip())
                    if data.get("type") == "result":
                        final_result = data.get("data", {})
                except Exception:
                    pass

        # Persist assistant message
        if final_result:
            db.add_message(
                conv_id,
                "assistant",
                final_result.get("answer", ""),
                sources=final_result.get("sources", []),
                draft_response=final_result.get("draft_response", ""),
                themes=final_result.get("themes", []),
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ── Documents ─────────────────────────────────────────────────────────────────

@app.route("/documents")
@login_required
def documents():
    docs = db.list_fund_documents()
    return render_template(
        "documents.html",
        user=_current_user(),
        documents=docs,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/documents/upload", methods=["POST"])
@login_required
def upload_document():
    if "file" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("documents"))

    file = request.files["file"]
    doc_name = request.form.get("name", "").strip() or file.filename
    doc_type = request.form.get("doc_type", "Other")

    if not file.filename or not _allowed(file.filename):
        flash("Unsupported file type. Allowed: PDF, DOCX, TXT, MD", "error")
        return redirect(url_for("documents"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    try:
        user = _current_user()
        doc_id = dp.ingest_fund_document(filepath, doc_name, doc_type, user["id"])
        flash(f'Document "{doc_name}" uploaded and indexed successfully.', "success")
    except Exception as e:
        flash(f"Error processing document: {e}", "error")

    return redirect(url_for("documents"))


@app.route("/documents/<int:doc_id>/delete", methods=["POST"])
@login_required
def delete_document(doc_id):
    db.delete_fund_document(doc_id)
    flash("Document removed.", "success")
    return redirect(url_for("documents"))


@app.route("/documents/<int:doc_id>/view")
@login_required
def view_document(doc_id):
    """Serve a fund document inline for in-browser viewing (admin/analyst)."""
    conn = db.get_db()
    doc = conn.execute("SELECT * FROM fund_documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    if not doc:
        flash("Document not found.", "error")
        return redirect(url_for("documents"))
    filepath = doc["filepath"]
    if not filepath or not os.path.exists(os.path.join(UPLOAD_FOLDER, os.path.basename(filepath))):
        flash("File not available.", "error")
        return redirect(url_for("documents"))
    full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(filepath))
    return send_file(full_path, as_attachment=False)


@app.route("/investor/documents/<int:doc_id>/view")
@investor_login_required
def investor_view_document(doc_id):
    """Serve an assigned fund document inline for investor in-browser viewing."""
    inv = _current_investor()
    assigned_ids = db.get_assigned_document_ids(inv["session_id"])
    if doc_id not in assigned_ids:
        return "Access denied", 403
    conn = db.get_db()
    doc = conn.execute("SELECT * FROM fund_documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    if not doc:
        return "Not found", 404
    filepath = doc["filepath"]
    if not filepath or not os.path.exists(os.path.join(UPLOAD_FOLDER, os.path.basename(filepath))):
        return "File not available", 404
    full_path = os.path.join(UPLOAD_FOLDER, os.path.basename(filepath))
    return send_file(full_path, as_attachment=False)


# ── Investor Sessions ─────────────────────────────────────────────────────────

@app.route("/investors")
@login_required
def investors():
    sessions = db.list_investor_sessions()
    return render_template(
        "investors.html",
        user=_current_user(),
        investor_sessions=sessions,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/investors/new", methods=["POST"])
@login_required
def new_investor_session():
    investor_name = request.form.get("investor_name", "").strip()
    investor_entity = request.form.get("investor_entity", "").strip()
    notes = request.form.get("notes", "").strip()
    if not investor_name:
        flash("Investor name is required.", "error")
        return redirect(url_for("investors"))
    user = _current_user()
    sid = db.create_investor_session(investor_name, investor_entity, notes, user["id"])
    flash(f'Investor session for "{investor_name}" created.', "success")
    return redirect(url_for("investor_detail", session_id=sid))


@app.route("/investors/<int:session_id>")
@login_required
def investor_detail(session_id):
    inv_session = db.get_investor_session(session_id)
    if not inv_session:
        flash("Investor session not found.", "error")
        return redirect(url_for("investors"))

    conn = db.get_db()
    inv_docs = conn.execute(
        "SELECT * FROM investor_documents WHERE investor_session_id=? ORDER BY uploaded_at DESC",
        (session_id,)
    ).fetchall()
    conv_list = conn.execute(
        "SELECT * FROM conversations WHERE investor_session_id=? ORDER BY updated_at DESC",
        (session_id,)
    ).fetchall()
    conn.close()

    all_fund_docs = db.list_fund_documents()
    assigned_doc_ids = db.get_assigned_document_ids(session_id)
    investor_user = db.get_investor_user(session_id)

    return render_template(
        "investor_detail.html",
        user=_current_user(),
        inv_session=inv_session,
        inv_docs=[dict(r) for r in inv_docs],
        conversations=[dict(r) for r in conv_list],
        all_fund_docs=all_fund_docs,
        assigned_doc_ids=assigned_doc_ids,
        investor_user=investor_user,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/investors/<int:session_id>/upload", methods=["POST"])
@login_required
def upload_investor_doc(session_id):
    if "file" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("investor_detail", session_id=session_id))

    file = request.files["file"]
    doc_name = request.form.get("name", "").strip() or file.filename
    doc_type = request.form.get("doc_type", "Other")

    if not file.filename or not _allowed(file.filename):
        flash("Unsupported file type.", "error")
        return redirect(url_for("investor_detail", session_id=session_id))

    filename = secure_filename(file.filename)
    unique_name = f"inv_{session_id}_{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    try:
        dp.ingest_investor_document(filepath, doc_name, doc_type, session_id)
        flash(f'Document "{doc_name}" uploaded for this investor.', "success")
    except Exception as e:
        flash(f"Error processing document: {e}", "error")

    return redirect(url_for("investor_detail", session_id=session_id))


@app.route("/investors/<int:session_id>/start-chat", methods=["POST"])
@login_required
def start_investor_chat(session_id):
    user = _current_user()
    inv = db.get_investor_session(session_id)
    title = f"DDQ – {inv['investor_name']}" if inv else "DDQ Session"
    conv_id = db.create_conversation(
        created_by=user["id"],
        investor_session_id=session_id,
        title=title,
    )
    return redirect(url_for("chat_view", conv_id=conv_id))


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.route("/analytics")
@login_required
def analytics():
    themes = db.get_theme_analytics()
    recent_questions = db.get_recent_questions(50)
    doc_stats = db.get_document_citation_stats()
    stats = db.get_dashboard_stats()
    return render_template(
        "analytics.html",
        user=_current_user(),
        themes=themes,
        recent_questions=recent_questions,
        doc_stats=doc_stats,
        stats=stats,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


# ── Team Management (admin only) ──────────────────────────────────────────────

@app.route("/team")
@login_required
def team():
    if _current_user()["role"] != "admin":
        flash("Admin access required.", "error")
        return redirect(url_for("dashboard"))
    conn = db.get_db()
    users = conn.execute("SELECT id, email, name, role, created_at FROM users ORDER BY created_at").fetchall()
    conn.close()
    return render_template(
        "team.html",
        user=_current_user(),
        users=[dict(u) for u in users],
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/team/add", methods=["POST"])
@login_required
def add_team_member():
    if _current_user()["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    email = request.form.get("email", "").strip().lower()
    name = request.form.get("name", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "analyst")
    if not all([email, name, password]):
        flash("All fields required.", "error")
        return redirect(url_for("team"))
    try:
        db.create_user(email, name, password, role)
        flash(f"User {name} added.", "success")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect(url_for("team"))


# ── API helpers ───────────────────────────────────────────────────────────────

@app.route("/api/themes")
@login_required
def api_themes():
    return jsonify(db.get_theme_analytics())


@app.route("/api/ingest-folder", methods=["POST"])
@login_required
def api_ingest_folder():
    """Bulk ingest the /documents folder (admin utility)."""
    if _current_user()["role"] != "admin":
        return jsonify({"error": "Admin only"}), 403
    results = dp.ingest_from_folder(FUND_DOCS_FOLDER, _current_user()["id"])
    return jsonify({"results": results})


# ── Admin: Investor Portal Management ─────────────────────────────────────────

@app.route("/investors/<int:session_id>/assign-docs", methods=["POST"])
@login_required
def assign_investor_docs(session_id):
    if _current_user()["role"] not in ("admin", "analyst"):
        flash("Access denied.", "error")
        return redirect(url_for("investor_detail", session_id=session_id))
    doc_ids = [int(x) for x in request.form.getlist("doc_ids")]
    user = _current_user()
    db.assign_documents_to_investor(session_id, doc_ids, user["id"])
    flash("Document access updated.", "success")
    return redirect(url_for("investor_detail", session_id=session_id))


@app.route("/investors/<int:session_id>/create-credentials", methods=["POST"])
@login_required
def create_investor_credentials(session_id):
    if _current_user()["role"] != "admin":
        flash("Admin access required.", "error")
        return redirect(url_for("investor_detail", session_id=session_id))
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    inv = db.get_investor_session(session_id)
    if not email or not password:
        flash("Email and password are required.", "error")
        return redirect(url_for("investor_detail", session_id=session_id))
    try:
        db.create_investor_user(email, inv["investor_name"] if inv else "Investor", password, session_id)
        flash(f"Investor login created: {email}", "success")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect(url_for("investor_detail", session_id=session_id))


# ── Investor Portal ────────────────────────────────────────────────────────────

@app.route("/investor/login", methods=["GET", "POST"])
def investor_login():
    if "investor_user_id" in session:
        return redirect(url_for("investor_portal"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = db.verify_login(email, password)
        if user and user.get("role") == "investor" and user.get("investor_session_id"):
            session["investor_user_id"] = user["id"]
            session["investor_name"] = user["name"]
            session["investor_session_id"] = user["investor_session_id"]
            return redirect(url_for("investor_portal"))
        flash("Invalid credentials.", "error")
    return render_template("investor_login.html",
                           fund_name=os.getenv("FUND_NAME", "DDQ Platform"))


@app.route("/investor/logout")
def investor_logout():
    session.pop("investor_user_id", None)
    session.pop("investor_name", None)
    session.pop("investor_session_id", None)
    return redirect(url_for("investor_login"))


@app.route("/investor/portal")
@investor_login_required
def investor_portal():
    inv = _current_investor()
    investor_session = db.get_investor_session(inv["session_id"])
    conversations = db.list_investor_conversations(inv["session_id"])
    assigned_docs = db.get_assigned_documents(inv["session_id"])
    return render_template(
        "investor_portal.html",
        investor=inv,
        investor_session=investor_session,
        conversations=conversations,
        assigned_docs=assigned_docs,
        active_conv=None,
        messages=[],
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/investor/chat/new", methods=["POST"])
@investor_login_required
def investor_new_chat():
    inv = _current_investor()
    investor_session = db.get_investor_session(inv["session_id"])
    title = f"DDQ – {investor_session['investor_name']}" if investor_session else "My Questions"
    # Use admin user id as created_by (or store investor user id)
    conv_id = db.create_conversation(
        created_by=inv["id"],
        investor_session_id=inv["session_id"],
        title=title,
    )
    return redirect(url_for("investor_chat_view", conv_id=conv_id))


@app.route("/investor/chat/<int:conv_id>")
@investor_login_required
def investor_chat_view(conv_id):
    inv = _current_investor()
    conv = db.get_conversation(conv_id)
    # Ensure this conversation belongs to this investor session
    if not conv or conv.get("investor_session_id") != inv["session_id"]:
        flash("Conversation not found.", "error")
        return redirect(url_for("investor_portal"))
    messages = db.get_messages(conv_id)
    investor_session = db.get_investor_session(inv["session_id"])
    conversations = db.list_investor_conversations(inv["session_id"])
    assigned_docs = db.get_assigned_documents(inv["session_id"])
    return render_template(
        "investor_portal.html",
        investor=inv,
        investor_session=investor_session,
        conversations=conversations,
        assigned_docs=assigned_docs,
        active_conv=conv,
        messages=messages,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/investor/chat/<int:conv_id>/stream")
@investor_login_required
def investor_chat_stream(conv_id):
    question = request.args.get("q", "").strip()
    inv = _current_investor()

    if not question:
        return Response('data: {"type":"error","message":"Empty question"}\n\n',
                        mimetype="text/event-stream")

    conv = db.get_conversation(conv_id)
    if not conv or conv.get("investor_session_id") != inv["session_id"]:
        return Response('data: {"type":"error","message":"Not found"}\n\n',
                        mimetype="text/event-stream")

    investor_session = db.get_investor_session(inv["session_id"])
    investor_name = investor_session["investor_name"] if investor_session else None

    db.add_message(conv_id, "user", question)

    history = db.get_messages(conv_id)
    if len([m for m in history if m["role"] == "user"]) == 1 and not conv["title"]:
        db.update_conversation_title(conv_id, question[:80])

    agent_history = [
        m for m in history
        if not (m["role"] == "user" and m["content"] == question)
    ]

    def generate():
        final_result = None
        for chunk in stream_answer(
            question=question,
            conversation_history=agent_history,
            investor_session_id=inv["session_id"],
            investor_name=investor_name,
            is_investor=True,
        ):
            yield chunk
            if '"type": "result"' in chunk or '"type":"result"' in chunk:
                try:
                    data = json.loads(chunk.replace("data: ", "").strip())
                    if data.get("type") == "result":
                        final_result = data.get("data", {})
                except Exception:
                    pass

        if final_result:
            db.add_message(
                conv_id, "assistant",
                final_result.get("answer", ""),
                sources=final_result.get("sources", []),
                draft_response=final_result.get("draft_response", ""),
                themes=final_result.get("themes", []),
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Investor: Request human handover ──────────────────────────────────────────

@app.route("/investor/chat/<int:conv_id>/request-handover", methods=["POST"])
@investor_login_required
def investor_request_handover(conv_id):
    inv = _current_investor()
    conv = db.get_conversation(conv_id)
    if not conv or conv.get("investor_session_id") != inv["session_id"]:
        return jsonify({"error": "not found"}), 404
    reason = request.json.get("reason", "Investor requested human assistance") if request.is_json else "Investor requested human assistance"
    # Add system message visible in chat
    db.add_message(conv_id, "system",
                   "Connecting you to a human agent. Please wait — a team member will join shortly.")
    db.create_handover_request(conv_id, inv["session_id"], reason)
    return jsonify({"status": "ok"})


@app.route("/investor/chat/<int:conv_id>/send", methods=["POST"])
@investor_login_required
def investor_chat_send(conv_id):
    """Investor sends a message to the human agent during an active handover."""
    inv = _current_investor()
    conv = db.get_conversation(conv_id)
    if not conv or conv.get("investor_session_id") != inv["session_id"]:
        return jsonify({"error": "not found"}), 404
    message = (request.json or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "empty"}), 400
    msg_id = db.add_message(conv_id, "user", message)
    return jsonify({"status": "ok", "msg_id": msg_id})


@app.route("/investor/chat/<int:conv_id>/poll")
@investor_login_required
def investor_chat_poll(conv_id):
    """Investor polls for new messages and conversation status."""
    inv = _current_investor()
    conv = db.get_conversation(conv_id)
    if not conv or conv.get("investor_session_id") != inv["session_id"]:
        return jsonify({"error": "not found"}), 404
    last_id = int(request.args.get("last_id", 0))
    new_msgs = db.get_messages_since(conv_id, last_id)
    handover = db.get_handover_for_conversation(conv_id)
    return jsonify({
        "messages": new_msgs,
        "conv_status": conv.get("status", "active"),
        "handover": handover,
    })


# ── Agent: Handover dashboard & chat ──────────────────────────────────────────

@app.route("/agent/handovers")
@login_required
def agent_handovers():
    handovers = db.get_pending_handovers()
    return render_template(
        "agent_handovers.html",
        user=_current_user(),
        handovers=handovers,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/agent/handovers/<int:handover_id>/claim", methods=["POST"])
@login_required
def agent_claim_handover(handover_id):
    user = _current_user()
    db.claim_handover(handover_id, user["id"])
    # Get conversation id to redirect to chat view
    conn = db.get_db()
    row = conn.execute("SELECT conversation_id FROM handover_requests WHERE id=?", (handover_id,)).fetchone()
    conn.close()
    if row:
        return redirect(url_for("agent_handover_chat", handover_id=handover_id))
    return redirect(url_for("agent_handovers"))


@app.route("/agent/handovers/<int:handover_id>/chat")
@login_required
def agent_handover_chat(handover_id):
    conn = db.get_db()
    handover = conn.execute("""
        SELECT h.*, c.title, c.investor_session_id, s.investor_name, s.investor_entity,
               u.name as claimed_by_name, c.status as conv_status
        FROM handover_requests h
        JOIN conversations c ON h.conversation_id = c.id
        LEFT JOIN investor_sessions s ON h.investor_session_id = s.id
        LEFT JOIN users u ON h.claimed_by = u.id
        WHERE h.id=?
    """, (handover_id,)).fetchone()
    conn.close()
    if not handover:
        flash("Handover not found.", "error")
        return redirect(url_for("agent_handovers"))
    handover = dict(handover)
    messages = db.get_messages(handover["conversation_id"])
    return render_template(
        "agent_handover_chat.html",
        user=_current_user(),
        handover=handover,
        messages=messages,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/agent/handovers/<int:handover_id>/reply", methods=["POST"])
@login_required
def agent_handover_reply(handover_id):
    user = _current_user()
    message = request.form.get("message", "").strip()
    if not message:
        return jsonify({"error": "empty"}), 400
    conn = db.get_db()
    row = conn.execute("SELECT conversation_id FROM handover_requests WHERE id=?", (handover_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    msg_id = db.add_message(row[0], "human_agent", message)
    return jsonify({"status": "ok", "msg_id": msg_id, "agent_name": user["name"]})


@app.route("/agent/handovers/<int:handover_id>/poll")
@login_required
def agent_handover_poll(handover_id):
    """Agent polls for new investor messages during live chat."""
    conn = db.get_db()
    row = conn.execute("SELECT conversation_id FROM handover_requests WHERE id=?", (handover_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    last_id = int(request.args.get("last_id", 0))
    new_msgs = db.get_messages_since(row[0], last_id)
    return jsonify({"messages": new_msgs})


@app.route("/agent/handovers/<int:handover_id>/resolve", methods=["POST"])
@login_required
def agent_resolve_handover(handover_id):
    conn = db.get_db()
    row = conn.execute("SELECT conversation_id FROM handover_requests WHERE id=?", (handover_id,)).fetchone()
    conn.close()
    if row:
        db.add_message(row[0], "system", "This conversation has been resolved by the fund team. You may continue using the portal.")
    db.resolve_handover(handover_id)
    flash("Handover resolved.", "success")
    return redirect(url_for("agent_handovers"))


@app.route("/api/handovers/pending-count")
@login_required
def api_pending_count():
    return jsonify({"count": db.get_pending_handover_count()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
