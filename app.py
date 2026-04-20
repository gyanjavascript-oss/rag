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
from agent import stream_answer, generate_investor_profile

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

import re as _re

@app.template_filter('clean_question')
def clean_question_filter(text: str) -> str:
    """Strip [Agent — Category]: prefix from questions shown in admin views."""
    q = _re.sub(r'^\[[^\]]+\]:\s*', '', (text or '')).strip()
    return (q[0].upper() + q[1:]) if q else (text or '')

@app.template_filter('from_json')
def from_json_filter(value):
    """Parse a JSON string into a Python object for use in templates."""
    if not value:
        return []
    try:
        import json as _json
        return _json.loads(value)
    except Exception:
        return []

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
FUND_DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FUND_DOCS_FOLDER, exist_ok=True)


def _make_conv_title(question: str) -> str:
    """Strip agent context prefix and return a clean, capitalised conversation title."""
    import re
    # Strip [Agent Name — Category]: prefix added by the chat UI
    q = re.sub(r'^\[[^\]]+\]:\s*', '', question).strip()
    # Capitalise first letter, trim to 72 chars
    if q:
        q = q[0].upper() + q[1:]
    return q[:72] if q else question[:72]

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
        # Validate that the session's investor_session_id still exists in DB
        sid = session.get("investor_session_id")
        if sid and not db.get_investor_session(sid):
            session.clear()
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


def _parse_qa_text(text: str) -> list[dict]:
    """Parse Q:/A: formatted text into list of {question, answer} dicts.
    Supports blocks separated by blank lines with Q: and A: prefixes,
    or numbered like '1. Q: ...' style.
    """
    import re
    entries = []
    # Normalize: split on double newlines to get blocks
    blocks = re.split(r'\n{2,}', text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        q, a = "", ""
        for line in lines:
            lc = line.lower()
            if lc.startswith("q:") or re.match(r'^\d+[\.\)]\s*q:', lc):
                q = re.sub(r'^(\d+[\.\)]\s*)?q:\s*', '', line, flags=re.IGNORECASE).strip()
            elif lc.startswith("a:") or re.match(r'^\d+[\.\)]\s*a:', lc):
                a = re.sub(r'^(\d+[\.\)]\s*)?a:\s*', '', line, flags=re.IGNORECASE).strip()
            elif q and not a:
                q += " " + line  # continuation of question
            elif a:
                a += " " + line  # continuation of answer
        if q and a:
            entries.append({"question": q, "answer": a})
    return entries


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("investor_login"))


@app.route("/admin")
def admin_index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.errorhandler(403)
def forbidden(e):
    # API requests get JSON, browser requests get redirected to login
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({"error": "Forbidden"}), 403
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


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = db.get_user_by_id(session["user_id"])
    if request.method == "POST":
        action = request.form.get("action")
        if action == "update_profile":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip()
            if not name or not email:
                flash("Name and email are required.", "error")
            elif db.update_user_profile(session["user_id"], name, email):
                session["user_name"] = name
                flash("Profile updated.", "success")
            else:
                flash("Email already in use.", "error")
        elif action == "change_password":
            current = request.form.get("current_password", "")
            new = request.form.get("new_password", "")
            confirm = request.form.get("confirm_password", "")
            if not current or not new or not confirm:
                flash("All password fields are required.", "error")
            elif new != confirm:
                flash("New passwords do not match.", "error")
            elif len(new) < 6:
                flash("Password must be at least 6 characters.", "error")
            elif db.change_user_password(session["user_id"], current, new):
                flash("Password changed successfully.", "success")
            else:
                flash("Current password is incorrect.", "error")
        return redirect(url_for("profile"))
    return render_template("profile.html", user=user)


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    stats = db.get_dashboard_stats()
    themes = db.get_theme_analytics()
    recent_questions = db.get_recent_questions(10)
    recent_conversations = db.list_conversations(limit=8)
    doc_stats = db.get_document_citation_stats()
    llm_usage_stats = db.get_llm_usage_stats() if _current_user().get("role") == "admin" else []
    return render_template(
        "dashboard.html",
        user=_current_user(),
        stats=stats,
        themes=themes,
        recent_questions=recent_questions,
        recent_conversations=recent_conversations,
        doc_stats=doc_stats,
        llm_usage_stats=llm_usage_stats,
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
        db.update_conversation_title(conv_id, _make_conv_title(question))

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
    """Legacy multi-file form POST — kept for fallback."""
    files = request.files.getlist("file")
    if not files or all(f.filename == "" for f in files):
        flash("No file selected.", "error")
        return redirect(url_for("documents"))
    doc_type = request.form.get("doc_type", "Other")
    user = _current_user()
    ok = 0
    for file in files:
        if not file.filename or not _allowed(file.filename):
            continue
        doc_name = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{doc_name}")
        file.save(filepath)
        try:
            dp.ingest_fund_document(filepath, doc_name, doc_type, user["id"], run_ai=True)
            ok += 1
        except Exception:
            pass
    flash(f'{ok} document{"s" if ok != 1 else ""} uploaded.', "success")
    return redirect(url_for("documents"))


@app.route("/documents/upload-one", methods=["POST"])
@login_required
def upload_document_one():
    """Upload and AI-process a single file. Returns JSON for per-file AJAX progress."""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "No file"}), 400
    if not _allowed(file.filename):
        return jsonify({"status": "error", "message": "Unsupported file type"}), 400

    doc_type = request.form.get("doc_type", "Other")
    user = _current_user()
    doc_name = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{doc_name}")
    file.save(filepath)

    try:
        doc_id = dp.ingest_fund_document(filepath, doc_name, doc_type, user["id"], run_ai=True)
        doc = db.get_fund_document(doc_id)
        return jsonify({
            "status": "ok",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "ai_doc_type": doc.get("ai_doc_type") or doc_type,
            "summary": (doc.get("summary") or "")[:120],
            "page_count": doc.get("page_count") or 0,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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
    doc = db.get_fund_document(doc_id)
    if not doc:
        return "Not found", 404
    filepath = doc.get("filepath", "")
    if not filepath:
        return "File not available", 404
    full_path = filepath if os.path.isabs(filepath) else os.path.join(UPLOAD_FOLDER, os.path.basename(filepath))
    if not os.path.exists(full_path):
        return "File not available", 404
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

    profile = None
    raw_profile = inv_session.get("profile_text")
    if raw_profile:
        try:
            profile = json.loads(raw_profile)
        except Exception:
            pass

    all_agents = db.list_marketplace_agents()
    for a in all_agents:
        if isinstance(a.get("tools"), str):
            try: a["tools"] = json.loads(a["tools"])
            except Exception: a["tools"] = []
    assigned_agent_ids = db.get_investor_agent_ids(session_id)

    return render_template(
        "investor_detail.html",
        user=_current_user(),
        inv_session=inv_session,
        inv_docs=[dict(r) for r in inv_docs],
        conversations=[dict(r) for r in conv_list],
        all_fund_docs=all_fund_docs,
        assigned_doc_ids=assigned_doc_ids,
        investor_user=investor_user,
        profile=profile,
        all_agents=all_agents,
        assigned_agent_ids=assigned_agent_ids,
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
    team_users = conn.execute("SELECT id, email, name, role, created_at FROM users WHERE role != 'investor' ORDER BY created_at").fetchall()
    investor_users = conn.execute("""
        SELECT u.id, u.email, u.name, u.role, u.created_at, s.investor_name, s.investor_entity
        FROM users u
        LEFT JOIN investor_sessions s ON s.id = u.investor_session_id
        WHERE u.role = 'investor'
        ORDER BY u.created_at DESC
    """).fetchall()
    conn.close()
    roles = db.list_roles()
    return render_template(
        "team.html",
        user=_current_user(),
        users=[dict(u) for u in team_users],
        investor_users=[dict(u) for u in investor_users],
        roles=roles,
        current_user_id=session["user_id"],
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


@app.route("/team/user/<int:user_id>/role", methods=["POST"])
@login_required
def change_user_role(user_id):
    if _current_user()["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    if user_id == session["user_id"]:
        flash("You cannot change your own role.", "error")
        return redirect(url_for("team"))
    role = request.form.get("role", "analyst")
    if role not in ("admin", "analyst"):
        flash("Invalid role.", "error")
        return redirect(url_for("team"))
    db.update_user_role(user_id, role)
    flash("Role updated.", "success")
    return redirect(url_for("team"))


@app.route("/team/user/<int:user_id>/delete", methods=["POST"])
@login_required
def delete_team_member(user_id):
    if _current_user()["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    if user_id == session["user_id"]:
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("team"))
    db.delete_user(user_id)
    flash("User removed.", "success")
    return redirect(url_for("team"))


@app.route("/team/roles/add", methods=["POST"])
@login_required
def add_role():
    if _current_user()["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    permissions = request.form.getlist("permissions")
    if not name:
        flash("Role name is required.", "error")
        return redirect(url_for("team"))
    try:
        db.create_role(name, description, permissions)
        flash(f"Role '{name}' created.", "success")
    except Exception as e:
        flash(f"Error: role name already exists.", "error")
    return redirect(url_for("team"))


@app.route("/team/roles/<int:role_id>/edit", methods=["POST"])
@login_required
def edit_role(role_id):
    if _current_user()["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    permissions = request.form.getlist("permissions")
    db.update_role(role_id, name, description, permissions)
    flash("Role updated.", "success")
    return redirect(url_for("team"))


@app.route("/team/roles/<int:role_id>/delete", methods=["POST"])
@login_required
def delete_role(role_id):
    if _current_user()["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    db.delete_role(role_id)
    flash("Role deleted.", "success")
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


@app.route("/investors/<int:session_id>/generate-profile", methods=["POST"])
@login_required
def investor_generate_profile(session_id):
    inv = db.get_investor_session(session_id)
    if not inv:
        return jsonify({"error": "Not found"}), 404
    questions = db.get_investor_questions(session_id)
    profile = generate_investor_profile(
        investor_name=inv["investor_name"],
        entity=inv.get("investor_entity", ""),
        notes=inv.get("notes", ""),
        questions=questions,
    )
    if profile:
        import json as _json
        db.save_investor_profile(session_id, _json.dumps(profile))
        return jsonify({"ok": True, "profile": profile})
    return jsonify({"error": "Could not generate profile"}), 500


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


@app.route("/investor/profile", methods=["GET", "POST"])
@investor_login_required
def investor_profile():
    inv = _current_investor()
    user = db.get_investor_user_by_id(inv["id"])
    if request.method == "POST":
        action = request.form.get("action")
        if action == "update_profile":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip()
            if not name or not email:
                flash("Name and email are required.", "error")
            elif db.update_user_profile(inv["id"], name, email):
                flash("Profile updated.", "success")
            else:
                flash("Email already in use.", "error")
        elif action == "change_password":
            current = request.form.get("current_password", "")
            new = request.form.get("new_password", "")
            confirm = request.form.get("confirm_password", "")
            if not current or not new or not confirm:
                flash("All password fields are required.", "error")
            elif new != confirm:
                flash("New passwords do not match.", "error")
            elif len(new) < 6:
                flash("Password must be at least 6 characters.", "error")
            elif db.change_user_password(inv["id"], current, new):
                flash("Password changed successfully.", "success")
            else:
                flash("Current password is incorrect.", "error")
        return redirect(url_for("investor_profile"))
    return render_template("investor_profile.html",
                           user=user,
                           fund_name=os.getenv("FUND_NAME", "the Fund"))


@app.route("/investor/logout")
def investor_logout():
    session.pop("investor_user_id", None)
    session.pop("investor_name", None)
    session.pop("investor_session_id", None)
    return redirect(url_for("investor_login"))


# ── Investor: My Answers (session-scoped Q&A) ─────────────────────────────────

@app.route("/investor/my-answers")
@investor_login_required
def investor_my_answers():
    inv = _current_investor()
    answers = db.list_session_answers(inv["session_id"])
    return render_template("investor_my_answers.html", answers=answers)

@app.route("/investor/my-answers/add", methods=["POST"])
@investor_login_required
def investor_add_answer():
    inv = _current_investor()
    question = request.form.get("question", "").strip()
    answer = request.form.get("answer", "").strip()
    if question and answer:
        db.add_session_answer(inv["session_id"], question, answer)
    return redirect(url_for("investor_my_answers"))

@app.route("/investor/my-answers/bulk", methods=["POST"])
@investor_login_required
def investor_bulk_answers():
    """Parse bulk paste: sections separated by blank lines, Q: and A: prefixes."""
    inv = _current_investor()
    raw = request.form.get("bulk_text", "")
    entries = _parse_qa_text(raw)
    for e in entries:
        db.add_session_answer(inv["session_id"], e["question"], e["answer"])
    return redirect(url_for("investor_my_answers"))

@app.route("/investor/my-answers/<int:answer_id>/delete", methods=["POST"])
@investor_login_required
def investor_delete_answer(answer_id):
    inv = _current_investor()
    db.delete_session_answer(answer_id, inv["session_id"])
    return redirect(url_for("investor_my_answers"))


@app.route("/investor/portal")
@investor_login_required
def investor_portal():
    inv = _current_investor()
    investor_session = db.get_investor_session(inv["session_id"])
    conversations = db.list_investor_conversations(inv["session_id"])
    assigned_docs = db.get_assigned_documents(inv["session_id"])
    assigned_agent_count = len(db.get_investor_agent_ids(inv["session_id"]))
    return render_template(
        "investor_portal.html",
        investor=inv,
        investor_session=investor_session,
        conversations=conversations,
        assigned_docs=assigned_docs,
        assigned_agent_count=assigned_agent_count,
        active_conv=None,
        messages=[],
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/investor/chat/new", methods=["POST"])
@investor_login_required
def investor_new_chat():
    inv = _current_investor()
    title = request.form.get("title") or None
    conv_id = db.create_conversation(
        created_by=inv["id"],
        investor_session_id=inv["session_id"],
        title=title,
    )
    # Return JSON only for explicit AJAX requests, otherwise redirect
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"conversation_id": conv_id})
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
    assigned_agent_count = len(db.get_investor_agent_ids(inv["session_id"]))
    return render_template(
        "investor_portal.html",
        investor=inv,
        investor_session=investor_session,
        conversations=conversations,
        assigned_docs=assigned_docs,
        assigned_agent_count=assigned_agent_count,
        active_conv=conv,
        messages=messages,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/investor/chat/<int:conv_id>/delete", methods=["POST"])
@investor_login_required
def investor_delete_chat(conv_id):
    inv = _current_investor()
    db.soft_delete_investor_conversation(conv_id, inv["session_id"])
    return redirect(url_for("investor_portal"))


@app.route("/investor/chat/<int:conv_id>/rename", methods=["POST"])
@investor_login_required
def investor_rename_chat(conv_id):
    inv = _current_investor()
    title = request.form.get("title", "").strip()
    if title:
        db.rename_investor_conversation(conv_id, inv["session_id"], title)
    return redirect(url_for("investor_chat_view", conv_id=conv_id))


@app.route("/investor/chat/<int:conv_id>/stream")
@investor_login_required
def investor_chat_stream(conv_id):
    raw_question = request.args.get("q", "").strip()
    agent_id = request.args.get("agent_id", type=int)
    active_tools_param = request.args.get("tools", "").strip()
    active_tools = [t.strip() for t in active_tools_param.split(",") if t.strip()] if active_tools_param else None
    inv = _current_investor()

    if not raw_question:
        return Response('data: {"type":"error","message":"Empty question"}\n\n',
                        mimetype="text/event-stream")

    conv = db.get_conversation(conv_id)
    if not conv or conv.get("investor_session_id") != inv["session_id"]:
        return Response('data: {"type":"error","message":"Not found"}\n\n',
                        mimetype="text/event-stream")

    # Strip agent context prefix before saving — keep clean question in DB
    # but pass full contextual question to the LLM so it knows the agent persona
    question = _make_conv_title(raw_question)  # strips [Agent — Cat]: prefix
    question_for_llm = raw_question            # keeps prefix for LLM context

    investor_session = db.get_investor_session(inv["session_id"])
    investor_name = investor_session["investor_name"] if investor_session else None

    # Load agent memory if this is an agent chat
    agent_memories = []
    if agent_id:
        agent_memories = db.get_agent_memory(agent_id, inv["session_id"])

    db.add_message(conv_id, "user", question)

    history = db.get_messages(conv_id)
    if len([m for m in history if m["role"] == "user"]) == 1:
        db.update_conversation_title(conv_id, question)

    agent_history = [
        m for m in history
        if not (m["role"] == "user" and m["content"] == question)
    ]

    def generate():
        final_result = None
        for chunk in stream_answer(
            question=question_for_llm,
            conversation_history=agent_history,
            investor_session_id=inv["session_id"],
            investor_name=investor_name,
            is_investor=True,
            agent_memories=agent_memories,
            allowed_tools=active_tools,
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
            # Save memory: what was learned from this exchange
            if agent_id:
                confidence = final_result.get("confidence", "")
                gaps = final_result.get("gaps")
                if confidence == "low" or gaps:
                    db.add_agent_memory(
                        agent_id, inv["session_id"], "gap",
                        f"Q: {question[:200]} | Gap: {str(gaps)[:300]}"
                    )
                elif confidence == "high":
                    themes = final_result.get("themes", [])
                    if themes:
                        db.add_agent_memory(
                            agent_id, inv["session_id"], "interest",
                            f"Investor asked about: {', '.join(themes[:3])} — '{question[:150]}'"
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
    msg_id = db.add_message(row['conversation_id'], "human_agent", message)
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
    new_msgs = db.get_messages_since(row['conversation_id'], last_id)
    return jsonify({"messages": new_msgs})


@app.route("/agent/handovers/<int:handover_id>/resolve", methods=["POST"])
@login_required
def agent_resolve_handover(handover_id):
    conn = db.get_db()
    row = conn.execute("SELECT conversation_id FROM handover_requests WHERE id=?", (handover_id,)).fetchone()
    conn.close()
    if row:
        db.add_message(row['conversation_id'], "system", "This conversation has been resolved by the fund team. You may continue using the portal.")
    db.resolve_handover(handover_id)
    flash("Handover resolved.", "success")
    return redirect(url_for("agent_handovers"))


@app.route("/api/handovers/pending-count")
@login_required
def api_pending_count():
    return jsonify({"count": db.get_pending_handover_count()})


# ── Knowledge Base ─────────────────────────────────────────────────────────────

@app.route("/knowledge-base")
@login_required
def knowledge_base():
    entries = db.list_kb_entries()
    return render_template(
        "knowledge_base.html",
        user=_current_user(),
        entries=entries,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/knowledge-base/add", methods=["POST"])
@login_required
def kb_add():
    question = request.form.get("question", "").strip()
    answer = request.form.get("answer", "").strip()
    tags = request.form.get("tags", "").strip()
    if not question or not answer:
        flash("Question and answer are required.", "error")
        return redirect(url_for("knowledge_base"))
    db.add_kb_entry(question, answer, tags, _current_user()["id"])
    flash("Knowledge base entry added.", "success")
    return redirect(url_for("knowledge_base"))


@app.route("/knowledge-base/<int:entry_id>/edit", methods=["GET", "POST"])
@login_required
def kb_edit(entry_id):
    entry = db.get_kb_entry(entry_id)
    if not entry:
        flash("Entry not found.", "error")
        return redirect(url_for("knowledge_base"))
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        answer = request.form.get("answer", "").strip()
        tags = request.form.get("tags", "").strip()
        if not question or not answer:
            flash("Question and answer are required.", "error")
        else:
            db.update_kb_entry(entry_id, question, answer, tags)
            flash("Entry updated.", "success")
            return redirect(url_for("knowledge_base"))
    return render_template(
        "knowledge_base.html",
        user=_current_user(),
        entries=db.list_kb_entries(),
        edit_entry=entry,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/knowledge-base/<int:entry_id>/delete", methods=["POST"])
@login_required
def kb_delete(entry_id):
    db.delete_kb_entry(entry_id)
    flash("Entry deleted.", "success")
    return redirect(url_for("knowledge_base"))


@app.route("/admin/kb/bulk-import", methods=["POST"])
@login_required
def kb_bulk_import():
    raw = request.form.get("bulk_text", "")
    entries = _parse_qa_text(raw)
    count = db.bulk_import_kb(entries, _current_user()["id"])
    flash(f"Imported {count} entries into the Knowledge Base.", "success")
    return redirect(url_for("knowledge_base"))


# ── LLM Key Management ────────────────────────────────────────────────────────

@app.route("/admin/llm-keys")
@login_required
def llm_keys():
    if _current_user().get("role") != "admin":
        flash("Access denied.", "error")
        return redirect(url_for("admin_index"))
    keys = db.list_llm_keys()
    stats = db.get_llm_usage_stats()
    return render_template(
        "llm_keys.html",
        user=_current_user(),
        keys=keys,
        stats=stats,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
    )


@app.route("/admin/llm-keys/add", methods=["POST"])
@login_required
def llm_keys_add():
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Access denied"}), 403
    from llm_crypto import encrypt_key
    name = request.form.get("name", "").strip()
    provider = request.form.get("provider", "openai").strip()
    model = request.form.get("model", "gpt-4o").strip()
    api_key = request.form.get("api_key", "").strip()
    if not name or not api_key:
        flash("Name and API key are required.", "error")
        return redirect(url_for("llm_keys"))
    hint = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    enc = encrypt_key(api_key)
    base_url = request.form.get("base_url", "").strip()
    db.add_llm_key(name, provider, model, enc, hint, base_url=base_url)
    flash("LLM key added.", "success")
    return redirect(url_for("llm_keys"))


@app.route("/admin/llm-keys/<int:key_id>/edit", methods=["POST"])
@login_required
def llm_keys_edit(key_id):
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Access denied"}), 403
    from llm_crypto import encrypt_key
    name = request.form.get("name", "").strip()
    model = request.form.get("model", "").strip()
    is_active = request.form.get("is_active") == "1"
    api_key = request.form.get("api_key", "").strip()
    updates = {}
    if name:
        updates["name"] = name
    if model:
        updates["model"] = model
    updates["is_active"] = 1 if is_active else 0
    if api_key:
        updates["api_key_enc"] = encrypt_key(api_key)
        updates["api_key_hint"] = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    base_url = request.form.get("base_url", "").strip()
    updates["base_url"] = base_url
    db.update_llm_key(key_id, **updates)
    flash("LLM key updated.", "success")
    return redirect(url_for("llm_keys"))


@app.route("/admin/llm-keys/<int:key_id>/delete", methods=["POST"])
@login_required
def llm_keys_delete(key_id):
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Access denied"}), 403
    db.delete_llm_key(key_id)
    flash("LLM key deleted.", "success")
    return redirect(url_for("llm_keys"))


@app.route("/admin/llm-keys/<int:key_id>/move", methods=["POST"])
@login_required
def llm_keys_move(key_id):
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Access denied"}), 403
    direction = request.form.get("direction")
    if direction in ("up", "down"):
        db.move_llm_key(key_id, direction)
    return redirect(url_for("llm_keys"))


# ── Agent Marketplace ──────────────────────────────────────────────────────────


# ── Custom Agent Builder ───────────────────────────────────────────────────────

CUSTOM_AGENT_TOOLS = {
    "web_search":               "🔍 Web Search",
    "browse_url":               "🌐 URL Browser",
    "search_fund_documents":    "📄 Fund Document Search",
    "search_investor_documents":"👤 Investor Document Search",
    "list_available_documents": "📋 List Documents",
    "find_similar_questions":   "💬 Find Similar Questions",
}

@app.route("/custom-agents")
@login_required
def custom_agents():
    agents = db.list_custom_agents()
    return render_template("custom_agents.html", agents=agents, user=_current_user())


@app.route("/custom-agents/new", methods=["GET", "POST"])
@login_required
def custom_agent_new():
    if request.method == "POST":
        tools = request.form.getlist("tools")
        agent = db.create_custom_agent(
            name=request.form["name"].strip(),
            description=request.form.get("description", "").strip(),
            icon=request.form.get("icon", "🤖").strip() or "🤖",
            system_prompt=request.form.get("system_prompt", "").strip(),
            user_prompt=request.form.get("user_prompt", "").strip(),
            input_type=request.form.get("input_type", "chat"),
            output_type=request.form.get("output_type", "chat"),
            output_webhook_url=request.form.get("output_webhook_url", "").strip(),
            output_webhook_secret=request.form.get("output_webhook_secret", "").strip(),
            tools=tools,
            created_by=session["user_id"],
        )
        flash(f"Agent '{agent['name']}' created.", "success")
        return redirect(url_for("custom_agent_detail", agent_id=agent["id"]))
    return render_template("custom_agent_form.html", agent=None, all_tools=CUSTOM_AGENT_TOOLS, user=_current_user())


@app.route("/custom-agents/<int:agent_id>", methods=["GET", "POST"])
@login_required
def custom_agent_detail(agent_id):
    agent = db.get_custom_agent(agent_id)
    if not agent:
        flash("Agent not found.", "error")
        return redirect(url_for("custom_agents"))
    if request.method == "POST":
        action = request.form.get("action")
        if action == "delete":
            db.delete_custom_agent(agent_id)
            flash("Agent deleted.", "success")
            return redirect(url_for("custom_agents"))
        tools = request.form.getlist("tools")
        db.update_custom_agent(
            agent_id=agent_id,
            name=request.form["name"].strip(),
            description=request.form.get("description", "").strip(),
            icon=request.form.get("icon", "🤖").strip() or "🤖",
            system_prompt=request.form.get("system_prompt", "").strip(),
            user_prompt=request.form.get("user_prompt", "").strip(),
            input_type=request.form.get("input_type", "chat"),
            output_type=request.form.get("output_type", "chat"),
            output_webhook_url=request.form.get("output_webhook_url", "").strip(),
            output_webhook_secret=request.form.get("output_webhook_secret", "").strip(),
            tools=tools,
        )
        flash("Agent updated.", "success")
        return redirect(url_for("custom_agent_detail", agent_id=agent_id))
    runs = db.get_custom_agent_runs(agent_id, limit=20)
    schedules = db.list_agent_schedules(agent_id)
    return render_template("custom_agent_form.html", agent=agent,
                           all_tools=CUSTOM_AGENT_TOOLS, runs=runs,
                           schedules=schedules, user=_current_user())


@app.route("/custom-agents/<int:agent_id>/chat")
@login_required
def custom_agent_chat(agent_id):
    agent = db.get_custom_agent(agent_id)
    if not agent:
        return redirect(url_for("custom_agents"))
    return render_template("custom_agent_chat.html", agent=agent, user=_current_user())


@app.route("/custom-agents/<int:agent_id>/stream")
@login_required
def custom_agent_stream(agent_id):
    agent = db.get_custom_agent(agent_id)
    if not agent:
        return Response('data: {"type":"error","message":"Agent not found"}\n\n',
                        mimetype="text/event-stream")
    question = request.args.get("q", "").strip()
    if not question:
        return Response('data: {"type":"error","message":"Empty input"}\n\n',
                        mimetype="text/event-stream")

    # Build custom system prompt
    custom_prompt = agent.get("system_prompt") or ""
    # Apply user prompt template: replace {{input}} with the actual question
    user_prompt_tpl = agent.get("user_prompt") or ""
    question_for_llm = user_prompt_tpl.replace("{{input}}", question).strip() if user_prompt_tpl else question
    agent_tools = agent.get("tools") or []

    def generate():
        full_answer = ""
        final_result = None
        for chunk in stream_answer(
            question=question_for_llm,
            conversation_history=[],
            is_investor=False,
            custom_system_prompt=custom_prompt or None,
            allowed_tools=agent_tools or None,
        ):
            yield chunk
            if '"type": "result"' in chunk or '"type":"result"' in chunk:
                try:
                    data = json.loads(chunk.replace("data: ", "").strip())
                    if data.get("type") == "result":
                        final_result = data.get("data", {})
                        full_answer = final_result.get("answer", "")
                except Exception:
                    pass

        # Save run
        if final_result:
            db.save_custom_agent_run(
                agent_id=agent_id,
                input_text=question,
                output_text=full_answer,
                sources=final_result.get("sources", []),
                confidence=final_result.get("confidence", ""),
                input_src="chat",
            )
            # Fire webhook if configured
            _fire_webhook(agent, full_answer, final_result, question)

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


def _fire_webhook(agent: dict, answer: str, result: dict, input_text: str):
    url = agent.get("output_webhook_url", "").strip()
    if not url:
        return
    try:
        import urllib.request, hmac, hashlib
        payload = json.dumps({
            "agent_id": agent["id"],
            "agent_name": agent["name"],
            "input": input_text,
            "output": answer,
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", ""),
        }).encode()
        headers = {"Content-Type": "application/json"}
        secret = agent.get("output_webhook_secret", "").strip()
        if secret:
            sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
            headers["X-Signature-SHA256"] = f"sha256={sig}"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


# ── Public API endpoint ────────────────────────────────────────────────────────

@app.route("/api/v1/agents/<api_key>/run", methods=["POST"])
def custom_agent_api_run(api_key):
    agent = db.get_custom_agent_by_key(api_key)
    if not agent:
        return jsonify({"error": "Invalid API key"}), 401
    if agent.get("input_type") not in ("api", "both"):
        return jsonify({"error": "This agent does not accept API input"}), 403

    body = request.get_json(silent=True) or {}
    question = (body.get("input") or body.get("question") or body.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'input' field"}), 400

    custom_prompt = agent.get("system_prompt") or ""
    user_prompt_tpl = agent.get("user_prompt") or ""
    question_for_llm = user_prompt_tpl.replace("{{input}}", question).strip() if user_prompt_tpl else question
    agent_tools = agent.get("tools") or []
    from agent import answer_question
    result = answer_question(
        question=question_for_llm,
        conversation_history=[],
        is_investor=False,
        custom_system_prompt=custom_prompt or None,
        allowed_tools=agent_tools or None,
    )

    answer = result.get("answer", "")
    db.save_custom_agent_run(
        agent_id=agent["id"],
        input_text=question,
        output_text=answer,
        sources=result.get("sources", []),
        confidence=result.get("confidence", ""),
        input_src="api",
    )
    _fire_webhook(agent, answer, result, question)

    return jsonify({
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "input": question,
        "output": answer,
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", ""),
        "themes": result.get("themes", []),
    })


# ── Agent Scheduler routes ────────────────────────────────────────────────────

@app.route("/custom-agents/<int:agent_id>/schedules", methods=["POST"])
@login_required
def create_agent_schedule(agent_id):
    agent = db.get_custom_agent(agent_id)
    if not agent:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or request.form
    schedule = db.create_agent_schedule(
        agent_id=agent_id,
        name=(data.get("name") or "").strip() or "Scheduled Run",
        input_text=(data.get("input_text") or "").strip(),
        schedule_type=data.get("schedule_type", "interval"),
        interval_minutes=int(data.get("interval_minutes") or 60),
        daily_time=data.get("daily_time", "09:00"),
        weekly_day=int(data.get("weekly_day") or 1),
    )
    return jsonify(schedule)


@app.route("/custom-agents/<int:agent_id>/schedules/<int:schedule_id>/toggle", methods=["POST"])
@login_required
def toggle_agent_schedule(agent_id, schedule_id):
    db.toggle_agent_schedule(schedule_id)
    return jsonify({"ok": True})


@app.route("/custom-agents/<int:agent_id>/schedules/<int:schedule_id>", methods=["DELETE"])
@login_required
def delete_agent_schedule(agent_id, schedule_id):
    db.delete_agent_schedule(schedule_id)
    return jsonify({"ok": True})


@app.route("/api/v1/cron/tick", methods=["POST"])
def cron_tick():
    """Called by server cron every minute to execute due schedules."""
    secret = request.headers.get("X-Cron-Secret", "")
    if secret != app.config.get("CRON_SECRET", "vcurd-cron-2026"):
        return jsonify({"error": "Unauthorized"}), 401

    from agent import answer_question
    due = db.get_due_schedules()
    results = []
    for sched in due:
        try:
            custom_prompt = sched.get("system_prompt") or ""
            user_prompt_tpl = sched.get("user_prompt") or ""
            q = sched["input_text"]
            question_for_llm = user_prompt_tpl.replace("{{input}}", q).strip() if user_prompt_tpl else q

            # Parse agent tools
            tools = sched.get("tools") or []
            if isinstance(tools, str):
                try: tools = json.loads(tools)
                except Exception: tools = []

            result = answer_question(
                question=question_for_llm,
                conversation_history=[],
                is_investor=False,
                custom_system_prompt=custom_prompt or None,
                allowed_tools=tools or None,
            )
            answer = result.get("answer", "")
            agent_obj = {
                "id": sched["agent_id"], "output_type": sched.get("output_type", "chat"),
                "output_webhook_url": sched.get("output_webhook_url", ""),
                "output_webhook_secret": sched.get("output_webhook_secret", ""),
                "api_key": sched.get("api_key", ""), "name": sched.get("name", ""),
            }
            db.save_custom_agent_run(
                agent_id=sched["agent_id"],
                input_text=q,
                output_text=answer,
                sources=result.get("sources", []),
                confidence=result.get("confidence", ""),
                input_src="schedule",
            )
            _fire_webhook(agent_obj, answer, result, q)
            results.append({"schedule_id": sched["id"], "status": "ok"})
        except Exception as ex:
            results.append({"schedule_id": sched["id"], "status": "error", "error": str(ex)})

    return jsonify({"ran": len(due), "results": results})


@app.route("/agent-marketplace")
@login_required
def agent_marketplace():
    category = request.args.get("category", "")
    agents = db.list_marketplace_agents(category if category else None)
    categories = db.list_marketplace_categories()
    # Parse tools JSON for display
    for a in agents:
        if isinstance(a.get("tools"), str):
            try:
                a["tools"] = json.loads(a["tools"])
            except Exception:
                a["tools"] = []
    return render_template("agent_marketplace.html",
                           agents=agents,
                           categories=categories,
                           selected_category=category,
                           user=_current_user())


@app.route("/agent-marketplace/<int:agent_id>/assign", methods=["POST"])
@login_required
def agent_marketplace_assign(agent_id):
    investor_session_id = request.form.get("investor_session_id", type=int)
    if not investor_session_id:
        flash("Please select an investor.", "error")
        return redirect(url_for("agent_marketplace"))
    user = _current_user()
    db.assign_agent_to_investor(agent_id, investor_session_id, user["id"])
    return redirect(url_for("agent_marketplace"))


@app.route("/agent-marketplace/<int:agent_id>/unassign", methods=["POST"])
@login_required
def agent_marketplace_unassign(agent_id):
    investor_session_id = request.form.get("investor_session_id", type=int)
    if investor_session_id:
        db.unassign_agent_from_investor(agent_id, investor_session_id)
    return redirect(request.referrer or url_for("agent_marketplace"))


@app.route("/investors/<int:session_id>/assign-agent", methods=["POST"])
@login_required
def investor_assign_agent(session_id):
    selected_ids = set(int(x) for x in request.form.getlist("agent_ids") if x.isdigit())
    current_ids = db.get_investor_agent_ids(session_id)
    user_id = _current_user()["id"]
    for aid in selected_ids - current_ids:
        db.assign_agent_to_investor(aid, session_id, user_id)
    for aid in current_ids - selected_ids:
        db.unassign_agent_from_investor(aid, session_id)
    return redirect(url_for("investor_detail", session_id=session_id))


@app.route("/investors/<int:session_id>/unassign-agent/<int:agent_id>", methods=["POST"])
@login_required
def investor_unassign_agent(session_id, agent_id):
    db.unassign_agent_from_investor(agent_id, session_id)
    return redirect(url_for("investor_detail", session_id=session_id))


@app.route("/investor/agents")
@investor_login_required
def investor_agents():
    inv = _current_investor()
    sid = inv.get("session_id")
    agents = db.get_assigned_agents(sid) if sid else []
    for a in agents:
        if isinstance(a.get("tools"), str):
            try:
                a["tools"] = json.loads(a["tools"])
            except Exception:
                a["tools"] = []
    investor = db.get_investor_session(sid)
    return render_template("investor_agents.html", agents=agents, investor=investor)


@app.route("/investor/agents/<int:agent_id>/chat")
@app.route("/investor/agents/<int:agent_id>/chat/<int:conv_id>")
@investor_login_required
def investor_agent_chat(agent_id, conv_id=None):
    inv = _current_investor()
    sid = inv.get("session_id")
    agent = db.get_marketplace_agent(agent_id)
    if not agent:
        flash("Agent not found.", "error")
        return redirect(url_for("investor_agents"))
    assigned_ids = db.get_investor_agent_ids(sid)
    if agent_id not in assigned_ids:
        flash("This agent is not assigned to you.", "error")
        return redirect(url_for("investor_agents"))
    if isinstance(agent.get("tools"), str):
        try: agent["tools"] = json.loads(agent["tools"])
        except Exception: agent["tools"] = []
    investor = db.get_investor_session(sid)

    # Create a new conversation if none provided
    if not conv_id:
        conv_id = db.create_conversation(
            created_by=inv["id"],
            investor_session_id=sid,
            title=f"{agent['name']} Session",
        )
        return redirect(url_for("investor_agent_chat", agent_id=agent_id, conv_id=conv_id))

    # Verify conversation belongs to this investor
    conv = db.get_conversation(conv_id)
    if not conv or conv.get("investor_session_id") != sid:
        return redirect(url_for("investor_agent_chat", agent_id=agent_id))

    messages = db.get_messages(conv_id)
    return render_template("investor_agent_chat.html",
                           agent=agent, investor=investor,
                           conv_id=conv_id, messages=messages)


# ── Plugins ───────────────────────────────────────────────────────────────────

import plugins as plg
import fund_research as fr


@app.route("/plugins")
@login_required
def plugins_dashboard():
    investor_sessions = db.list_investor_sessions()
    return render_template(
        "plugins.html",
        user=_current_user(),
        investor_sessions=investor_sessions,
        fund_name=os.getenv("FUND_NAME", "DDQ Platform"),
        active_tab=request.args.get("tab", "consistency"),
    )


def _plugin_route(fn, *args, **kwargs):
    """Wrap any plugin call so exceptions always return JSON, never HTML."""
    try:
        return jsonify(fn(*args, **kwargs))
    except Exception as e:
        import traceback
        app.logger.error("Plugin error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Plugin 1 — Consistency Audit
@app.route("/api/plugins/consistency-audit", methods=["POST"])
@login_required
def api_plugin_consistency_audit():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id") or (request.form.get("session_id"))
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    return _plugin_route(plg.run_consistency_audit, int(session_id))


# Plugin 2 — Gap Report
@app.route("/api/plugins/gap-report", methods=["POST"])
@login_required
def api_plugin_gap_report():
    return _plugin_route(plg.run_gap_report)


# Plugin 3 — Investor Memory
@app.route("/api/plugins/investor-memory", methods=["POST"])
@login_required
def api_plugin_investor_memory():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id") or request.form.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    return _plugin_route(plg.build_investor_memory, int(session_id))


# Plugin 4 — Staleness Monitor
@app.route("/api/plugins/staleness", methods=["POST"])
@login_required
def api_plugin_staleness():
    return _plugin_route(plg.run_staleness_monitor)


# Plugin 5 — EDGAR Sync
@app.route("/api/plugins/edgar-sync", methods=["POST"])
@login_required
def api_plugin_edgar_sync():
    body = request.get_json(silent=True) or {}
    return _plugin_route(plg.run_edgar_sync, body.get("fund_name", "").strip())


# Plugin 6 — ESG Auto-Population
@app.route("/api/plugins/esg-autopop", methods=["POST"])
@login_required
def api_plugin_esg_autopop():
    return _plugin_route(plg.run_esg_autopop)


# Plugin 7 — Jurisdiction Mapping
@app.route("/api/plugins/jurisdiction", methods=["POST"])
@login_required
def api_plugin_jurisdiction():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id") or request.form.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    return _plugin_route(plg.run_jurisdiction_mapping, int(session_id))


# Plugin 8 — Staleness Orchestrator
@app.route("/api/plugins/staleness-orchestrator", methods=["POST"])
@login_required
def api_plugin_staleness_orchestrator():
    return _plugin_route(plg.run_staleness_orchestrator)


# ── Fund Research (Risk Assessment Agent) ─────────────────────────────────────

@app.route("/api/fund-research/memory", methods=["GET"])
@login_required
def api_fund_research_memory():
    """Admin: view what the research agent has learned."""
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Admin only"}), 403
    conn = db.get_db()
    rows = conn.execute(
        "SELECT * FROM fund_research_memory ORDER BY hits DESC, updated_at DESC LIMIT 200"
    ).fetchall()
    db.put_db(conn)
    return jsonify([dict(r) for r in rows])

@app.route("/api/fund-research/memory/<int:mid>/delete", methods=["POST"])
@login_required
def api_fund_research_memory_delete(mid):
    """Admin: delete a learned memory entry."""
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Admin only"}), 403
    conn = db.get_db()
    conn.execute("DELETE FROM fund_research_memory WHERE id=%s", (mid,))
    conn.commit()
    db.put_db(conn)
    return jsonify({"ok": True})

@app.route("/api/fund-quickpick", methods=["GET"])
@login_required
def api_fund_quickpick_list():
    conn = db.get_db()
    rows = conn.execute("SELECT * FROM fund_quickpick ORDER BY fund_name").fetchall()
    db.put_db(conn)
    return jsonify([dict(r) for r in rows])

@app.route("/api/fund-quickpick/add", methods=["POST"])
@login_required
def api_fund_quickpick_add():
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json() or {}
    name = data.get("fund_name", "").strip()
    ticker = data.get("ticker", "").strip().upper()
    if not name or not ticker:
        return jsonify({"error": "Name and ticker required"}), 400
    conn = db.get_db()
    conn.execute(
        "INSERT INTO fund_quickpick (fund_name, ticker, category, exchange, country) VALUES (?,?,?,?,?)",
        (name, ticker, data.get("category","ETF"), data.get("exchange",""), data.get("country","")),
    )
    conn.commit()
    db.put_db(conn)
    return jsonify({"ok": True})

@app.route("/api/fund-quickpick/<int:qid>/edit", methods=["POST"])
@login_required
def api_fund_quickpick_edit(qid):
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json() or {}
    conn = db.get_db()
    conn.execute(
        "UPDATE fund_quickpick SET fund_name=?, ticker=?, category=?, exchange=?, country=? WHERE id=?",
        (data.get("fund_name",""), data.get("ticker","").upper(),
         data.get("category","ETF"), data.get("exchange",""), data.get("country",""), qid),
    )
    conn.commit()
    db.put_db(conn)
    return jsonify({"ok": True})

@app.route("/api/fund-quickpick/<int:qid>/delete", methods=["POST"])
@login_required
def api_fund_quickpick_delete(qid):
    if _current_user().get("role") != "admin":
        return jsonify({"error": "Admin only"}), 403
    conn = db.get_db()
    conn.execute("DELETE FROM fund_quickpick WHERE id=?", (qid,))
    conn.commit()
    db.put_db(conn)
    return jsonify({"ok": True})


@app.route("/fund-research")
@login_required
def fund_research():
    conn = db.get_db()
    rows = conn.execute(
        "SELECT w.*, r.generated_at AS report_at, r.report_json FROM fund_watchlist w "
        "LEFT JOIN fund_research_reports r ON r.fund_id = w.id "
        "WHERE w.deleted_at IS NULL "
        "ORDER BY w.added_at DESC"
    ).fetchall()
    db.put_db(conn)

    import json as _json
    funds = []
    for row in rows:
        f = dict(row)
        rj = {}
        try:
            rj = _json.loads(f.get("report_json") or "{}") or {}
        except Exception:
            pass
        ra = rj.get("risk_assessment", {}) or {}
        geo = rj.get("geopolitical_risks", {}) or {}
        av = rj.get("analyst_verdict", {}) or {}
        f["risk_rating"] = ra.get("overall_risk_rating", "")
        f["geo_risk"] = geo.get("overall_geo_risk", "")
        f["conviction"] = av.get("conviction_rating", "")
        funds.append(f)

    u = _current_user()
    qp_conn = db.get_db()
    qp_rows = qp_conn.execute("SELECT * FROM fund_quickpick ORDER BY fund_name").fetchall()
    db.put_db(qp_conn)
    return render_template("fund_research.html",
                           user=u,
                           funds=funds,
                           is_admin=(u.get("role") == "admin"),
                           db_quickpicks=[dict(r) for r in qp_rows],
                           fund_name=os.getenv("FUND_NAME", "DDQ Platform"))


@app.route("/fund-research/add", methods=["POST"])
@login_required
def fund_research_add():
    name = request.form.get("fund_name", "").strip()
    ticker = request.form.get("ticker", "").strip().upper()
    category = request.form.get("category", "Equity").strip()
    notes = request.form.get("notes", "").strip()
    if not name:
        flash("Fund name is required.", "error")
        return redirect(url_for("fund_research"))
    conn = db.get_db()
    conn.execute(
        "INSERT INTO fund_watchlist (fund_name, ticker, category, notes, added_by) VALUES (?,?,?,?,?)",
        (name, ticker, category, notes, _current_user()["id"]),
    )
    conn.commit()
    db.put_db(conn)
    flash(f'"{name}" added to watchlist.', "success")
    return redirect(url_for("fund_research"))


@app.route("/fund-research/<int:fund_id>/delete", methods=["POST"])
@login_required
def fund_research_delete(fund_id):
    conn = db.get_db()
    conn.execute("UPDATE fund_watchlist SET deleted_at=NOW() WHERE id=?", (fund_id,))
    conn.commit()
    db.put_db(conn)
    flash("Fund removed.", "success")
    return redirect(url_for("fund_research"))


@app.route("/fund-research/<int:fund_id>/edit", methods=["POST"])
@login_required
def fund_research_edit(fund_id):
    name = request.form.get("fund_name", "").strip()
    ticker = request.form.get("ticker", "").strip().upper()
    category = request.form.get("category", "Equity").strip()
    notes = request.form.get("notes", "").strip()
    if not name:
        flash("Fund name is required.", "error")
        return redirect(url_for("fund_research"))
    conn = db.get_db()
    conn.execute(
        "UPDATE fund_watchlist SET fund_name=?, ticker=?, category=?, notes=? WHERE id=? AND deleted_at IS NULL",
        (name, ticker, category, notes, fund_id),
    )
    conn.commit()
    db.put_db(conn)
    flash(f'"{name}" updated.', "success")
    return redirect(url_for("fund_research"))


@app.route("/fund-research/<int:fund_id>/report")
@login_required
def fund_research_report(fund_id):
    conn = db.get_db()
    fund = conn.execute("SELECT * FROM fund_watchlist WHERE id=? AND deleted_at IS NULL", (fund_id,)).fetchone()
    db.put_db(conn)
    if not fund:
        flash("Fund not found.", "error")
        return redirect(url_for("fund_research"))
    report = fr.get_latest_report(fund_id)
    return render_template("fund_research_report.html",
                           user=_current_user(),
                           fund=dict(fund),
                           report=report,
                           fund_name=os.getenv("FUND_NAME", "DDQ Platform"))


import threading as _threading

# In-memory job store: job_id -> {"status": "running"|"done"|"error", "error": str}
_fr_jobs = {}

@app.route("/api/fund-research/<int:fund_id>/generate", methods=["POST"])
@login_required
def api_fund_research_generate(fund_id):
    import uuid
    job_id = str(uuid.uuid4())
    _fr_jobs[job_id] = {"status": "running", "thoughts": []}

    def _log(phase, text):
        entry = {"phase": phase, "text": text, "ts": __import__("time").time()}
        _fr_jobs[job_id]["thoughts"].append(entry)

    def _run():
        try:
            fr.generate_report(fund_id, log=_log)
            # Persist thoughts into the saved report so the page can show them
            fr.save_thoughts(fund_id, _fr_jobs[job_id]["thoughts"])
            _fr_jobs[job_id]["status"] = "done"
        except Exception as e:
            import traceback
            app.logger.error("Fund research error: %s\n%s", e, traceback.format_exc())
            _fr_jobs[job_id]["status"] = "error"
            _fr_jobs[job_id]["error"] = str(e)

    t = _threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/fund-research/job/<job_id>", methods=["GET"])
@login_required
def api_fund_research_job(job_id):
    job = _fr_jobs.get(job_id)
    if not job:
        return jsonify({"status": "unknown"})
    offset = request.args.get("offset", 0, type=int)
    thoughts = job.get("thoughts", [])
    return jsonify({
        "status": job["status"],
        "error": job.get("error"),
        "thoughts": thoughts[offset:],
        "total": len(thoughts),
    })


# ── Company Research ──────────────────────────────────────────────────────────
import company_research as cr
_cr_jobs: dict = {}

@app.route("/company-research")
@login_required
def company_research():
    conn = db.get_db()
    rows = conn.execute(
        "SELECT cw.*, crr.report_json, crr.generated_at AS report_date "
        "FROM company_watchlist cw "
        "LEFT JOIN company_research_reports crr ON crr.company_id=cw.id "
        "WHERE cw.deleted_at IS NULL ORDER BY cw.created_at DESC"
    ).fetchall()
    db.put_db(conn)
    companies = []
    for row in rows:
        c = dict(row)
        rj = {}
        if c.get("report_json"):
            try: rj = json.loads(c["report_json"])
            except Exception: pass
        cs = rj.get("company_score", {})
        c["overall_score"]       = cs.get("overall_score", "")
        c["investment_verdict"]  = cs.get("investment_verdict", "")
        c["overall_risk"]        = rj.get("risk_assessment", {}).get("overall_risk", "")
        c["sentiment_label"]     = rj.get("social_sentiment", {}).get("overall_sentiment", "")
        companies.append(c)
    u = _current_user()
    return render_template("company_research.html",
                           user=u, companies=companies,
                           is_admin=(u.get("role")=="admin"),
                           fund_name=os.getenv("FUND_NAME","DDQ Platform"))

@app.route("/company-research/add", methods=["POST"])
@login_required
def company_research_add():
    name    = request.form.get("company_name","").strip()
    ticker  = request.form.get("ticker","").strip().upper()
    sector  = request.form.get("sector","").strip()
    country = request.form.get("country","").strip()
    notes   = request.form.get("notes","").strip()
    if not name:
        flash("Company name is required.", "error")
        return redirect(url_for("company_research"))
    conn = db.get_db()
    conn.execute(
        "INSERT INTO company_watchlist (company_name, ticker, sector, country, notes, created_by) VALUES (%s,%s,%s,%s,%s,%s)",
        (name, ticker, sector, country, notes, _current_user().get("id"))
    )
    conn.commit()
    db.put_db(conn)
    flash(f"{name} added to watchlist.", "success")
    return redirect(url_for("company_research"))

@app.route("/company-research/<int:cid>/delete", methods=["POST"])
@login_required
def company_research_delete(cid):
    conn = db.get_db()
    conn.execute("UPDATE company_watchlist SET deleted_at=NOW() WHERE id=%s", (cid,))
    conn.commit()
    db.put_db(conn)
    return redirect(url_for("company_research"))

@app.route("/company-research/<int:cid>/edit", methods=["POST"])
@login_required
def company_research_edit(cid):
    conn = db.get_db()
    conn.execute(
        "UPDATE company_watchlist SET company_name=%s, ticker=%s, sector=%s, country=%s, notes=%s WHERE id=%s",
        (request.form.get("company_name",""), request.form.get("ticker","").upper(),
         request.form.get("sector",""), request.form.get("country",""),
         request.form.get("notes",""), cid)
    )
    conn.commit()
    db.put_db(conn)
    return redirect(url_for("company_research"))

@app.route("/company-research/<int:cid>/report")
@login_required
def company_research_report(cid):
    conn = db.get_db()
    co = conn.execute("SELECT * FROM company_watchlist WHERE id=%s AND deleted_at IS NULL", (cid,)).fetchone()
    db.put_db(conn)
    if not co:
        flash("Company not found.", "error")
        return redirect(url_for("company_research"))
    report = cr.get_latest_report(cid)
    return render_template("company_research_report.html",
                           user=_current_user(), company=dict(co),
                           report=report,
                           fund_name=os.getenv("FUND_NAME","DDQ Platform"))

@app.route("/api/company-research/<int:cid>/generate", methods=["POST"])
@login_required
def api_company_research_generate(cid):
    job_id = f"cr_{cid}_{int(time.time())}"
    _cr_jobs[job_id] = {"status": "running", "thoughts": []}

    def _log(phase, text):
        _cr_jobs[job_id]["thoughts"].append({"phase": phase, "text": text, "ts": time.time()})

    def _run():
        try:
            cr.generate_report(cid, log=_log)
            cr.save_thoughts(cid, _cr_jobs[job_id]["thoughts"])
            _cr_jobs[job_id]["status"] = "done"
        except Exception as e:
            import traceback
            app.logger.error("Company research error: %s\n%s", e, traceback.format_exc())
            _cr_jobs[job_id]["status"] = "error"
            _cr_jobs[job_id]["error"] = str(e)

    _threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/company-research/job/<job_id>")
@login_required
def api_company_research_job(job_id):
    job = _cr_jobs.get(job_id)
    if not job:
        return jsonify({"status": "unknown"})
    offset = request.args.get("offset", 0, type=int)
    thoughts = job.get("thoughts", [])
    return jsonify({"status": job["status"], "error": job.get("error"),
                    "thoughts": thoughts[offset:], "total": len(thoughts)})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
