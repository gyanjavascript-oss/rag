"""
SQLite database schema and query helpers for the DDQ Platform.
"""
import sqlite3
import hashlib
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "ddq_platform.db")


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
        -- Users (internal team members)
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT UNIQUE NOT NULL,
            name          TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'analyst',
            created_at    TEXT DEFAULT (datetime('now'))
        );

        -- Fund documents (LPA, presentations, policies, memos, etc.)
        CREATE TABLE IF NOT EXISTS fund_documents (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            doc_type      TEXT NOT NULL DEFAULT 'other',
            filepath      TEXT,
            content       TEXT,
            status        TEXT DEFAULT 'active',
            uploaded_by   INTEGER,
            uploaded_at   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (uploaded_by) REFERENCES users(id)
        );

        -- Document chunks for retrieval
        CREATE TABLE IF NOT EXISTS document_chunks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id   INTEGER NOT NULL,
            chunk_text    TEXT NOT NULL,
            chunk_index   INTEGER NOT NULL,
            page_ref      TEXT,
            section_ref   TEXT,
            FOREIGN KEY (document_id) REFERENCES fund_documents(id) ON DELETE CASCADE
        );

        -- FTS index over document chunks
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_text,
            content='document_chunks',
            content_rowid='id',
            tokenize='porter ascii'
        );

        -- Investor sessions (one per investor/prospect)
        CREATE TABLE IF NOT EXISTS investor_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            investor_name   TEXT NOT NULL,
            investor_entity TEXT,
            created_by      INTEGER,
            notes           TEXT,
            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (created_by) REFERENCES users(id)
        );

        -- Investor-specific uploaded documents
        CREATE TABLE IF NOT EXISTS investor_documents (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            investor_session_id INTEGER NOT NULL,
            name                TEXT NOT NULL,
            filepath            TEXT,
            doc_type            TEXT DEFAULT 'other',
            content             TEXT,
            uploaded_at         TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id) ON DELETE CASCADE
        );

        -- Investor document chunks
        CREATE TABLE IF NOT EXISTS investor_doc_chunks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            investor_doc_id INTEGER NOT NULL,
            chunk_text      TEXT NOT NULL,
            chunk_index     INTEGER NOT NULL,
            FOREIGN KEY (investor_doc_id) REFERENCES investor_documents(id) ON DELETE CASCADE
        );

        -- FTS index for investor doc chunks
        CREATE VIRTUAL TABLE IF NOT EXISTS inv_chunks_fts USING fts5(
            chunk_text,
            content='investor_doc_chunks',
            content_rowid='id',
            tokenize='porter ascii'
        );

        -- Conversations (a thread of Q&A, optionally linked to an investor session)
        CREATE TABLE IF NOT EXISTS conversations (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            title               TEXT,
            investor_session_id INTEGER,
            created_by          INTEGER,
            created_at          TEXT DEFAULT (datetime('now')),
            updated_at          TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id),
            FOREIGN KEY (created_by) REFERENCES users(id)
        );

        -- Individual messages in a conversation
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            sources         TEXT,
            draft_response  TEXT,
            themes          TEXT,
            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        -- Extracted themes for analytics
        CREATE TABLE IF NOT EXISTS question_themes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id  INTEGER NOT NULL,
            theme       TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
        );

        -- Human agent handover requests
        CREATE TABLE IF NOT EXISTS handover_requests (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id     INTEGER NOT NULL,
            investor_session_id INTEGER,
            reason              TEXT,
            status              TEXT DEFAULT 'pending',
            requested_at        TEXT DEFAULT (datetime('now')),
            claimed_by          INTEGER,
            claimed_at          TEXT,
            resolved_at         TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id),
            FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id),
            FOREIGN KEY (claimed_by) REFERENCES users(id)
        );

        -- Fund documents assigned to investor sessions (investor portal access control)
        CREATE TABLE IF NOT EXISTS document_assignments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            investor_session_id INTEGER NOT NULL,
            document_id         INTEGER NOT NULL,
            assigned_by         INTEGER,
            assigned_at         TEXT DEFAULT (datetime('now')),
            UNIQUE(investor_session_id, document_id),
            FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES fund_documents(id) ON DELETE CASCADE,
            FOREIGN KEY (assigned_by) REFERENCES users(id)
        );
    """)

    # Migrations for columns added after initial release
    for migration in [
        "ALTER TABLE users ADD COLUMN investor_session_id INTEGER REFERENCES investor_sessions(id)",
        "ALTER TABLE conversations ADD COLUMN status TEXT DEFAULT 'active'",
    ]:
        try:
            c.execute(migration)
        except Exception:
            pass

    # Seed default admin user
    admin_hash = _hash("admin123")
    c.execute("""
        INSERT OR IGNORE INTO users (email, name, password_hash, role)
        VALUES (?, ?, ?, 'admin')
    """, ("admin@fund.com", "Administrator", admin_hash))

    conn.commit()
    conn.close()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def get_user_by_email(email: str) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_login(email: str, password: str) -> dict:
    user = get_user_by_email(email)
    if user and user["password_hash"] == _hash(password):
        return user
    return None


def create_user(email: str, name: str, password: str, role: str = "analyst") -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO users (email, name, password_hash, role)
        VALUES (?, ?, ?, ?)
    """, (email, name, _hash(password), role))
    conn.commit()
    uid = c.lastrowid
    conn.close()
    return uid


# ── Documents ─────────────────────────────────────────────────────────────────

def list_fund_documents() -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT d.*, u.name as uploaded_by_name
        FROM fund_documents d
        LEFT JOIN users u ON d.uploaded_by = u.id
        WHERE d.status = 'active'
        ORDER BY d.uploaded_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_fund_document(doc_id: int) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM fund_documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_fund_document(doc_id: int):
    conn = get_db()
    conn.execute("UPDATE fund_documents SET status='deleted' WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()


def _fts_query(query: str) -> str:
    """Build an FTS5 OR query from a natural language string."""
    stop = {"the","a","an","is","are","was","were","be","been","being","have","has",
            "had","do","does","did","will","would","could","should","may","might",
            "of","in","on","at","to","for","with","by","from","about","and","or","not"}
    words = [w.strip('.,?!;:"()') for w in query.lower().split()]
    terms = [w for w in words if len(w) > 2 and w not in stop]
    if not terms:
        terms = [query.replace('"', '""')]
    return " OR ".join(terms)


def search_fund_documents(query: str, limit: int = 6) -> list[dict]:
    conn = get_db()
    try:
        fts_q = _fts_query(query)
        rows = conn.execute("""
            SELECT dc.chunk_text, dc.page_ref, dc.section_ref,
                   d.name as doc_name, d.doc_type, d.id as document_id,
                   bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN document_chunks dc ON chunks_fts.rowid = dc.id
            JOIN fund_documents d ON dc.document_id = d.id
            WHERE chunks_fts MATCH ? AND d.status = 'active'
            ORDER BY bm25(chunks_fts)
            LIMIT ?
        """, (fts_q, limit)).fetchall()
    except Exception:
        rows = conn.execute("""
            SELECT dc.chunk_text, dc.page_ref, dc.section_ref,
                   d.name as doc_name, d.doc_type, d.id as document_id,
                   0 as score
            FROM document_chunks dc
            JOIN fund_documents d ON dc.document_id = d.id
            WHERE dc.chunk_text LIKE ? AND d.status = 'active'
            LIMIT ?
        """, (f"%{query}%", limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_investor_documents(query: str, investor_session_id: int, limit: int = 4) -> list[dict]:
    conn = get_db()
    try:
        fts_q = _fts_query(query)
        rows = conn.execute("""
            SELECT idc.chunk_text, id.name as doc_name, id.doc_type,
                   bm25(inv_chunks_fts) as score
            FROM inv_chunks_fts
            JOIN investor_doc_chunks idc ON inv_chunks_fts.rowid = idc.id
            JOIN investor_documents id ON idc.investor_doc_id = id.id
            WHERE inv_chunks_fts MATCH ? AND id.investor_session_id = ?
            ORDER BY bm25(inv_chunks_fts)
            LIMIT ?
        """, (fts_q, investor_session_id, limit)).fetchall()
    except Exception:
        rows = conn.execute("""
            SELECT idc.chunk_text, id.name as doc_name, id.doc_type, 0 as score
            FROM investor_doc_chunks idc
            JOIN investor_documents id ON idc.investor_doc_id = id.id
            WHERE idc.chunk_text LIKE ? AND id.investor_session_id = ?
            LIMIT ?
        """, (f"%{query}%", investor_session_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def find_similar_questions(question: str, limit: int = 3) -> list[dict]:
    """Return previously answered similar questions for context."""
    words = [w for w in question.lower().split() if len(w) > 4][:5]
    if not words:
        return []
    conn = get_db()
    results = []
    for word in words:
        rows = conn.execute("""
            SELECT m.content as question, m2.content as answer,
                   m.created_at, c.id as conversation_id
            FROM messages m
            JOIN messages m2 ON m2.conversation_id = m.conversation_id
                             AND m2.role = 'assistant'
                             AND m2.id = (
                                SELECT id FROM messages
                                WHERE conversation_id = m.conversation_id
                                AND role = 'assistant'
                                AND id > m.id
                                LIMIT 1
                             )
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.role = 'user' AND m.content LIKE ?
            LIMIT 2
        """, (f"%{word}%",)).fetchall()
        for r in rows:
            d = dict(r)
            if not any(x["question"] == d["question"] for x in results):
                results.append(d)
    conn.close()
    return results[:limit]


# ── Investor Sessions ─────────────────────────────────────────────────────────

def list_investor_sessions() -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT s.*, u.name as created_by_name,
               COUNT(DISTINCT id.id) as doc_count,
               COUNT(DISTINCT c.id) as conversation_count
        FROM investor_sessions s
        LEFT JOIN users u ON s.created_by = u.id
        LEFT JOIN investor_documents id ON id.investor_session_id = s.id
        LEFT JOIN conversations c ON c.investor_session_id = s.id
        GROUP BY s.id
        ORDER BY s.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_investor_session(session_id: int) -> dict:
    conn = get_db()
    row = conn.execute("""
        SELECT s.*, u.name as created_by_name
        FROM investor_sessions s
        LEFT JOIN users u ON s.created_by = u.id
        WHERE s.id = ?
    """, (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_investor_session(investor_name: str, investor_entity: str,
                             notes: str, created_by: int) -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO investor_sessions (investor_name, investor_entity, notes, created_by)
        VALUES (?, ?, ?, ?)
    """, (investor_name, investor_entity, notes, created_by))
    conn.commit()
    sid = c.lastrowid
    conn.close()
    return sid


# ── Conversations & Messages ──────────────────────────────────────────────────

def create_conversation(created_by: int, investor_session_id: int = None,
                         title: str = None) -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO conversations (title, investor_session_id, created_by)
        VALUES (?, ?, ?)
    """, (title, investor_session_id, created_by))
    conn.commit()
    cid = c.lastrowid
    conn.close()
    return cid


def get_conversation(conv_id: int) -> dict:
    conn = get_db()
    row = conn.execute("""
        SELECT c.*, u.name as created_by_name,
               s.investor_name, s.investor_entity
        FROM conversations c
        LEFT JOIN users u ON c.created_by = u.id
        LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
        WHERE c.id = ?
    """, (conv_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_conversations(user_id: int = None, limit: int = 30) -> list[dict]:
    conn = get_db()
    query = """
        SELECT c.id, c.title, c.created_at, c.updated_at,
               u.name as created_by_name,
               s.investor_name,
               COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN users u ON c.created_by = u.id
        LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
        LEFT JOIN messages m ON m.conversation_id = c.id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT ?
    """
    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_messages(conv_id: int) -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC
    """, (conv_id,)).fetchall()
    conn.close()
    msgs = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else []
        d["themes"] = json.loads(d["themes"]) if d["themes"] else []
        msgs.append(d)
    return msgs


def add_message(conv_id: int, role: str, content: str,
                sources: list = None, draft_response: str = None,
                themes: list = None) -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO messages (conversation_id, role, content, sources, draft_response, themes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (conv_id, role, content,
          json.dumps(sources or []),
          draft_response,
          json.dumps(themes or [])))
    conn.execute("UPDATE conversations SET updated_at=datetime('now') WHERE id=?", (conv_id,))
    msg_id = c.lastrowid

    # Store themes for analytics
    for theme in (themes or []):
        c.execute("INSERT INTO question_themes (message_id, theme) VALUES (?, ?)",
                  (msg_id, theme))

    conn.commit()
    conn.close()
    return msg_id


def update_conversation_title(conv_id: int, title: str):
    conn = get_db()
    conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, conv_id))
    conn.commit()
    conn.close()


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_dashboard_stats() -> dict:
    conn = get_db()
    c = conn.cursor()
    stats = {
        "total_questions": c.execute(
            "SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0],
        "total_conversations": c.execute(
            "SELECT COUNT(*) FROM conversations").fetchone()[0],
        "total_documents": c.execute(
            "SELECT COUNT(*) FROM fund_documents WHERE status='active'").fetchone()[0],
        "total_investors": c.execute(
            "SELECT COUNT(*) FROM investor_sessions").fetchone()[0],
        "total_chunks": c.execute(
            "SELECT COUNT(*) FROM document_chunks").fetchone()[0],
    }
    conn.close()
    return stats


def get_theme_analytics() -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT theme, COUNT(*) as count
        FROM question_themes
        GROUP BY theme
        ORDER BY count DESC
        LIMIT 20
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document_citation_stats() -> list[dict]:
    """Which documents are referenced most in answers."""
    conn = get_db()
    rows = conn.execute("""
        SELECT d.name, d.doc_type, COUNT(m.id) as citation_count
        FROM fund_documents d
        JOIN document_chunks dc ON dc.document_id = d.id
        JOIN messages m ON m.sources LIKE '%' || d.name || '%'
        WHERE d.status = 'active'
        GROUP BY d.id
        ORDER BY citation_count DESC
        LIMIT 10
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_assigned_fund_documents(query: str, investor_session_id: int, limit: int = 6) -> list[dict]:
    """Search only fund documents assigned to this investor session."""
    assigned_ids = get_assigned_document_ids(investor_session_id)
    if not assigned_ids:
        return []
    conn = get_db()
    placeholders = ",".join("?" * len(assigned_ids))
    try:
        fts_q = _fts_query(query)
        rows = conn.execute(f"""
            SELECT dc.chunk_text, dc.page_ref, dc.section_ref,
                   d.name as doc_name, d.doc_type, d.id as document_id,
                   bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN document_chunks dc ON chunks_fts.rowid = dc.id
            JOIN fund_documents d ON dc.document_id = d.id
            WHERE chunks_fts MATCH ? AND d.status = 'active' AND d.id IN ({placeholders})
            ORDER BY bm25(chunks_fts)
            LIMIT ?
        """, [fts_q] + assigned_ids + [limit]).fetchall()
    except Exception:
        rows = conn.execute(f"""
            SELECT dc.chunk_text, dc.page_ref, dc.section_ref,
                   d.name as doc_name, d.doc_type, d.id as document_id, 0 as score
            FROM document_chunks dc
            JOIN fund_documents d ON dc.document_id = d.id
            WHERE dc.chunk_text LIKE ? AND d.status = 'active' AND d.id IN ({placeholders})
            LIMIT ?
        """, [f"%{query}%"] + assigned_ids + [limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Document Assignments (investor portal) ────────────────────────────────────

def get_assigned_document_ids(investor_session_id: int) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT document_id FROM document_assignments WHERE investor_session_id=?",
        (investor_session_id,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_assigned_documents(investor_session_id: int) -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT d.id, d.name, d.doc_type, da.assigned_at
        FROM document_assignments da
        JOIN fund_documents d ON da.document_id = d.id
        WHERE da.investor_session_id = ? AND d.status = 'active'
        ORDER BY d.name
    """, (investor_session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def assign_documents_to_investor(investor_session_id: int, doc_ids: list, assigned_by: int):
    """Replace all document assignments for an investor session."""
    conn = get_db()
    conn.execute("DELETE FROM document_assignments WHERE investor_session_id=?", (investor_session_id,))
    for doc_id in doc_ids:
        try:
            conn.execute("""
                INSERT INTO document_assignments (investor_session_id, document_id, assigned_by)
                VALUES (?, ?, ?)
            """, (investor_session_id, doc_id, assigned_by))
        except Exception:
            pass
    conn.commit()
    conn.close()


# ── Investor User Accounts ────────────────────────────────────────────────────

def create_investor_user(email: str, name: str, password: str, investor_session_id: int) -> int:
    conn = get_db()
    c = conn.cursor()
    # Remove any existing investor user for this session
    conn.execute("DELETE FROM users WHERE investor_session_id=?", (investor_session_id,))
    c.execute("""
        INSERT INTO users (email, name, password_hash, role, investor_session_id)
        VALUES (?, ?, ?, 'investor', ?)
    """, (email, name, _hash(password), investor_session_id))
    conn.commit()
    uid = c.lastrowid
    conn.close()
    return uid


def get_investor_user(investor_session_id: int) -> dict:
    conn = get_db()
    row = conn.execute(
        "SELECT id, email, name, role, investor_session_id FROM users WHERE investor_session_id=?",
        (investor_session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_investor_conversations(investor_session_id: int) -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT c.id, c.title, c.created_at, c.updated_at,
               COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.investor_session_id = ?
        GROUP BY c.id
        ORDER BY c.updated_at DESC
    """, (investor_session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Human Agent Handover ──────────────────────────────────────────────────────

def create_handover_request(conversation_id: int, investor_session_id: int, reason: str) -> int:
    conn = get_db()
    c = conn.cursor()
    # Cancel any previous open request for this conversation
    conn.execute("""
        UPDATE handover_requests SET status='cancelled'
        WHERE conversation_id=? AND status='pending'
    """, (conversation_id,))
    c.execute("""
        INSERT INTO handover_requests (conversation_id, investor_session_id, reason)
        VALUES (?, ?, ?)
    """, (conversation_id, investor_session_id, reason))
    conn.execute("UPDATE conversations SET status='pending_handover' WHERE id=?", (conversation_id,))
    conn.commit()
    hid = c.lastrowid
    conn.close()
    return hid


def get_pending_handovers() -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT h.*, c.title as conv_title,
               s.investor_name, s.investor_entity
        FROM handover_requests h
        JOIN conversations c ON h.conversation_id = c.id
        LEFT JOIN investor_sessions s ON h.investor_session_id = s.id
        WHERE h.status IN ('pending', 'claimed')
        ORDER BY h.requested_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pending_handover_count() -> int:
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM handover_requests WHERE status='pending'"
    ).fetchone()[0]
    conn.close()
    return count


def get_handover_for_conversation(conversation_id: int) -> dict:
    conn = get_db()
    row = conn.execute("""
        SELECT h.*, u.name as claimed_by_name
        FROM handover_requests h
        LEFT JOIN users u ON h.claimed_by = u.id
        WHERE h.conversation_id=? AND h.status IN ('pending','claimed')
        ORDER BY h.requested_at DESC LIMIT 1
    """, (conversation_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def claim_handover(handover_id: int, user_id: int):
    conn = get_db()
    conn.execute("""
        UPDATE handover_requests SET status='claimed', claimed_by=?, claimed_at=datetime('now')
        WHERE id=?
    """, (user_id, handover_id))
    # Update conversation status
    row = conn.execute("SELECT conversation_id FROM handover_requests WHERE id=?", (handover_id,)).fetchone()
    if row:
        conn.execute("UPDATE conversations SET status='human_active' WHERE id=?", (row[0],))
    conn.commit()
    conn.close()


def resolve_handover(handover_id: int):
    conn = get_db()
    conn.execute("""
        UPDATE handover_requests SET status='resolved', resolved_at=datetime('now')
        WHERE id=?
    """, (handover_id,))
    row = conn.execute("SELECT conversation_id FROM handover_requests WHERE id=?", (handover_id,)).fetchone()
    if row:
        conn.execute("UPDATE conversations SET status='active' WHERE id=?", (row[0],))
    conn.commit()
    conn.close()


def get_messages_since(conversation_id: int, last_id: int) -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM messages WHERE conversation_id=? AND id>? ORDER BY created_at ASC
    """, (conversation_id, last_id)).fetchall()
    conn.close()
    msgs = []
    for r in rows:
        d = dict(r)
        d["sources"] = []
        d["themes"] = []
        msgs.append(d)
    return msgs


def update_conversation_status(conversation_id: int, status: str):
    conn = get_db()
    conn.execute("UPDATE conversations SET status=? WHERE id=?", (status, conversation_id))
    conn.commit()
    conn.close()


def get_recent_questions(limit: int = 20) -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT m.content as question, m.themes, m.created_at,
               c.title as conversation_title,
               s.investor_name
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
        WHERE m.role = 'user'
        ORDER BY m.created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d["themes"] = json.loads(d["themes"]) if d["themes"] else []
        results.append(d)
    return results
