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

        -- Admin knowledge base: trained Q&A pairs
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            tags        TEXT,
            created_by  INTEGER,
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (created_by) REFERENCES users(id)
        );

        -- FTS5 index for knowledge base search
        CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
            question,
            answer,
            content='knowledge_base',
            content_rowid='id',
            tokenize='porter ascii'
        );
    """)

    # Roles table for RBAC
    c.executescript("""
        CREATE TABLE IF NOT EXISTS roles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            description TEXT,
            permissions TEXT NOT NULL DEFAULT '[]',
            is_system   INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """)

    # Seed system roles
    all_perms = '["dashboard","qa_sessions","upload_documents","delete_documents","manage_investors","manage_kb","live_support","team_management","analytics"]'
    analyst_perms = '["dashboard","qa_sessions","live_support","analytics"]'
    c.execute("INSERT OR IGNORE INTO roles (name, description, permissions, is_system) VALUES ('admin', 'Full access to all features', ?, 1)", (all_perms,))
    c.execute("INSERT OR IGNORE INTO roles (name, description, permissions, is_system) VALUES ('analyst', 'View and use core features, no admin actions', ?, 1)", (analyst_perms,))

    # LLM key management tables
    c.executescript("""
        CREATE TABLE IF NOT EXISTS llm_keys (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            provider     TEXT NOT NULL DEFAULT 'openai',
            model        TEXT NOT NULL DEFAULT 'gpt-4o',
            api_key_enc  TEXT NOT NULL,
            api_key_hint TEXT DEFAULT '',
            priority     INTEGER NOT NULL DEFAULT 10,
            is_active    INTEGER NOT NULL DEFAULT 1,
            created_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS llm_usage (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            llm_key_id    INTEGER NOT NULL,
            provider      TEXT NOT NULL,
            model         TEXT NOT NULL,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd      REAL DEFAULT 0.0,
            created_at    TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (llm_key_id) REFERENCES llm_keys(id) ON DELETE CASCADE
        );
    """)

    # Migrations for columns added after initial release
    for migration in [
        "ALTER TABLE users ADD COLUMN investor_session_id INTEGER REFERENCES investor_sessions(id)",
        "ALTER TABLE conversations ADD COLUMN status TEXT DEFAULT 'active'",
        "ALTER TABLE llm_keys ADD COLUMN api_key_hint TEXT DEFAULT ''",
        "ALTER TABLE llm_keys ADD COLUMN base_url TEXT DEFAULT ''",
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


def get_user_by_id(user_id: int) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_user_profile(user_id: int, name: str, email: str) -> bool:
    conn = get_db()
    try:
        conn.execute("UPDATE users SET name=?, email=? WHERE id=?", (name, email, user_id))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def change_user_password(user_id: int, current_password: str, new_password: str) -> bool:
    conn = get_db()
    row = conn.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    if not row or row["password_hash"] != _hash(current_password):
        conn.close()
        return False
    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (_hash(new_password), user_id))
    conn.commit()
    conn.close()
    return True


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


def list_roles() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM roles ORDER BY is_system DESC, name").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["permissions"] = json.loads(d["permissions"] or "[]")
        result.append(d)
    return result


def get_role(role_id: int) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM roles WHERE id=?", (role_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["permissions"] = json.loads(d["permissions"] or "[]")
    return d


def create_role(name: str, description: str, permissions: list) -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO roles (name, description, permissions) VALUES (?, ?, ?)",
              (name.lower().replace(" ", "_"), description, json.dumps(permissions)))
    conn.commit()
    rid = c.lastrowid
    conn.close()
    return rid


def update_role(role_id: int, name: str, description: str, permissions: list):
    conn = get_db()
    role = conn.execute("SELECT name, is_system FROM roles WHERE id=?", (role_id,)).fetchone()
    if not role:
        conn.close()
        return
    # Admin always keeps full permissions — never editable
    if role["name"] == "admin":
        conn.close()
        return
    if role["is_system"]:
        # System roles (non-admin): update permissions only
        conn.execute("UPDATE roles SET permissions=? WHERE id=?", (json.dumps(permissions), role_id))
    else:
        # Custom roles: update everything
        conn.execute("UPDATE roles SET name=?, description=?, permissions=? WHERE id=?",
                     (name.lower().replace(" ", "_"), description, json.dumps(permissions), role_id))
    conn.commit()
    conn.close()


def delete_role(role_id: int):
    conn = get_db()
    conn.execute("DELETE FROM roles WHERE id=? AND is_system=0", (role_id,))
    conn.commit()
    conn.close()


def update_user_role(user_id: int, role: str):
    conn = get_db()
    conn.execute("UPDATE users SET role=? WHERE id=? AND role != 'investor'", (role, user_id))
    conn.commit()
    conn.close()


def delete_user(user_id: int):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=? AND role != 'investor'", (user_id,))
    conn.commit()
    conn.close()


def get_investor_user_by_id(user_id: int) -> dict:
    conn = get_db()
    row = conn.execute(
        "SELECT id, email, name, role, investor_session_id, created_at FROM users WHERE id=? AND role='investor'",
        (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


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
        WHERE c.investor_session_id = ? AND (c.status IS NULL OR c.status != 'deleted')
        GROUP BY c.id
        ORDER BY c.updated_at DESC
    """, (investor_session_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def soft_delete_investor_conversation(conv_id: int, investor_session_id: int):
    conn = get_db()
    conn.execute(
        "UPDATE conversations SET status='deleted' WHERE id=? AND investor_session_id=?",
        (conv_id, investor_session_id)
    )
    conn.commit()
    conn.close()


def rename_investor_conversation(conv_id: int, investor_session_id: int, title: str):
    conn = get_db()
    conn.execute(
        "UPDATE conversations SET title=? WHERE id=? AND investor_session_id=?",
        (title.strip(), conv_id, investor_session_id)
    )
    conn.commit()
    conn.close()


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


# ── Knowledge Base ────────────────────────────────────────────────────────────

def list_kb_entries() -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT kb.*, u.name as created_by_name
        FROM knowledge_base kb
        LEFT JOIN users u ON kb.created_by = u.id
        ORDER BY kb.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_kb_entry(entry_id: int) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM knowledge_base WHERE id=?", (entry_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_kb_entry(question: str, answer: str, tags: str, created_by: int) -> int:
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO knowledge_base (question, answer, tags, created_by)
        VALUES (?, ?, ?, ?)
    """, (question.strip(), answer.strip(), tags.strip() if tags else "", created_by))
    entry_id = c.lastrowid
    # Sync FTS index
    conn.execute("INSERT INTO kb_fts(rowid, question, answer) VALUES (?, ?, ?)",
                 (entry_id, question.strip(), answer.strip()))
    conn.commit()
    conn.close()
    return entry_id


def update_kb_entry(entry_id: int, question: str, answer: str, tags: str):
    conn = get_db()
    conn.execute("""
        UPDATE knowledge_base SET question=?, answer=?, tags=?, updated_at=datetime('now')
        WHERE id=?
    """, (question.strip(), answer.strip(), tags.strip() if tags else "", entry_id))
    # Rebuild FTS row
    conn.execute("DELETE FROM kb_fts WHERE rowid=?", (entry_id,))
    conn.execute("INSERT INTO kb_fts(rowid, question, answer) VALUES (?, ?, ?)",
                 (entry_id, question.strip(), answer.strip()))
    conn.commit()
    conn.close()


def delete_kb_entry(entry_id: int):
    conn = get_db()
    conn.execute("DELETE FROM kb_fts WHERE rowid=?", (entry_id,))
    conn.execute("DELETE FROM knowledge_base WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()


_KB_STOP = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "of","in","on","at","to","for","with","by","from","about","and","or","not",
    "what","how","when","where","who","which","why","can","your","our","my","tell",
    "please","give","me","us","you","its","this","that","these","those","also",
}


def _kb_terms(query: str) -> list[str]:
    """Extract significant terms from a query for KB matching."""
    words = [w.strip('.,?!;:"()[]') for w in query.lower().split()]
    return [w for w in words if len(w) > 2 and w not in _KB_STOP]


def _run_kb_fts(conn, fts_q: str, limit: int) -> list:
    try:
        return conn.execute("""
            SELECT kb.id, kb.question, kb.answer, kb.tags,
                   bm25(kb_fts) as score
            FROM kb_fts
            JOIN knowledge_base kb ON kb_fts.rowid = kb.id
            WHERE kb_fts MATCH ?
            ORDER BY bm25(kb_fts)
            LIMIT ?
        """, (fts_q, limit)).fetchall()
    except Exception:
        return []


def search_kb(query: str, limit: int = 3) -> list[dict]:
    """Search knowledge base. Tries AND query first, falls back to OR with score filter."""
    conn = get_db()
    terms = _kb_terms(query)

    rows = []
    if terms:
        # 1. Try strict AND — all keywords must appear in the KB entry
        and_q = " AND ".join(terms)
        rows = _run_kb_fts(conn, and_q, limit)

    if not rows and terms:
        # 2. Fallback: OR query, but only accept results with a decent BM25 score.
        #    Score threshold -0.5: at least a few terms must genuinely overlap.
        or_q = " OR ".join(terms)
        candidates = _run_kb_fts(conn, or_q, limit)
        rows = [r for r in candidates if r["score"] < -0.5]

    conn.close()
    return [dict(r) for r in rows]


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


# ── LLM Key Management ────────────────────────────────────────────────────────

COST_RATES = {
    # OpenAI
    "gpt-4o":                       {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":                  {"input": 0.15,   "output": 0.60},
    "gpt-4":                        {"input": 30.00,  "output": 60.00},
    "gpt-4-turbo":                  {"input": 10.00,  "output": 30.00},
    "gpt-3.5-turbo":                {"input": 0.50,   "output": 1.50},
    # Anthropic
    "claude-opus-4-6":              {"input": 15.00,  "output": 75.00},
    "claude-sonnet-4-6":            {"input": 3.00,   "output": 15.00},
    "claude-haiku-4-5-20251001":    {"input": 0.80,   "output": 4.00},
    "claude-3-5-sonnet-20241022":   {"input": 3.00,   "output": 15.00},
    "claude-3-5-haiku-20241022":    {"input": 0.80,   "output": 4.00},
    "claude-3-haiku-20240307":      {"input": 0.25,   "output": 1.25},
    "claude-3-opus-20240229":       {"input": 15.00,  "output": 75.00},
    # Google Gemini
    "gemini-2.0-flash":             {"input": 0.10,   "output": 0.40},
    "gemini-1.5-pro":               {"input": 1.25,   "output": 5.00},
    "gemini-1.5-flash":             {"input": 0.075,  "output": 0.30},
    # Groq (free tier / hosted)
    "llama-3.3-70b-versatile":      {"input": 0.59,   "output": 0.79},
    "llama-3.1-8b-instant":         {"input": 0.05,   "output": 0.08},
    "mixtral-8x7b-32768":           {"input": 0.27,   "output": 0.27},
    # Mistral
    "mistral-large-latest":         {"input": 2.00,   "output": 6.00},
    "mistral-small-latest":         {"input": 0.20,   "output": 0.60},
    "open-mistral-nemo":            {"input": 0.15,   "output": 0.15},
    # Ollama — local, no cost
    "llama3.2":                     {"input": 0.0,    "output": 0.0},
    "llama3.1":                     {"input": 0.0,    "output": 0.0},
    "mistral":                      {"input": 0.0,    "output": 0.0},
    "gemma2":                       {"input": 0.0,    "output": 0.0},
    "phi4":                         {"input": 0.0,    "output": 0.0},
    "qwen2.5":                      {"input": 0.0,    "output": 0.0},
}


def list_llm_keys() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM llm_keys ORDER BY priority ASC, id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_active_llm_keys() -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM llm_keys WHERE is_active=1 ORDER BY priority ASC, id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_llm_key(key_id: int) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM llm_keys WHERE id=?", (key_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_llm_key(name: str, provider: str, model: str, api_key_enc: str, hint: str,
                priority: int = None, base_url: str = "") -> int:
    conn = get_db()
    c = conn.cursor()
    if priority is None:
        row = conn.execute("SELECT MAX(priority) FROM llm_keys").fetchone()
        priority = (row[0] or 0) + 1
    c.execute(
        "INSERT INTO llm_keys (name, provider, model, api_key_enc, api_key_hint, priority, base_url) VALUES (?,?,?,?,?,?,?)",
        (name, provider, model, api_key_enc, hint, priority, base_url or "")
    )
    conn.commit()
    rid = c.lastrowid
    conn.close()
    return rid


def update_llm_key(key_id: int, **kwargs):
    if not kwargs:
        return
    allowed = {"name", "model", "priority", "is_active", "api_key_enc", "api_key_hint", "base_url"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [key_id]
    conn = get_db()
    conn.execute(f"UPDATE llm_keys SET {set_clause} WHERE id=?", values)
    conn.commit()
    conn.close()


def delete_llm_key(key_id: int):
    conn = get_db()
    conn.execute("DELETE FROM llm_keys WHERE id=?", (key_id,))
    conn.commit()
    conn.close()


def move_llm_key(key_id: int, direction: str):
    conn = get_db()
    row = conn.execute("SELECT priority FROM llm_keys WHERE id=?", (key_id,)).fetchone()
    if not row:
        conn.close()
        return
    cur = row["priority"]
    if direction == "up":
        neighbor = conn.execute(
            "SELECT id, priority FROM llm_keys WHERE priority < ? ORDER BY priority DESC LIMIT 1",
            (cur,)
        ).fetchone()
    else:
        neighbor = conn.execute(
            "SELECT id, priority FROM llm_keys WHERE priority > ? ORDER BY priority ASC LIMIT 1",
            (cur,)
        ).fetchone()
    if neighbor:
        conn.execute("UPDATE llm_keys SET priority=? WHERE id=?", (neighbor["priority"], key_id))
        conn.execute("UPDATE llm_keys SET priority=? WHERE id=?", (cur, neighbor["id"]))
        conn.commit()
    conn.close()


def log_llm_usage(key_id: int, provider: str, model: str, input_tokens: int, output_tokens: int):
    rates = COST_RATES.get(model, {"input": 0, "output": 0})
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    conn = get_db()
    conn.execute(
        "INSERT INTO llm_usage (llm_key_id, provider, model, input_tokens, output_tokens, cost_usd) VALUES (?,?,?,?,?,?)",
        (key_id, provider, model, input_tokens, output_tokens, cost)
    )
    conn.commit()
    conn.close()


def get_llm_usage_stats() -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT
            k.id, k.name, k.provider, k.model, k.is_active, k.priority, k.api_key_hint,
            COALESCE(SUM(u.input_tokens), 0)  AS total_input,
            COALESCE(SUM(u.output_tokens), 0) AS total_output,
            COALESCE(SUM(u.cost_usd), 0)      AS total_cost,
            COUNT(u.id)                        AS total_calls,
            MAX(u.created_at)                  AS last_used
        FROM llm_keys k
        LEFT JOIN llm_usage u ON u.llm_key_id = k.id
        GROUP BY k.id
        ORDER BY k.priority ASC, k.id ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
