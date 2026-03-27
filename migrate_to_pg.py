"""
One-time migration: SQLite (ddq_platform.db) → PostgreSQL (ddqplatform)
Run on the server: python3 migrate_to_pg.py
"""
import sqlite3
import psycopg2
import psycopg2.extras
import json
import os
import sys

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "ddq_platform.db")
PG_URL = os.getenv("DATABASE_URL", "postgresql://ddquser:ddq_secure_2026@localhost/ddqplatform")

def migrate():
    if not os.path.exists(SQLITE_PATH):
        print(f"SQLite DB not found at {SQLITE_PATH}")
        sys.exit(1)

    sq = sqlite3.connect(SQLITE_PATH)
    sq.row_factory = sqlite3.Row
    pg = psycopg2.connect(PG_URL)
    pg.autocommit = False
    cur = pg.cursor()

    print("Initialising PostgreSQL schema...")
    # Import and run init_db from the new database.py
    sys.path.insert(0, os.path.dirname(__file__))
    import database as db
    db.init_db()
    print("Schema ready.")

    def migrate_table(table, sq_sql, pg_sql, transform=None):
        try:
            rows = sq.execute(sq_sql).fetchall()
        except Exception as e:
            print(f"  {table}: skipped (not in SQLite: {e})")
            return
        if not rows:
            print(f"  {table}: 0 rows (skipped)")
            return
        count = 0
        for r in rows:
            d = dict(r)
            if transform:
                d = transform(d)
            if d is None:
                continue
            try:
                cur.execute(pg_sql, d)
                count += 1
            except Exception as e:
                pg.rollback()
                print(f"  WARN {table} row {d.get('id')}: {e}")
                continue
        pg.commit()
        print(f"  {table}: {count} rows migrated")

    print("\nMigrating investor_sessions...")
    migrate_table(
        "investor_sessions",
        "SELECT * FROM investor_sessions",
        """INSERT INTO investor_sessions (id, investor_name, investor_entity, created_by, notes, created_at, profile_text, profile_generated_at)
           VALUES (%(id)s, %(investor_name)s, %(investor_entity)s, %(created_by)s, %(notes)s, %(created_at)s, %(profile_text)s, %(profile_generated_at)s)
           ON CONFLICT (id) DO NOTHING""",
        lambda d: {**d, "profile_text": d.get("profile_text") or "", "profile_generated_at": d.get("profile_generated_at") or None}
    )

    print("Migrating users...")
    migrate_table(
        "users",
        "SELECT * FROM users",
        """INSERT INTO users (id, email, name, password_hash, role, investor_session_id, created_at)
           VALUES (%(id)s, %(email)s, %(name)s, %(password_hash)s, %(role)s, %(investor_session_id)s, %(created_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating fund_documents...")
    migrate_table(
        "fund_documents",
        "SELECT * FROM fund_documents",
        """INSERT INTO fund_documents (id, name, doc_type, filepath, content, status, uploaded_by, uploaded_at)
           VALUES (%(id)s, %(name)s, %(doc_type)s, %(filepath)s, %(content)s, %(status)s, %(uploaded_by)s, %(uploaded_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating document_chunks...")
    migrate_table(
        "document_chunks",
        "SELECT * FROM document_chunks",
        """INSERT INTO document_chunks (id, document_id, chunk_text, chunk_index, page_ref, section_ref)
           VALUES (%(id)s, %(document_id)s, %(chunk_text)s, %(chunk_index)s, %(page_ref)s, %(section_ref)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating investor_documents...")
    migrate_table(
        "investor_documents",
        "SELECT * FROM investor_documents",
        """INSERT INTO investor_documents (id, investor_session_id, name, filepath, doc_type, content, uploaded_at)
           VALUES (%(id)s, %(investor_session_id)s, %(name)s, %(filepath)s, %(doc_type)s, %(content)s, %(uploaded_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating investor_doc_chunks...")
    migrate_table(
        "investor_doc_chunks",
        "SELECT * FROM investor_doc_chunks",
        """INSERT INTO investor_doc_chunks (id, investor_doc_id, chunk_text, chunk_index)
           VALUES (%(id)s, %(investor_doc_id)s, %(chunk_text)s, %(chunk_index)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating roles...")
    migrate_table(
        "roles",
        "SELECT * FROM roles",
        """INSERT INTO roles (id, name, description, permissions, is_system, created_at)
           VALUES (%(id)s, %(name)s, %(description)s, %(permissions)s, %(is_system)s, %(created_at)s)
           ON CONFLICT (name) DO NOTHING""",
    )

    print("Migrating llm_keys...")
    migrate_table(
        "llm_keys",
        "SELECT * FROM llm_keys",
        """INSERT INTO llm_keys (id, name, provider, model, api_key_enc, api_key_hint, priority, is_active, base_url, created_at)
           VALUES (%(id)s, %(name)s, %(provider)s, %(model)s, %(api_key_enc)s, %(api_key_hint)s, %(priority)s, %(is_active)s, %(base_url)s, %(created_at)s)
           ON CONFLICT (id) DO NOTHING""",
        lambda d: {**d, "base_url": d.get("base_url") or ""}
    )

    print("Migrating conversations...")
    migrate_table(
        "conversations",
        "SELECT * FROM conversations",
        """INSERT INTO conversations (id, title, investor_session_id, created_by, created_at, updated_at, status)
           VALUES (%(id)s, %(title)s, %(investor_session_id)s, %(created_by)s, %(created_at)s, %(updated_at)s, %(status)s)
           ON CONFLICT (id) DO NOTHING""",
        lambda d: {**d, "status": d.get("status") or "active"}
    )

    print("Migrating messages...")
    migrate_table(
        "messages",
        "SELECT * FROM messages",
        """INSERT INTO messages (id, conversation_id, role, content, sources, draft_response, themes, created_at)
           VALUES (%(id)s, %(conversation_id)s, %(role)s, %(content)s, %(sources)s, %(draft_response)s, %(themes)s, %(created_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating question_themes...")
    migrate_table(
        "question_themes",
        "SELECT * FROM question_themes",
        """INSERT INTO question_themes (id, message_id, theme, created_at)
           VALUES (%(id)s, %(message_id)s, %(theme)s, %(created_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating document_assignments...")
    migrate_table(
        "document_assignments",
        "SELECT * FROM document_assignments",
        """INSERT INTO document_assignments (id, investor_session_id, document_id, assigned_by, assigned_at)
           VALUES (%(id)s, %(investor_session_id)s, %(document_id)s, %(assigned_by)s, %(assigned_at)s)
           ON CONFLICT (investor_session_id, document_id) DO NOTHING""",
    )

    print("Migrating handover_requests...")
    migrate_table(
        "handover_requests",
        "SELECT * FROM handover_requests",
        """INSERT INTO handover_requests (id, conversation_id, investor_session_id, reason, status, requested_at, claimed_by, claimed_at, resolved_at)
           VALUES (%(id)s, %(conversation_id)s, %(investor_session_id)s, %(reason)s, %(status)s, %(requested_at)s, %(claimed_by)s, %(claimed_at)s, %(resolved_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating knowledge_base...")
    migrate_table(
        "knowledge_base",
        "SELECT * FROM knowledge_base",
        """INSERT INTO knowledge_base (id, question, answer, tags, created_by, created_at, updated_at)
           VALUES (%(id)s, %(question)s, %(answer)s, %(tags)s, %(created_by)s, %(created_at)s, %(updated_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating session_answers...")
    migrate_table(
        "session_answers",
        "SELECT * FROM session_answers",
        """INSERT INTO session_answers (id, investor_session_id, question, answer, created_at)
           VALUES (%(id)s, %(investor_session_id)s, %(question)s, %(answer)s, %(created_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    print("Migrating llm_usage...")
    migrate_table(
        "llm_usage",
        "SELECT * FROM llm_usage",
        """INSERT INTO llm_usage (id, llm_key_id, provider, model, input_tokens, output_tokens, cost_usd, created_at)
           VALUES (%(id)s, %(llm_key_id)s, %(provider)s, %(model)s, %(input_tokens)s, %(output_tokens)s, %(cost_usd)s, %(created_at)s)
           ON CONFLICT (id) DO NOTHING""",
    )

    # Reset sequences so new inserts get correct IDs
    print("\nResetting sequences...")
    sequences = [
        ("users", "users_id_seq"),
        ("investor_sessions", "investor_sessions_id_seq"),
        ("fund_documents", "fund_documents_id_seq"),
        ("document_chunks", "document_chunks_id_seq"),
        ("investor_documents", "investor_documents_id_seq"),
        ("investor_doc_chunks", "investor_doc_chunks_id_seq"),
        ("roles", "roles_id_seq"),
        ("llm_keys", "llm_keys_id_seq"),
        ("conversations", "conversations_id_seq"),
        ("messages", "messages_id_seq"),
        ("question_themes", "question_themes_id_seq"),
        ("document_assignments", "document_assignments_id_seq"),
        ("handover_requests", "handover_requests_id_seq"),
        ("knowledge_base", "knowledge_base_id_seq"),
        ("session_answers", "session_answers_id_seq"),
        ("llm_usage", "llm_usage_id_seq"),
    ]
    for table, seq in sequences:
        try:
            cur.execute(f"SELECT setval('{seq}', COALESCE((SELECT MAX(id) FROM {table}), 1))")
            pg.commit()
            print(f"  {seq} reset")
        except Exception as e:
            pg.rollback()
            print(f"  WARN {seq}: {e}")

    sq.close()
    pg.close()
    print("\nMigration complete!")

if __name__ == "__main__":
    migrate()
