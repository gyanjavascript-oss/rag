"""
PostgreSQL database schema and query helpers for the DDQ Platform.
"""
import psycopg2
import psycopg2.pool
import psycopg2.extras
import hashlib
import json
import os
import re

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ddquser:ddq_secure_2026@localhost/ddqplatform"
)

# Force PostgreSQL date/timestamp columns to return as ISO strings
# so Jinja templates can do things like value[:10] safely.
def _pg_dt_to_str(value, cur):
    return None if value is None else str(value)

_DT_TYPE = psycopg2.extensions.new_type(
    (1082, 1114, 1184),   # date, timestamp, timestamptz OIDs
    'DT_AS_STR', _pg_dt_to_str
)
psycopg2.extensions.register_type(_DT_TYPE)

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, DATABASE_URL)
    return _pool


class _ConnWrapper:
    """Thin wrapper so app.py can call conn.execute() like SQLite,
    while also passing through psycopg2 cursor() calls for database.py internals."""
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        """SQLite-compatible execute: auto-replaces ? with %s."""
        sql = sql.replace("?", "%s")
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params or ())
        return _CurWrapper(cur)

    def cursor(self, cursor_factory=None):
        """Pass through for database.py internal use.
        Default is plain cursor (for scalar queries); pass RealDictCursor explicitly for row dicts."""
        if cursor_factory:
            return self._conn.cursor(cursor_factory=cursor_factory)
        return self._conn.cursor()

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        put_db(self._conn)


class _CurWrapper:
    def __init__(self, cur):
        self._cur = cur

    def fetchone(self):
        row = self._cur.fetchone()
        return _row(dict(row)) if row else None

    def fetchall(self):
        return [_row(dict(r)) for r in self._cur.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


def get_db():
    conn = _get_pool().getconn()
    return _ConnWrapper(conn)


def put_db(conn):
    # Accept both raw psycopg2 conn and _ConnWrapper
    if isinstance(conn, _ConnWrapper):
        _get_pool().putconn(conn._conn)
    else:
        _get_pool().putconn(conn)


def _row(d: dict) -> dict:
    """Convert datetime objects to ISO strings so templates can do [:10] slicing."""
    import datetime
    return {
        k: v.isoformat() if isinstance(v, (datetime.datetime, datetime.date)) else v
        for k, v in d.items()
    }


def init_db():
    conn = get_db()
    try:
        cur = conn.cursor()

        # Users
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                name          TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'analyst',
                created_at    TIMESTAMP DEFAULT NOW()
            )
        """)

        # Fund documents
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fund_documents (
                id           SERIAL PRIMARY KEY,
                name         TEXT NOT NULL,
                doc_type     TEXT NOT NULL DEFAULT 'Other',
                ai_doc_type  TEXT DEFAULT '',
                filepath     TEXT,
                content      TEXT,
                summary      TEXT DEFAULT '',
                key_topics   TEXT DEFAULT '[]',
                page_count   INTEGER DEFAULT 0,
                search_meta  TSVECTOR,
                status       TEXT DEFAULT 'active',
                uploaded_by  INTEGER,
                uploaded_at  TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (uploaded_by) REFERENCES users(id)
            )
        """)

        # Document chunks
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id            SERIAL PRIMARY KEY,
                document_id   INTEGER NOT NULL,
                chunk_text    TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                page_ref      TEXT,
                section_ref   TEXT,
                search_vector tsvector,
                FOREIGN KEY (document_id) REFERENCES fund_documents(id) ON DELETE CASCADE
            )
        """)

        # Page summaries — AI-generated per-page summaries for hybrid search
        cur.execute("SAVEPOINT before_page_summaries")
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS page_summaries (
                    id            SERIAL PRIMARY KEY,
                    document_id   INTEGER NOT NULL,
                    page_num      INTEGER NOT NULL,
                    summary       TEXT NOT NULL,
                    search_vector TSVECTOR,
                    FOREIGN KEY (document_id) REFERENCES fund_documents(id) ON DELETE CASCADE
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_page_summaries_doc
                ON page_summaries(document_id)
            """)
            cur.execute("RELEASE SAVEPOINT before_page_summaries")
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT before_page_summaries")

        # Migrations for new fund_documents columns
        for col, defn in [
            ("ai_doc_type", "TEXT DEFAULT ''"),
            ("summary",     "TEXT DEFAULT ''"),
            ("key_topics",  "TEXT DEFAULT '[]'"),
            ("page_count",  "INTEGER DEFAULT 0"),
            ("search_meta", "TSVECTOR"),
        ]:
            cur.execute(f"""
                ALTER TABLE fund_documents ADD COLUMN IF NOT EXISTS {col} {defn}
            """)

        # Investor sessions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS investor_sessions (
                id              SERIAL PRIMARY KEY,
                investor_name   TEXT NOT NULL,
                investor_entity TEXT,
                created_by      INTEGER,
                notes           TEXT,
                created_at      TIMESTAMP DEFAULT NOW(),
                profile_text    TEXT DEFAULT '',
                profile_generated_at TEXT DEFAULT '',
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Investor documents
        cur.execute("""
            CREATE TABLE IF NOT EXISTS investor_documents (
                id                  SERIAL PRIMARY KEY,
                investor_session_id INTEGER NOT NULL,
                name                TEXT NOT NULL,
                filepath            TEXT,
                doc_type            TEXT DEFAULT 'other',
                content             TEXT,
                uploaded_at         TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id) ON DELETE CASCADE
            )
        """)

        # Investor doc chunks
        cur.execute("""
            CREATE TABLE IF NOT EXISTS investor_doc_chunks (
                id              SERIAL PRIMARY KEY,
                investor_doc_id INTEGER NOT NULL,
                chunk_text      TEXT NOT NULL,
                chunk_index     INTEGER NOT NULL,
                search_vector   tsvector,
                FOREIGN KEY (investor_doc_id) REFERENCES investor_documents(id) ON DELETE CASCADE
            )
        """)

        # Conversations
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id                  SERIAL PRIMARY KEY,
                title               TEXT,
                investor_session_id INTEGER,
                created_by          INTEGER,
                status              TEXT DEFAULT 'active',
                created_at          TIMESTAMP DEFAULT NOW(),
                updated_at          TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id),
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Messages
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                sources         TEXT,
                draft_response  TEXT,
                themes          TEXT,
                created_at      TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Question themes
        cur.execute("""
            CREATE TABLE IF NOT EXISTS question_themes (
                id         SERIAL PRIMARY KEY,
                message_id INTEGER NOT NULL,
                theme      TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
            )
        """)

        # Handover requests
        cur.execute("""
            CREATE TABLE IF NOT EXISTS handover_requests (
                id                  SERIAL PRIMARY KEY,
                conversation_id     INTEGER NOT NULL,
                investor_session_id INTEGER,
                reason              TEXT,
                status              TEXT DEFAULT 'pending',
                requested_at        TIMESTAMP DEFAULT NOW(),
                claimed_by          INTEGER,
                claimed_at          TIMESTAMP,
                resolved_at         TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id),
                FOREIGN KEY (claimed_by) REFERENCES users(id)
            )
        """)

        # Document assignments
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_assignments (
                id                  SERIAL PRIMARY KEY,
                investor_session_id INTEGER NOT NULL,
                document_id         INTEGER NOT NULL,
                assigned_by         INTEGER,
                assigned_at         TIMESTAMP DEFAULT NOW(),
                UNIQUE(investor_session_id, document_id),
                FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES fund_documents(id) ON DELETE CASCADE,
                FOREIGN KEY (assigned_by) REFERENCES users(id)
            )
        """)

        # Knowledge base
        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id            SERIAL PRIMARY KEY,
                question      TEXT NOT NULL,
                answer        TEXT NOT NULL,
                tags          TEXT,
                created_by    INTEGER,
                created_at    TIMESTAMP DEFAULT NOW(),
                updated_at    TIMESTAMP DEFAULT NOW(),
                search_vector tsvector,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Session answers
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_answers (
                id                  SERIAL PRIMARY KEY,
                investor_session_id INTEGER NOT NULL,
                question            TEXT NOT NULL,
                answer              TEXT NOT NULL,
                created_at          TIMESTAMP DEFAULT NOW(),
                search_vector       tsvector,
                FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id) ON DELETE CASCADE
            )
        """)

        # Roles
        cur.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id          SERIAL PRIMARY KEY,
                name        TEXT UNIQUE NOT NULL,
                description TEXT,
                permissions TEXT NOT NULL DEFAULT '[]',
                is_system   INTEGER DEFAULT 0,
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """)

        # LLM keys
        cur.execute("""
            CREATE TABLE IF NOT EXISTS llm_keys (
                id           SERIAL PRIMARY KEY,
                name         TEXT NOT NULL,
                provider     TEXT NOT NULL DEFAULT 'openai',
                model        TEXT NOT NULL DEFAULT 'gpt-4o',
                api_key_enc  TEXT NOT NULL,
                api_key_hint TEXT DEFAULT '',
                base_url     TEXT DEFAULT '',
                priority     INTEGER NOT NULL DEFAULT 10,
                is_active    INTEGER NOT NULL DEFAULT 1,
                created_at   TIMESTAMP DEFAULT NOW()
            )
        """)

        # LLM usage
        cur.execute("""
            CREATE TABLE IF NOT EXISTS llm_usage (
                id            SERIAL PRIMARY KEY,
                llm_key_id    INTEGER NOT NULL,
                provider      TEXT NOT NULL,
                model         TEXT NOT NULL,
                input_tokens  INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_usd      REAL DEFAULT 0.0,
                created_at    TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (llm_key_id) REFERENCES llm_keys(id) ON DELETE CASCADE
            )
        """)

        # Agent Marketplace
        cur.execute("""
            CREATE TABLE IF NOT EXISTS marketplace_agents (
                id          SERIAL PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                category    TEXT NOT NULL DEFAULT 'General',
                tools       TEXT NOT NULL DEFAULT '[]',
                icon        TEXT DEFAULT '🤖',
                source_ref  TEXT DEFAULT '',
                knowledge   TEXT DEFAULT '',
                is_active   INTEGER NOT NULL DEFAULT 1,
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """)
        # Agent memory — per-agent, per-investor learnings (after marketplace_agents)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                id                  SERIAL PRIMARY KEY,
                agent_id            INTEGER NOT NULL,
                investor_session_id INTEGER NOT NULL,
                memory_type         TEXT NOT NULL DEFAULT 'learning',
                content             TEXT NOT NULL,
                created_at          TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (agent_id) REFERENCES marketplace_agents(id) ON DELETE CASCADE,
                FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_assignments (
                id                  SERIAL PRIMARY KEY,
                agent_id            INTEGER NOT NULL,
                investor_session_id INTEGER NOT NULL,
                assigned_by         INTEGER,
                assigned_at         TIMESTAMP DEFAULT NOW(),
                UNIQUE(agent_id, investor_session_id),
                FOREIGN KEY (agent_id) REFERENCES marketplace_agents(id) ON DELETE CASCADE,
                FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (assigned_by) REFERENCES users(id)
            )
        """)

        # Custom agent builder
        cur.execute("""
            CREATE TABLE IF NOT EXISTS custom_agents (
                id                  SERIAL PRIMARY KEY,
                name                TEXT NOT NULL,
                description         TEXT DEFAULT '',
                icon                TEXT DEFAULT '🤖',
                system_prompt       TEXT DEFAULT '',
                user_prompt         TEXT DEFAULT '',
                input_type          TEXT NOT NULL DEFAULT 'chat',
                output_type         TEXT NOT NULL DEFAULT 'chat',
                output_webhook_url  TEXT DEFAULT '',
                output_webhook_secret TEXT DEFAULT '',
                api_key             TEXT NOT NULL UNIQUE,
                tools               TEXT NOT NULL DEFAULT '[]',
                is_active           INTEGER NOT NULL DEFAULT 1,
                created_by          INTEGER,
                created_at          TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)
        # Migration: add user_prompt column if missing
        cur.execute("""
            ALTER TABLE custom_agents ADD COLUMN IF NOT EXISTS user_prompt TEXT DEFAULT ''
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS custom_agent_runs (
                id          SERIAL PRIMARY KEY,
                agent_id    INTEGER NOT NULL,
                input_text  TEXT NOT NULL,
                output_text TEXT DEFAULT '',
                sources     TEXT DEFAULT '[]',
                confidence  TEXT DEFAULT '',
                input_src   TEXT DEFAULT 'chat',
                status      TEXT DEFAULT 'pending',
                created_at  TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (agent_id) REFERENCES custom_agents(id) ON DELETE CASCADE
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_schedules (
                id               SERIAL PRIMARY KEY,
                agent_id         INTEGER NOT NULL,
                name             TEXT NOT NULL,
                input_text       TEXT NOT NULL,
                schedule_type    TEXT NOT NULL DEFAULT 'interval',
                interval_minutes INTEGER DEFAULT 60,
                daily_time       TEXT DEFAULT '09:00',
                weekly_day       INTEGER DEFAULT 1,
                is_active        INTEGER NOT NULL DEFAULT 1,
                last_run_at      TIMESTAMP,
                next_run_at      TIMESTAMP,
                run_count        INTEGER DEFAULT 0,
                created_at       TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (agent_id) REFERENCES custom_agents(id) ON DELETE CASCADE
            )
        """)

        # Fund watchlist — funds queued for Risk Assessment Agent research
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fund_watchlist (
                id                 SERIAL PRIMARY KEY,
                fund_name          TEXT NOT NULL,
                ticker             TEXT DEFAULT '',
                category           TEXT DEFAULT 'Equity',
                notes              TEXT DEFAULT '',
                added_by           INTEGER REFERENCES users(id),
                added_at           TIMESTAMP DEFAULT NOW(),
                last_researched_at TIMESTAMP,
                deleted_at         TIMESTAMP DEFAULT NULL
            )
        """)

        # Fund research reports — one per fund (upserted on each run)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fund_research_reports (
                id           SERIAL PRIMARY KEY,
                fund_id      INTEGER NOT NULL UNIQUE,
                report_json  TEXT NOT NULL DEFAULT '{}',
                generated_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (fund_id) REFERENCES fund_watchlist(id) ON DELETE CASCADE
            )
        """)

        # Plugin results — stores output from each plugin run
        cur.execute("""
            CREATE TABLE IF NOT EXISTS plugin_results (
                id                  SERIAL PRIMARY KEY,
                plugin_name         TEXT NOT NULL,
                investor_session_id INTEGER,
                result_json         TEXT NOT NULL DEFAULT '{}',
                created_at          TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (investor_session_id) REFERENCES investor_sessions(id) ON DELETE CASCADE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_plugin_results_name ON plugin_results(plugin_name)")
        # Migration: add deleted_at to fund_watchlist if not exists
        cur.execute("""
            ALTER TABLE fund_watchlist ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP DEFAULT NULL
        """)

        # Seed marketplace agents
        _seed_marketplace_agents(cur)

        # Migrations for columns added after initial release (IF NOT EXISTS is safe in PG 9.6+)
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS investor_session_id INTEGER REFERENCES investor_sessions(id)")
        cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active'")
        cur.execute("ALTER TABLE llm_keys ADD COLUMN IF NOT EXISTS api_key_hint TEXT DEFAULT ''")
        cur.execute("ALTER TABLE llm_keys ADD COLUMN IF NOT EXISTS base_url TEXT DEFAULT ''")
        cur.execute("ALTER TABLE investor_sessions ADD COLUMN IF NOT EXISTS profile_text TEXT DEFAULT ''")
        cur.execute("ALTER TABLE investor_sessions ADD COLUMN IF NOT EXISTS profile_generated_at TEXT DEFAULT ''")

        # ── tsvector triggers ──────────────────────────────────────────────────

        # document_chunks
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_chunk_search_vector()
            RETURNS TRIGGER AS $$
            BEGIN
              NEW.search_vector := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        cur.execute("""
            DROP TRIGGER IF EXISTS chunk_search_vector_trigger ON document_chunks
        """)
        cur.execute("""
            CREATE TRIGGER chunk_search_vector_trigger
            BEFORE INSERT OR UPDATE ON document_chunks
            FOR EACH ROW EXECUTE FUNCTION update_chunk_search_vector()
        """)

        # knowledge_base
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_kb_search_vector()
            RETURNS TRIGGER AS $$
            BEGIN
              NEW.search_vector := to_tsvector('english',
                COALESCE(NEW.question, '') || ' ' || COALESCE(NEW.answer, ''));
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        cur.execute("DROP TRIGGER IF EXISTS kb_search_vector_trigger ON knowledge_base")
        cur.execute("""
            CREATE TRIGGER kb_search_vector_trigger
            BEFORE INSERT OR UPDATE ON knowledge_base
            FOR EACH ROW EXECUTE FUNCTION update_kb_search_vector()
        """)

        # session_answers
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_session_answers_search_vector()
            RETURNS TRIGGER AS $$
            BEGIN
              NEW.search_vector := to_tsvector('english',
                COALESCE(NEW.question, '') || ' ' || COALESCE(NEW.answer, ''));
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        cur.execute("DROP TRIGGER IF EXISTS session_answers_search_vector_trigger ON session_answers")
        cur.execute("""
            CREATE TRIGGER session_answers_search_vector_trigger
            BEFORE INSERT OR UPDATE ON session_answers
            FOR EACH ROW EXECUTE FUNCTION update_session_answers_search_vector()
        """)

        # investor_doc_chunks
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_inv_chunk_search_vector()
            RETURNS TRIGGER AS $$
            BEGIN
              NEW.search_vector := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        cur.execute("DROP TRIGGER IF EXISTS inv_chunk_search_vector_trigger ON investor_doc_chunks")
        cur.execute("""
            CREATE TRIGGER inv_chunk_search_vector_trigger
            BEFORE INSERT OR UPDATE ON investor_doc_chunks
            FOR EACH ROW EXECUTE FUNCTION update_inv_chunk_search_vector()
        """)

        # ── GIN indexes ────────────────────────────────────────────────────────
        cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_fts ON document_chunks USING GIN(search_vector)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kb_fts ON knowledge_base USING GIN(search_vector)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_session_answers_fts ON session_answers USING GIN(search_vector)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_inv_chunks_fts ON investor_doc_chunks USING GIN(search_vector)")

        # ── Seed data ──────────────────────────────────────────────────────────

        all_perms = json.dumps([
            "dashboard", "qa_sessions", "upload_documents", "delete_documents",
            "manage_investors", "manage_kb", "live_support", "team_management", "analytics"
        ])
        analyst_perms = json.dumps(["dashboard", "qa_sessions", "live_support", "analytics"])

        cur.execute("""
            INSERT INTO roles (name, description, permissions, is_system)
            VALUES ('admin', 'Full access to all features', %s, 1)
            ON CONFLICT (name) DO NOTHING
        """, (all_perms,))
        cur.execute("""
            INSERT INTO roles (name, description, permissions, is_system)
            VALUES ('analyst', 'View and use core features, no admin actions', %s, 1)
            ON CONFLICT (name) DO NOTHING
        """, (analyst_perms,))

        # Sentinel LLM key for env-var usage tracking
        cur.execute("""
            INSERT INTO llm_keys (id, name, provider, model, api_key_enc, api_key_hint, priority, is_active)
            VALUES (1, 'Environment Key (OPENAI_API_KEY)', 'openai', 'gpt-4o', '', 'env', 999, 0)
            ON CONFLICT (id) DO NOTHING
        """)

        # Default admin user
        admin_hash = _hash("admin123")
        cur.execute("""
            INSERT INTO users (email, name, password_hash, role)
            VALUES ('admin@fund.com', 'Administrator', %s, 'admin')
            ON CONFLICT (email) DO NOTHING
        """, (admin_hash,))

        conn.commit()
    finally:
        put_db(conn)


# ── Agent Marketplace seed ─────────────────────────────────────────────────────

_MARKETPLACE_AGENTS = [
    {
        "name": "Document Analyzer",
        "description": "Extracts and summarizes key information from fund documents. Identifies critical clauses, terms, and disclosures across LPAs, PPMs, and subscription agreements.",
        "category": "Document Intelligence",
        "tools": ["Document Search", "Text Extraction", "Summarization"],
        "icon": "📄",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Uses Document Search to locate relevant sections across all uploaded fund files (LPAs, PPMs, side letters, subscription docs). Text Extraction pulls exact clauses, defined terms, and page references. Summarization condenses multi-page sections into concise investor-ready summaries. Knows how to parse legal boilerplate, identify key obligations, and flag non-standard provisions. Ask it to find specific clauses, compare document versions, or explain complex terms in plain language.",
    },
    {
        "name": "Risk Assessment Agent",
        "description": "Identifies and evaluates risk factors across fund documentation. Reviews concentration risk, liquidity risk, leverage constraints, and key-person dependencies.",
        "category": "Risk & Compliance",
        "tools": ["Document Search", "Risk Scoring", "Report Generation"],
        "icon": "⚠️",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search locates risk disclosures, concentration limits, and leverage covenants buried in offering documents. Risk Scoring evaluates each identified risk factor on likelihood and impact — covering concentration risk, liquidity mismatch, key-person dependency, counterparty exposure, and market risk. Report Generation compiles findings into a structured risk summary with severity ratings. Best used to ask: what are the top risks in this fund, what leverage is permitted, who are the key persons, and what happens if they leave.",
    },
    {
        "name": "ESG Analysis Agent",
        "description": "Reviews environmental, social, and governance policies embedded in fund documents. Flags ESG commitments, exclusion criteria, and reporting obligations.",
        "category": "ESG",
        "tools": ["Document Search", "Policy Review", "ESG Scoring"],
        "icon": "🌿",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search finds ESG-related sections including responsible investment policies, exclusion lists, and Article 8/9 SFDR disclosures. Policy Review cross-references commitments against UN PRI, TCFD, and SFDR frameworks to check alignment. ESG Scoring rates the fund's overall ESG posture across Environmental (carbon, climate), Social (labour, DEI), and Governance (board structure, conflicts) dimensions. Ask it about ESG ratings, exclusion criteria, climate risk disclosures, or UN PRI signatory status.",
    },
    {
        "name": "Legal Review Agent",
        "description": "Reviews legal terms in LPA and subscription agreements. Highlights governance rights, LP protections, transfer restrictions, and default provisions.",
        "category": "Legal",
        "tools": ["Document Search", "Clause Extraction", "Legal Summarization"],
        "icon": "⚖️",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search retrieves specific legal provisions from LPAs, subscription agreements, and side letters. Clause Extraction isolates defined terms, conditions precedent, representation and warranty clauses, indemnities, and governing law provisions with exact text and location. Legal Summarization translates complex legal language into plain-English explanations. Covers: GP removal rights, advisory committee composition, transfer restrictions, default remedies, no-fault divorce triggers, and amendment procedures.",
    },
    {
        "name": "Financial Performance Analyst",
        "description": "Analyzes fund performance metrics including IRR, MOIC, DPI, RVPI, and TVPI. Provides context against benchmarks and identifies performance trends.",
        "category": "Financial Analysis",
        "tools": ["Document Search", "Data Analysis", "Benchmarking"],
        "icon": "📈",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search retrieves performance data from fund reports, capital account statements, and investor letters. Data Analysis calculates IRR (gross/net), MOIC, DPI (distributions to paid-in), RVPI (residual value to paid-in), and TVPI — breaking them down by vintage year, investment type, and portfolio company. Benchmarking compares returns against Cambridge Associates, Preqin quartile rankings, and public market equivalents (PME). Ask it about vintage year performance, net vs gross IRR spread, distribution history, or how the fund ranks vs peers.",
    },
    {
        "name": "Fee Structure Analyzer",
        "description": "Breaks down management fees, carried interest, waterfall mechanics, hurdle rates, and clawback provisions. Calculates net-of-fee return scenarios.",
        "category": "Financial Analysis",
        "tools": ["Document Search", "Fee Modeling", "Calculation Engine"],
        "icon": "💰",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search extracts fee provisions from the LPA including management fee basis, step-downs, offsets, and waiver terms. Fee Modeling maps out the full waterfall — return of capital, preferred return (hurdle), GP catch-up, and carried interest split — showing exactly when and how distributions flow. Calculation Engine runs net-of-fee return scenarios at different performance levels, models the impact of fee offsets (monitoring, transaction fees), and computes the true cost of the carry arrangement. Ask it to model scenarios, explain the waterfall, or compare fee terms to market standard.",
    },
    {
        "name": "Compliance Checker",
        "description": "Reviews regulatory compliance disclosures including AIFMD, Form ADV, SEC filings, and FATCA/CRS requirements. Flags potential compliance gaps.",
        "category": "Risk & Compliance",
        "tools": ["Document Search", "Regulatory Database", "Compliance Scoring"],
        "icon": "✅",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search locates regulatory disclosures, compliance policies, and registration statements within fund documents. Regulatory Database cross-checks disclosures against current requirements: AIFMD (EU alternative fund managers), Form ADV (SEC investment adviser registration), FATCA/CRS (tax information exchange), MiFID II (conflicts of interest), and GDPR (data protection). Compliance Scoring rates overall compliance posture and identifies gaps, missing disclosures, or outdated filings. Ask about regulatory registrations, conflicts of interest disclosures, or anti-money laundering procedures.",
    },
    {
        "name": "Investment Strategy Analyzer",
        "description": "Reviews the fund's investment thesis, target sectors, geographies, and deal sourcing strategy. Benchmarks against peer funds and market conditions.",
        "category": "Strategy",
        "tools": ["Document Search", "Market Research", "Peer Comparison"],
        "icon": "🎯",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search retrieves the investment mandate, target sectors, stage focus, check-size parameters, and portfolio construction guidelines from the PPM or offering documents. Market Research (web search + browsing) pulls current market data, sector trends, and deal activity to validate the thesis against real conditions. Peer Comparison benchmarks the strategy against comparable fund managers — coverage, differentiation, and competitive positioning. Ask about the investment thesis, target sectors, deal sourcing edge, how the strategy has evolved, or how it compares to peer managers.",
    },
    {
        "name": "Portfolio Analytics Agent",
        "description": "Analyzes portfolio company composition, sector diversification, stage distribution, and valuation methodology. Tracks follow-on reserves and portfolio construction.",
        "category": "Financial Analysis",
        "tools": ["Document Search", "Portfolio Modeling", "Visualization"],
        "icon": "🗂️",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search retrieves portfolio schedules, company descriptions, investment dates, and cost/fair value data from fund reports. Portfolio Modeling calculates sector concentration, stage distribution (seed/A/B/growth), geographic exposure, follow-on reserve ratios, and ownership percentages. Visualization generates charts showing portfolio composition, NAV bridge, and company-level attribution. Ask it about portfolio concentration, sector breakdown, reserve strategy, largest positions, or which companies are marked up/down.",
    },
    {
        "name": "DDQ Auto-Responder",
        "description": "Automatically drafts professional responses to standard DDQ questions by searching fund documents and knowledge base. Trained on institutional DDQ formats.",
        "category": "DDQ Automation",
        "tools": ["Document Search", "Knowledge Base", "Draft Generation"],
        "icon": "✍️",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search retrieves source material from all fund documents to ground every response in actual fund data. Knowledge Base draws on previously approved DDQ answers, institutional best practices, and ILPA/QPAM standard question libraries. Draft Generation produces polished, fund-specific responses that follow institutional DDQ conventions — structured sections, appropriate disclaimers, and consistent tone. Handles standard question categories: AML/KYC, compliance, risk, performance, team, ESG, operations, and legal. Ask it to draft a response to any DDQ question or review/improve an existing answer.",
    },
    {
        "name": "Report Generator",
        "description": "Creates structured investor reports including quarterly updates, capital call notices, and distribution notices by synthesizing data from fund documents.",
        "category": "Reporting",
        "tools": ["Document Search", "Template Engine", "PDF Generation"],
        "icon": "📊",
        "source_ref": "https://github.com/anthropics/skills",
        "knowledge": "Document Search pulls financial data, portfolio updates, and narrative content from the fund's source documents. Template Engine applies institutional report formats — quarterly investor letters (QIR), capital account statements, capital call and distribution notices, and annual reports — ensuring consistent structure and required sections. Covers: portfolio company updates, NAV reconciliation, cash flow summaries, and LP-specific account information. Ask it to draft a quarterly update, prepare a capital call notice, summarise recent portfolio activity, or create a distribution notice.",
    },
    {
        "name": "Research Assistant",
        "description": "Provides market context, sector analysis, and comparable fund benchmarking to support investment decision-making and due diligence processes.",
        "category": "Research",
        "tools": ["Web Search", "URL Browser", "Document Search", "Summarization"],
        "icon": "🔬",
        "source_ref": "https://github.com/anthropics/skills",
        "knowledge": "Web Search queries live internet sources (DuckDuckGo) for real-time market data, company financials, news, and sector trends — always using the current year for accurate results. URL Browser opens and reads full web pages (Yahoo Finance, Reuters, Bloomberg, Wikipedia, company sites) to extract precise figures like market caps, revenue, valuation multiples, and news. Document Search checks fund documents for internal context. Summarization synthesises multi-source findings into a clear, cited answer with charts when data is numerical. Ask anything about external markets: company valuations, sector trends, competitor analysis, macroeconomic data, or recent news.",
    },
    {
        "name": "Tax & Structure Advisor",
        "description": "Reviews fund structure for tax efficiency, blocker entity usage, treaty eligibility, UBTI considerations, and ECI exposure for international investors.",
        "category": "Legal",
        "tools": ["Document Search", "Tax Analysis", "Structure Review"],
        "icon": "🏛️",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search retrieves fund structure diagrams, partnership agreement tax provisions, and side-letter tax accommodations. Tax Analysis evaluates US tax considerations: UBTI (unrelated business taxable income) for tax-exempt LPs, ECI (effectively connected income) for foreign investors, PFIC/CFC implications, withholding obligations, and K-1 reporting timelines. Structure Review assesses the fund vehicle stack — master/feeder, parallel funds, blocker corporations, SPVs — and explains why each entity exists and its tax purpose. Ask about the fund structure, tax documents timeline, blocker availability, treaty benefits, or whether UBTI is an issue for pension fund investors.",
    },
    {
        "name": "LP Rights Monitor",
        "description": "Tracks and summarizes LP rights including LPAC membership, co-investment rights, key-person protections, no-fault divorce provisions, and reporting obligations.",
        "category": "Legal",
        "tools": ["Document Search", "Rights Extraction", "Alert System"],
        "icon": "🛡️",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search locates LP-protective provisions across the LPA, side letters, and subscription documents. Rights Extraction catalogues all LP rights by type: information rights (financial statements, K-1 timing, site visits), consent rights (LPAC approval thresholds, major decisions), protective rights (key-person clause triggers, no-fault removal vote, GP cause removal), economic rights (MFN provisions, fee offsets, co-investment allocation). Alert System flags upcoming obligations, notice periods, and time-sensitive LP action windows. Ask it what rights you have as an LP, whether MFN applies, how to trigger the key-person clause, or what LPAC approval is required for GP decisions.",
    },
    {
        "name": "Valuation Review Agent",
        "description": "Reviews valuation methodology, fair value policies, independent valuation practices, and marks against industry standards and IPEV guidelines.",
        "category": "Financial Analysis",
        "tools": ["Document Search", "Valuation Models", "IPEV Guidelines"],
        "icon": "🔍",
        "source_ref": "https://github.com/ashishpatel26/500-AI-Agents-Projects",
        "knowledge": "Document Search retrieves valuation policies, fair value disclosures, and portfolio marks from financial statements and offering documents. Valuation Models applies and explains standard PE/VC valuation approaches: comparable company analysis (EV/Revenue, EV/EBITDA multiples), discounted cash flow (DCF), last-round financing method, and option-pricing model (OPM) for early-stage companies. IPEV Guidelines cross-checks the fund's stated methodology against the International Private Equity Valuation Board's best practice guidelines to identify deviations, stale marks, or aggressive assumptions. Ask about specific company valuations, the valuation basis used, whether independent valuers are used, or how marks compare to public peers.",
    },
]


def _seed_marketplace_agents(cur):
    # Add knowledge column if it doesn't exist yet (safe migration)
    cur.execute("""
        ALTER TABLE marketplace_agents ADD COLUMN IF NOT EXISTS knowledge TEXT DEFAULT ''
    """)
    for agent in _MARKETPLACE_AGENTS:
        cur.execute("""
            INSERT INTO marketplace_agents (name, description, category, tools, icon, source_ref, knowledge)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
              SET description = EXCLUDED.description,
                  tools       = EXCLUDED.tools,
                  knowledge   = EXCLUDED.knowledge
        """, (
            agent["name"], agent["description"], agent["category"],
            json.dumps(agent["tools"]), agent["icon"], agent["source_ref"],
            agent.get("knowledge", "")
        ))


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def get_user_by_email(email: str) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def verify_login(email: str, password: str) -> dict:
    user = get_user_by_email(email)
    if user and user["password_hash"] == _hash(password):
        return user
    return None


def create_user(email: str, name: str, password: str, role: str = "analyst") -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            INSERT INTO users (email, name, password_hash, role)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, (email, name, _hash(password), role))
        uid = cur.fetchone()["id"]
        conn.commit()
        return uid
    finally:
        put_db(conn)


def get_user_by_id(user_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def update_user_profile(user_id: int, name: str, email: str) -> bool:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET name=%s, email=%s WHERE id=%s", (name, email, user_id))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        put_db(conn)


def change_user_password(user_id: int, current_password: str, new_password: str) -> bool:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT password_hash FROM users WHERE id=%s", (user_id,))
        row = cur.fetchone()
        if not row or row["password_hash"] != _hash(current_password):
            return False
        cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (_hash(new_password), user_id))
        conn.commit()
        return True
    finally:
        put_db(conn)


# ── Documents ──────────────────────────────────────────────────────────────────

def get_fund_document(doc_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM fund_documents WHERE id=%s", (doc_id,))
        row = cur.fetchone()
        return _row(dict(row)) if row else {}
    finally:
        put_db(conn)


def list_fund_documents() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT d.*, u.name as uploaded_by_name
            FROM fund_documents d
            LEFT JOIN users u ON d.uploaded_by = u.id
            WHERE d.status = 'active'
            ORDER BY d.uploaded_at DESC
        """)
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_fund_document(doc_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM fund_documents WHERE id=%s", (doc_id,))
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def delete_fund_document(doc_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE fund_documents SET status='deleted' WHERE id=%s", (doc_id,))
        conn.commit()
    finally:
        put_db(conn)


def _to_tsquery(query: str) -> str:
    """Build a PostgreSQL tsquery from a natural language query."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "about", "and", "or", "not", "what", "how", "when",
        "where", "who", "which", "why", "can", "your", "our", "my", "tell",
        "please", "give", "me", "us", "you", "its", "this", "that", "these",
        "those", "also"
    }
    words = re.findall(r"[a-z0-9]+", query.lower())
    terms = [w for w in words if len(w) > 2 and w not in stop]
    if not terms:
        return None
    return " & ".join(terms)


def search_fund_documents(query: str, limit: int = 8) -> list:
    """
    Hybrid 3-layer search:
    Layer 1 — document metadata/summary (boost x2)
    Layer 2 — page summaries (boost x1.5)
    Layer 3 — chunk full-text (base score)
    Results are combined, deduplicated, and ranked by total score.
    Falls back to ILIKE if no FTS results.
    """
    tsq = _to_tsquery(query)
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if tsq:
            try:
                cur.execute("""
                    WITH
                    -- Layer 1: document-level metadata match
                    meta_boost AS (
                        SELECT id as doc_id,
                               ts_rank(search_meta, to_tsquery('english', %s)) * 2.0 as bonus
                        FROM fund_documents
                        WHERE search_meta @@ to_tsquery('english', %s)
                          AND status = 'active'
                    ),
                    -- Layer 2: page summary match
                    page_boost AS (
                        SELECT ps.document_id as doc_id,
                               'p.' || ps.page_num::text as page_ref,
                               ts_rank(ps.search_vector, to_tsquery('english', %s)) * 1.5 as bonus,
                               ps.summary as page_summary
                        FROM page_summaries ps
                        JOIN fund_documents d ON d.id = ps.document_id
                        WHERE ps.search_vector @@ to_tsquery('english', %s)
                          AND d.status = 'active'
                    ),
                    -- Layer 3: chunk scores
                    chunk_scores AS (
                        SELECT dc.chunk_text, dc.page_ref, dc.section_ref,
                               d.name as doc_name, d.doc_type, d.ai_doc_type,
                               d.summary as doc_summary, d.id as document_id,
                               COALESCE(ts_rank(dc.search_vector, to_tsquery('english', %s)), 0) as base_score
                        FROM document_chunks dc
                        JOIN fund_documents d ON dc.document_id = d.id
                        WHERE d.status = 'active'
                          AND (
                              dc.search_vector @@ to_tsquery('english', %s)
                              OR d.id IN (SELECT doc_id FROM meta_boost)
                              OR d.id IN (SELECT doc_id FROM page_boost)
                          )
                    )
                    SELECT cs.*,
                           COALESCE(mb.bonus, 0) as meta_bonus,
                           COALESCE(pb.bonus, 0) as page_bonus,
                           (cs.base_score
                            + COALESCE(mb.bonus, 0)
                            + COALESCE(pb.bonus, 0)) as total_score
                    FROM chunk_scores cs
                    LEFT JOIN meta_boost mb ON mb.doc_id = cs.document_id
                    LEFT JOIN page_boost pb ON pb.doc_id = cs.document_id
                                           AND pb.page_ref = cs.page_ref
                    ORDER BY total_score DESC
                    LIMIT %s
                """, (tsq, tsq, tsq, tsq, tsq, tsq, limit))
                rows = [dict(r) for r in cur.fetchall()]
                if rows:
                    return rows
            except Exception:
                conn.rollback()

        # Fallback: ILIKE on chunks
        cur.execute("""
            SELECT dc.chunk_text, dc.page_ref, dc.section_ref,
                   d.name as doc_name, d.doc_type, d.ai_doc_type,
                   d.summary as doc_summary, d.id as document_id,
                   0.0 as base_score, 0.0 as meta_bonus,
                   0.0 as page_bonus, 0.0 as total_score
            FROM document_chunks dc
            JOIN fund_documents d ON dc.document_id = d.id
            WHERE dc.chunk_text ILIKE %s AND d.status = 'active'
            LIMIT %s
        """, (f"%{query}%", limit))
        return [dict(r) for r in cur.fetchall()]
    finally:
        put_db(conn)


def search_investor_documents(query: str, investor_session_id: int, limit: int = 4) -> list:
    tsq = _to_tsquery(query)
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        rows = []
        if tsq:
            try:
                cur.execute("""
                    SELECT idc.chunk_text, id.name as doc_name, id.doc_type,
                           ts_rank(idc.search_vector, to_tsquery('english', %s)) as score
                    FROM investor_doc_chunks idc
                    JOIN investor_documents id ON idc.investor_doc_id = id.id
                    WHERE idc.search_vector @@ to_tsquery('english', %s)
                      AND id.investor_session_id = %s
                    ORDER BY score DESC
                    LIMIT %s
                """, (tsq, tsq, investor_session_id, limit))
                rows = cur.fetchall()
            except Exception:
                conn.rollback()
                rows = []
        if not rows:
            cur.execute("""
                SELECT idc.chunk_text, id.name as doc_name, id.doc_type, 0.0 as score
                FROM investor_doc_chunks idc
                JOIN investor_documents id ON idc.investor_doc_id = id.id
                WHERE idc.chunk_text ILIKE %s AND id.investor_session_id = %s
                LIMIT %s
            """, (f"%{query}%", investor_session_id, limit))
            rows = cur.fetchall()
        return list(rows)
    finally:
        put_db(conn)


def find_similar_questions(question: str, limit: int = 3) -> list:
    """Return previously answered similar questions for context."""
    words = [w for w in question.lower().split() if len(w) > 4][:5]
    if not words:
        return []
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        results = []
        for word in words:
            cur.execute("""
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
                WHERE m.role = 'user' AND m.content LIKE %s
                LIMIT 2
            """, (f"%{word}%",))
            for r in cur.fetchall():
                d = dict(r)
                if not any(x["question"] == d["question"] for x in results):
                    results.append(d)
        return results[:limit]
    finally:
        put_db(conn)


# ── Investor Sessions ──────────────────────────────────────────────────────────

def list_investor_sessions() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT s.id, s.investor_name, s.investor_entity, s.created_by,
                   s.notes, s.created_at, s.profile_text, s.profile_generated_at,
                   u.name as created_by_name,
                   COUNT(DISTINCT da.id) as doc_count,
                   COUNT(DISTINCT c.id) as conversation_count
            FROM investor_sessions s
            LEFT JOIN users u ON s.created_by = u.id
            LEFT JOIN document_assignments da ON da.investor_session_id = s.id
            LEFT JOIN conversations c ON c.investor_session_id = s.id
            GROUP BY s.id, s.investor_name, s.investor_entity, s.created_by,
                     s.notes, s.created_at, s.profile_text, s.profile_generated_at, u.name
            ORDER BY s.created_at DESC
        """)
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_investor_session(session_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT s.*, u.name as created_by_name
            FROM investor_sessions s
            LEFT JOIN users u ON s.created_by = u.id
            WHERE s.id = %s
        """, (session_id,))
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def create_investor_session(investor_name: str, investor_entity: str,
                             notes: str, created_by: int) -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            INSERT INTO investor_sessions (investor_name, investor_entity, notes, created_by)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, (investor_name, investor_entity, notes, created_by))
        sid = cur.fetchone()["id"]
        conn.commit()
        return sid
    finally:
        put_db(conn)


def get_investor_questions(session_id: int, limit: int = 50) -> list:
    """Return all user questions asked by this investor across all their conversations."""
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT m.content, m.themes, m.created_at
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.investor_session_id = %s AND m.role = 'user'
            ORDER BY m.created_at ASC
            LIMIT %s
        """, (session_id, limit))
        result = []
        for r in cur.fetchall():
            d = dict(r)
            d["themes"] = json.loads(d["themes"]) if d["themes"] else []
            result.append(d)
        return result
    finally:
        put_db(conn)


def save_investor_profile(session_id: int, profile_text: str):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE investor_sessions SET profile_text=%s, profile_generated_at=NOW() WHERE id=%s",
            (profile_text, session_id)
        )
        conn.commit()
    finally:
        put_db(conn)


# ── Conversations & Messages ───────────────────────────────────────────────────

def create_conversation(created_by: int, investor_session_id: int = None,
                         title: str = None) -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            INSERT INTO conversations (title, investor_session_id, created_by)
            VALUES (%s, %s, %s) RETURNING id
        """, (title, investor_session_id, created_by))
        cid = cur.fetchone()["id"]
        conn.commit()
        return cid
    finally:
        put_db(conn)


def get_conversation(conv_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT c.*, u.name as created_by_name,
                   s.investor_name, s.investor_entity
            FROM conversations c
            LEFT JOIN users u ON c.created_by = u.id
            LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
            WHERE c.id = %s
        """, (conv_id,))
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def list_conversations(user_id: int = None, limit: int = 30) -> list:  # noqa: ARG001
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   u.name as created_by_name,
                   s.investor_name,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN users u ON c.created_by = u.id
            LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id, c.title, c.created_at, c.updated_at, u.name, s.investor_name
            ORDER BY c.updated_at DESC
            LIMIT %s
        """, (limit,))
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_messages(conv_id: int) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT * FROM messages WHERE conversation_id=%s ORDER BY created_at ASC
        """, (conv_id,))
        msgs = []
        for r in cur.fetchall():
            d = dict(r)
            d["sources"] = json.loads(d["sources"]) if d["sources"] else []
            d["themes"] = json.loads(d["themes"]) if d["themes"] else []
            msgs.append(d)
        return msgs
    finally:
        put_db(conn)


def add_message(conv_id: int, role: str, content: str,
                sources: list = None, draft_response: str = None,
                themes: list = None) -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            INSERT INTO messages (conversation_id, role, content, sources, draft_response, themes)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
        """, (conv_id, role, content,
              json.dumps(sources or []),
              draft_response,
              json.dumps(themes or [])))
        msg_id = cur.fetchone()["id"]
        cur.execute("UPDATE conversations SET updated_at=NOW() WHERE id=%s", (conv_id,))

        # Store themes for analytics
        for theme in (themes or []):
            cur.execute("INSERT INTO question_themes (message_id, theme) VALUES (%s, %s)",
                        (msg_id, theme))

        conn.commit()
        return msg_id
    finally:
        put_db(conn)


def update_conversation_title(conv_id: int, title: str):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET title=%s WHERE id=%s", (title, conv_id))
        conn.commit()
    finally:
        put_db(conn)


# ── Analytics ──────────────────────────────────────────────────────────────────

def get_dashboard_stats() -> dict:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM messages WHERE role='user'")
        total_questions = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM fund_documents WHERE status='active'")
        total_documents = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM investor_sessions")
        total_investors = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = cur.fetchone()[0]
        return {
            "total_questions": total_questions,
            "total_conversations": total_conversations,
            "total_documents": total_documents,
            "total_investors": total_investors,
            "total_chunks": total_chunks,
        }
    finally:
        put_db(conn)


def get_theme_analytics() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT theme, COUNT(*) as count
            FROM question_themes
            GROUP BY theme
            ORDER BY count DESC
            LIMIT 20
        """)
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_document_citation_stats() -> list:
    """Which documents are referenced most in answers."""
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT d.name, d.doc_type, COUNT(m.id) as citation_count
            FROM fund_documents d
            JOIN document_chunks dc ON dc.document_id = d.id
            JOIN messages m ON m.sources LIKE '%' || d.name || '%'
            WHERE d.status = 'active'
            GROUP BY d.id, d.name, d.doc_type
            ORDER BY citation_count DESC
            LIMIT 10
        """)
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def search_assigned_fund_documents(query: str, investor_session_id: int, limit: int = 6) -> list:
    """Search only fund documents assigned to this investor session."""
    assigned_ids = get_assigned_document_ids(investor_session_id)
    if not assigned_ids:
        return []
    tsq = _to_tsquery(query)
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        rows = []
        if tsq:
            try:
                cur.execute("""
                    SELECT dc.chunk_text, dc.page_ref, dc.section_ref,
                           d.name as doc_name, d.doc_type, d.id as document_id,
                           ts_rank(dc.search_vector, to_tsquery('english', %s)) as score
                    FROM document_chunks dc
                    JOIN fund_documents d ON dc.document_id = d.id
                    WHERE dc.search_vector @@ to_tsquery('english', %s)
                      AND d.status = 'active'
                      AND d.id = ANY(%s::int[])
                    ORDER BY score DESC
                    LIMIT %s
                """, (tsq, tsq, assigned_ids, limit))
                rows = cur.fetchall()
            except Exception:
                conn.rollback()
                rows = []
        if not rows:
            cur.execute("""
                SELECT dc.chunk_text, dc.page_ref, dc.section_ref,
                       d.name as doc_name, d.doc_type, d.id as document_id,
                       0.0 as score
                FROM document_chunks dc
                JOIN fund_documents d ON dc.document_id = d.id
                WHERE dc.chunk_text ILIKE %s AND d.status = 'active'
                  AND d.id = ANY(%s::int[])
                LIMIT %s
            """, (f"%{query}%", assigned_ids, limit))
            rows = cur.fetchall()
        return list(rows)
    finally:
        put_db(conn)


# ── Document Assignments (investor portal) ────────────────────────────────────

def get_assigned_document_ids(investor_session_id: int) -> list:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT document_id FROM document_assignments WHERE investor_session_id=%s",
            (investor_session_id,)
        )
        return [r[0] for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_assigned_documents(investor_session_id: int) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT d.id, d.name, d.doc_type, da.assigned_at,
                   (SELECT dc.chunk_text FROM document_chunks dc
                    WHERE dc.document_id = d.id
                    ORDER BY dc.chunk_index LIMIT 1) AS summary_snippet
            FROM document_assignments da
            JOIN fund_documents d ON da.document_id = d.id
            WHERE da.investor_session_id = %s AND d.status = 'active'
            ORDER BY d.name
        """, (investor_session_id,))
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def assign_documents_to_investor(investor_session_id: int, doc_ids: list, assigned_by: int):
    """Replace all document assignments for an investor session."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_assignments WHERE investor_session_id=%s", (investor_session_id,))
        for doc_id in doc_ids:
            cur.execute("""
                INSERT INTO document_assignments (investor_session_id, document_id, assigned_by)
                VALUES (%s, %s, %s)
                ON CONFLICT (investor_session_id, document_id) DO NOTHING
            """, (investor_session_id, doc_id, assigned_by))
        conn.commit()
    finally:
        put_db(conn)


# ── Investor User Accounts ────────────────────────────────────────────────────

def create_investor_user(email: str, name: str, password: str, investor_session_id: int) -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Remove any existing investor user for this session
        cur.execute("DELETE FROM users WHERE investor_session_id=%s", (investor_session_id,))
        cur.execute("""
            INSERT INTO users (email, name, password_hash, role, investor_session_id)
            VALUES (%s, %s, %s, 'investor', %s) RETURNING id
        """, (email, name, _hash(password), investor_session_id))
        uid = cur.fetchone()["id"]
        conn.commit()
        return uid
    finally:
        put_db(conn)


def list_roles() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM roles ORDER BY is_system DESC, name")
        result = []
        for r in cur.fetchall():
            d = dict(r)
            d["permissions"] = json.loads(d["permissions"] or "[]")
            result.append(d)
        return result
    finally:
        put_db(conn)


def get_role(role_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM roles WHERE id=%s", (role_id,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["permissions"] = json.loads(d["permissions"] or "[]")
        return d
    finally:
        put_db(conn)


def create_role(name: str, description: str, permissions: list) -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "INSERT INTO roles (name, description, permissions) VALUES (%s, %s, %s) RETURNING id",
            (name.lower().replace(" ", "_"), description, json.dumps(permissions))
        )
        rid = cur.fetchone()["id"]
        conn.commit()
        return rid
    finally:
        put_db(conn)


def update_role(role_id: int, name: str, description: str, permissions: list):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT name, is_system FROM roles WHERE id=%s", (role_id,))
        role = cur.fetchone()
        if not role:
            return
        # Admin always keeps full permissions — never editable
        if role["name"] == "admin":
            return
        if role["is_system"]:
            # System roles (non-admin): update permissions only
            cur.execute("UPDATE roles SET permissions=%s WHERE id=%s",
                        (json.dumps(permissions), role_id))
        else:
            # Custom roles: update everything
            cur.execute("UPDATE roles SET name=%s, description=%s, permissions=%s WHERE id=%s",
                        (name.lower().replace(" ", "_"), description, json.dumps(permissions), role_id))
        conn.commit()
    finally:
        put_db(conn)


def delete_role(role_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM roles WHERE id=%s AND is_system=0", (role_id,))
        conn.commit()
    finally:
        put_db(conn)


def update_user_role(user_id: int, role: str):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET role=%s WHERE id=%s AND role != 'investor'", (role, user_id))
        conn.commit()
    finally:
        put_db(conn)


def delete_user(user_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id=%s AND role != 'investor'", (user_id,))
        conn.commit()
    finally:
        put_db(conn)


def get_investor_user_by_id(user_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, email, name, role, investor_session_id, created_at FROM users WHERE id=%s AND role='investor'",
            (user_id,)
        )
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def get_investor_user(investor_session_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, email, name, role, investor_session_id FROM users WHERE investor_session_id=%s",
            (investor_session_id,)
        )
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def list_investor_conversations(investor_session_id: int) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.investor_session_id = %s AND (c.status IS NULL OR c.status != 'deleted')
            GROUP BY c.id, c.title, c.created_at, c.updated_at
            ORDER BY c.updated_at DESC
        """, (investor_session_id,))
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def soft_delete_investor_conversation(conv_id: int, investor_session_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET status='deleted' WHERE id=%s AND investor_session_id=%s",
            (conv_id, investor_session_id)
        )
        conn.commit()
    finally:
        put_db(conn)


def rename_investor_conversation(conv_id: int, investor_session_id: int, title: str):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET title=%s WHERE id=%s AND investor_session_id=%s",
            (title.strip(), conv_id, investor_session_id)
        )
        conn.commit()
    finally:
        put_db(conn)


# ── Human Agent Handover ───────────────────────────────────────────────────────

def create_handover_request(conversation_id: int, investor_session_id: int, reason: str) -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Cancel any previous open request for this conversation
        cur.execute("""
            UPDATE handover_requests SET status='cancelled'
            WHERE conversation_id=%s AND status='pending'
        """, (conversation_id,))
        cur.execute("""
            INSERT INTO handover_requests (conversation_id, investor_session_id, reason)
            VALUES (%s, %s, %s) RETURNING id
        """, (conversation_id, investor_session_id, reason))
        hid = cur.fetchone()["id"]
        cur.execute("UPDATE conversations SET status='pending_handover' WHERE id=%s", (conversation_id,))
        conn.commit()
        return hid
    finally:
        put_db(conn)


def get_pending_handovers() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT h.*, c.title as conv_title,
                   s.investor_name, s.investor_entity
            FROM handover_requests h
            JOIN conversations c ON h.conversation_id = c.id
            LEFT JOIN investor_sessions s ON h.investor_session_id = s.id
            WHERE h.status IN ('pending', 'claimed')
            ORDER BY h.requested_at DESC
        """)
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_pending_handover_count() -> int:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM handover_requests WHERE status='pending'")
        return cur.fetchone()[0]
    finally:
        put_db(conn)


def get_handover_for_conversation(conversation_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT h.*, u.name as claimed_by_name
            FROM handover_requests h
            LEFT JOIN users u ON h.claimed_by = u.id
            WHERE h.conversation_id=%s AND h.status IN ('pending','claimed')
            ORDER BY h.requested_at DESC LIMIT 1
        """, (conversation_id,))
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def claim_handover(handover_id: int, user_id: int):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            UPDATE handover_requests SET status='claimed', claimed_by=%s, claimed_at=NOW()
            WHERE id=%s
        """, (user_id, handover_id))
        cur.execute("SELECT conversation_id FROM handover_requests WHERE id=%s", (handover_id,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE conversations SET status='human_active' WHERE id=%s",
                        (row["conversation_id"],))
        conn.commit()
    finally:
        put_db(conn)


def resolve_handover(handover_id: int):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            UPDATE handover_requests SET status='resolved', resolved_at=NOW()
            WHERE id=%s
        """, (handover_id,))
        cur.execute("SELECT conversation_id FROM handover_requests WHERE id=%s", (handover_id,))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE conversations SET status='active' WHERE id=%s",
                        (row["conversation_id"],))
        conn.commit()
    finally:
        put_db(conn)


def get_messages_since(conversation_id: int, last_id: int) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT * FROM messages WHERE conversation_id=%s AND id>%s ORDER BY created_at ASC
        """, (conversation_id, last_id))
        msgs = []
        for r in cur.fetchall():
            d = dict(r)
            d["sources"] = []
            d["themes"] = []
            msgs.append(d)
        return msgs
    finally:
        put_db(conn)


def update_conversation_status(conversation_id: int, status: str):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET status=%s WHERE id=%s", (status, conversation_id))
        conn.commit()
    finally:
        put_db(conn)


# ── Knowledge Base ─────────────────────────────────────────────────────────────

def list_kb_entries() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT kb.*, u.name as created_by_name
            FROM knowledge_base kb
            LEFT JOIN users u ON kb.created_by = u.id
            ORDER BY kb.updated_at DESC
        """)
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_kb_entry(entry_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM knowledge_base WHERE id=%s", (entry_id,))
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def add_kb_entry(question: str, answer: str, tags: str, created_by: int) -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            INSERT INTO knowledge_base (question, answer, tags, created_by)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, (question.strip(), answer.strip(), tags.strip() if tags else "", created_by))
        entry_id = cur.fetchone()["id"]
        conn.commit()
        return entry_id
    finally:
        put_db(conn)


def update_kb_entry(entry_id: int, question: str, answer: str, tags: str):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE knowledge_base SET question=%s, answer=%s, tags=%s, updated_at=NOW()
            WHERE id=%s
        """, (question.strip(), answer.strip(), tags.strip() if tags else "", entry_id))
        conn.commit()
    finally:
        put_db(conn)


def delete_kb_entry(entry_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM knowledge_base WHERE id=%s", (entry_id,))
        conn.commit()
    finally:
        put_db(conn)


_KB_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "about", "and", "or", "not", "what", "how", "when",
    "where", "who", "which", "why", "can", "your", "our", "my", "tell",
    "please", "give", "me", "us", "you", "its", "this", "that", "these",
    "those", "also",
}


def _kb_terms(query: str) -> list:
    """Extract significant terms from a query for KB matching."""
    words = [w.strip('.,?!;:"()[]') for w in query.lower().split()]
    return [w for w in words if len(w) > 2 and w not in _KB_STOP]


def search_kb(query: str, limit: int = 3) -> list:
    """Search knowledge base. Tries AND query first, falls back to OR with score filter."""
    terms = _kb_terms(query)
    if not terms:
        return []
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        rows = []
        # 1. Try strict AND — all keywords must appear in the KB entry
        and_q = " & ".join(terms)
        try:
            cur.execute("""
                SELECT id, question, answer, tags,
                       ts_rank(search_vector, to_tsquery('english', %s)) as score
                FROM knowledge_base
                WHERE search_vector @@ to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (and_q, and_q, limit))
            rows = cur.fetchall()
        except Exception:
            conn.rollback()
            rows = []

        if not rows:
            # 2. Fallback: OR query with score threshold
            or_q = " | ".join(terms)
            try:
                cur.execute("""
                    SELECT id, question, answer, tags,
                           ts_rank(search_vector, to_tsquery('english', %s)) as score
                    FROM knowledge_base
                    WHERE search_vector @@ to_tsquery('english', %s)
                      AND ts_rank(search_vector, to_tsquery('english', %s)) > 0.01
                    ORDER BY score DESC
                    LIMIT %s
                """, (or_q, or_q, or_q, limit))
                rows = cur.fetchall()
            except Exception:
                conn.rollback()
                rows = []

        return list(rows)
    finally:
        put_db(conn)


def add_session_answer(investor_session_id: int, question: str, answer: str) -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "INSERT INTO session_answers (investor_session_id, question, answer) VALUES (%s, %s, %s) RETURNING id",
            (investor_session_id, question, answer)
        )
        entry_id = cur.fetchone()["id"]
        conn.commit()
        return entry_id
    finally:
        put_db(conn)


def list_session_answers(investor_session_id: int) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM session_answers WHERE investor_session_id=%s ORDER BY created_at DESC",
            (investor_session_id,)
        )
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def delete_session_answer(answer_id: int, investor_session_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM session_answers WHERE id=%s AND investor_session_id=%s",
                    (answer_id, investor_session_id))
        conn.commit()
    finally:
        put_db(conn)


def search_session_answers(query: str, investor_session_id: int, limit: int = 1) -> list:
    """Search investor-provided session answers using same AND/OR strategy as search_kb."""
    terms = _kb_terms(query)
    if not terms:
        return []
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        rows = []
        and_q = " & ".join(terms)
        try:
            cur.execute("""
                SELECT id, question, answer,
                       ts_rank(search_vector, to_tsquery('english', %s)) as score
                FROM session_answers
                WHERE search_vector @@ to_tsquery('english', %s)
                  AND investor_session_id = %s
                ORDER BY score DESC
                LIMIT %s
            """, (and_q, and_q, investor_session_id, limit))
            rows = cur.fetchall()
        except Exception:
            conn.rollback()
            rows = []

        if not rows:
            or_q = " | ".join(terms)
            try:
                cur.execute("""
                    SELECT id, question, answer,
                           ts_rank(search_vector, to_tsquery('english', %s)) as score
                    FROM session_answers
                    WHERE search_vector @@ to_tsquery('english', %s)
                      AND investor_session_id = %s
                      AND ts_rank(search_vector, to_tsquery('english', %s)) > 0.01
                    ORDER BY score DESC
                    LIMIT %s
                """, (or_q, or_q, investor_session_id, or_q, limit))
                rows = cur.fetchall()
            except Exception:
                conn.rollback()
                rows = []

        return list(rows)
    finally:
        put_db(conn)


def bulk_import_kb(entries: list, created_by: int) -> int:
    """Import list of {question, answer, tags} dicts. Returns count imported."""
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        count = 0
        for e in entries:
            q = e.get("question", "").strip()
            a = e.get("answer", "").strip()
            if not q or not a:
                continue
            cur.execute(
                "INSERT INTO knowledge_base (question, answer, tags, created_by) VALUES (%s, %s, %s, %s)",
                (q, a, e.get("tags", ""), created_by)
            )
            count += 1
        conn.commit()
        return count
    finally:
        put_db(conn)


def get_recent_questions(limit: int = 20) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT m.content as question, m.themes, m.created_at,
                   c.title as conversation_title,
                   s.investor_name
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            LEFT JOIN investor_sessions s ON c.investor_session_id = s.id
            WHERE m.role = 'user'
            ORDER BY m.created_at DESC
            LIMIT %s
        """, (limit,))
        results = []
        for r in cur.fetchall():
            d = dict(r)
            d["themes"] = json.loads(d["themes"]) if d["themes"] else []
            results.append(d)
        return results
    finally:
        put_db(conn)


# ── LLM Key Management ─────────────────────────────────────────────────────────

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


def get_env_key_sentinel_id() -> int:
    """Returns the ID of the sentinel row used to track env-var key usage."""
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id FROM llm_keys WHERE api_key_hint='env' LIMIT 1")
        row = cur.fetchone()
        return row["id"] if row else 1
    finally:
        put_db(conn)


def list_llm_keys() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Exclude the internal sentinel row from the admin UI
        cur.execute(
            "SELECT * FROM llm_keys WHERE api_key_hint != 'env' ORDER BY priority ASC, id ASC"
        )
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_active_llm_keys() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM llm_keys WHERE is_active=1 AND api_key_hint != 'env' ORDER BY priority ASC, id ASC"
        )
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_llm_key(key_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM llm_keys WHERE id=%s", (key_id,))
        row = cur.fetchone()
        return _row(dict(row)) if row else None
    finally:
        put_db(conn)


def add_llm_key(name: str, provider: str, model: str, api_key_enc: str, hint: str,
                priority: int = None, base_url: str = "") -> int:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if priority is None:
            cur.execute("SELECT MAX(priority) FROM llm_keys")
            row = cur.fetchone()
            max_p = list(row.values())[0]
            priority = (max_p or 0) + 1
        cur.execute(
            "INSERT INTO llm_keys (name, provider, model, api_key_enc, api_key_hint, priority, base_url) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (name, provider, model, api_key_enc, hint, priority, base_url or "")
        )
        rid = cur.fetchone()["id"]
        conn.commit()
        return rid
    finally:
        put_db(conn)


def update_llm_key(key_id: int, **kwargs):
    if not kwargs:
        return
    allowed = {"name", "model", "priority", "is_active", "api_key_enc", "api_key_hint", "base_url"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k}=%s" for k in fields)
    values = list(fields.values()) + [key_id]
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(f"UPDATE llm_keys SET {set_clause} WHERE id=%s", values)
        conn.commit()
    finally:
        put_db(conn)


def delete_llm_key(key_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM llm_keys WHERE id=%s", (key_id,))
        conn.commit()
    finally:
        put_db(conn)


def move_llm_key_priority(key_id: int, direction: str):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT priority FROM llm_keys WHERE id=%s", (key_id,))
        row = cur.fetchone()
        if not row:
            return
        cur_priority = row["priority"]
        if direction == "up":
            cur.execute(
                "SELECT id, priority FROM llm_keys WHERE priority < %s ORDER BY priority DESC LIMIT 1",
                (cur_priority,)
            )
        else:
            cur.execute(
                "SELECT id, priority FROM llm_keys WHERE priority > %s ORDER BY priority ASC LIMIT 1",
                (cur_priority,)
            )
        neighbor = cur.fetchone()
        if neighbor:
            cur.execute("UPDATE llm_keys SET priority=%s WHERE id=%s",
                        (neighbor["priority"], key_id))
            cur.execute("UPDATE llm_keys SET priority=%s WHERE id=%s",
                        (cur_priority, neighbor["id"]))
            conn.commit()
    finally:
        put_db(conn)


def log_llm_usage(key_id: int, provider: str, model: str, input_tokens: int, output_tokens: int):
    rates = COST_RATES.get(model, {"input": 0, "output": 0})
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO llm_usage (llm_key_id, provider, model, input_tokens, output_tokens, cost_usd) VALUES (%s, %s, %s, %s, %s, %s)",
            (key_id, provider, model, input_tokens, output_tokens, cost)
        )
        conn.commit()
    finally:
        put_db(conn)


def get_llm_usage_stats() -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT
                k.name,
                COALESCE(u.model, k.model)         AS model,
                COALESCE(SUM(u.input_tokens), 0)   AS total_input,
                COALESCE(SUM(u.output_tokens), 0)  AS total_output,
                COALESCE(SUM(u.cost_usd), 0)       AS total_cost,
                COUNT(u.id)                         AS requests,
                MAX(u.created_at)                   AS last_used
            FROM llm_keys k
            LEFT JOIN llm_usage u ON u.llm_key_id = k.id
            GROUP BY k.id, k.name, k.model, k.priority, u.model
            ORDER BY k.priority ASC, k.id ASC
        """)
        rows = cur.fetchall()
        # Only return rows that have had usage
        return [_row(dict(r)) for r in rows if r["requests"] > 0]
    finally:
        put_db(conn)


# ── Agent Marketplace ──────────────────────────────────────────────────────────

def _parse_agent(d: dict) -> dict:
    """Parse JSON fields on a marketplace_agents row."""
    if isinstance(d.get("tools"), str):
        try:
            d["tools"] = json.loads(d["tools"])
        except Exception:
            d["tools"] = []
    if not isinstance(d.get("tools"), list):
        d["tools"] = []
    if d.get("knowledge") is None:
        d["knowledge"] = ""
    return d


def list_marketplace_agents(category: str = None) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if category:
            cur.execute("""
                SELECT * FROM marketplace_agents WHERE is_active=1 AND category=%s
                ORDER BY category, name
            """, (category,))
        else:
            cur.execute("SELECT * FROM marketplace_agents WHERE is_active=1 ORDER BY category, name")
        return [_parse_agent(_row(dict(r))) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_marketplace_agent(agent_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM marketplace_agents WHERE id=%s", (agent_id,))
        row = cur.fetchone()
        return _parse_agent(_row(dict(row))) if row else None
    finally:
        put_db(conn)


def assign_agent_to_investor(agent_id: int, investor_session_id: int, assigned_by: int) -> bool:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO agent_assignments (agent_id, investor_session_id, assigned_by)
            VALUES (%s, %s, %s)
            ON CONFLICT (agent_id, investor_session_id) DO NOTHING
        """, (agent_id, investor_session_id, assigned_by))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        put_db(conn)


def unassign_agent_from_investor(agent_id: int, investor_session_id: int) -> bool:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM agent_assignments WHERE agent_id=%s AND investor_session_id=%s
        """, (agent_id, investor_session_id))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        put_db(conn)


def get_assigned_agents(investor_session_id: int) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT ma.*, aa.assigned_at
            FROM marketplace_agents ma
            JOIN agent_assignments aa ON aa.agent_id = ma.id
            WHERE aa.investor_session_id=%s AND ma.is_active=1
            ORDER BY ma.category, ma.name
        """, (investor_session_id,))
        return [_parse_agent(_row(dict(r))) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_investor_agent_ids(investor_session_id: int) -> set:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT agent_id FROM agent_assignments WHERE investor_session_id=%s", (investor_session_id,))
        return {r[0] for r in cur.fetchall()}
    finally:
        put_db(conn)


def list_marketplace_categories() -> list:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT category FROM marketplace_agents WHERE is_active=1 ORDER BY category")
        return [r[0] for r in cur.fetchall()]
    finally:
        put_db(conn)


# ── Agent Memory ───────────────────────────────────────────────────────────────

def get_agent_memory(agent_id: int, investor_session_id: int, limit: int = 20) -> list:
    """Load this agent's memories for this investor (most recent first)."""
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT memory_type, content, created_at
            FROM agent_memory
            WHERE agent_id=%s AND investor_session_id=%s
            ORDER BY created_at DESC
            LIMIT %s
        """, (agent_id, investor_session_id, limit))
        return [_row(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def add_agent_memory(agent_id: int, investor_session_id: int,
                     memory_type: str, content: str):
    """Store a new memory for this agent+investor pair."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO agent_memory (agent_id, investor_session_id, memory_type, content)
            VALUES (%s, %s, %s, %s)
        """, (agent_id, investor_session_id, memory_type, content))
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        put_db(conn)


def clear_agent_memory(agent_id: int, investor_session_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM agent_memory WHERE agent_id=%s AND investor_session_id=%s",
                    (agent_id, investor_session_id))
        conn.commit()
    finally:
        put_db(conn)


# ── Custom Agent Builder ───────────────────────────────────────────────────────

def _gen_api_key() -> str:
    import secrets
    return "cag_" + secrets.token_urlsafe(32)


def create_custom_agent(name: str, description: str, icon: str, system_prompt: str,
                        user_prompt: str, input_type: str, output_type: str,
                        output_webhook_url: str, output_webhook_secret: str,
                        tools: list, created_by: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        api_key = _gen_api_key()
        cur.execute("""
            INSERT INTO custom_agents
              (name, description, icon, system_prompt, user_prompt, input_type, output_type,
               output_webhook_url, output_webhook_secret, tools, api_key, created_by)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING *
        """, (name, description, icon, system_prompt, user_prompt, input_type, output_type,
              output_webhook_url, output_webhook_secret, json.dumps(tools), api_key, created_by))
        row = cur.fetchone()
        conn.commit()
        return _parse_custom_agent(dict(row))
    finally:
        put_db(conn)


def update_custom_agent(agent_id: int, name: str, description: str, icon: str,
                        system_prompt: str, user_prompt: str, input_type: str,
                        output_type: str, output_webhook_url: str,
                        output_webhook_secret: str, tools: list) -> bool:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE custom_agents SET
              name=%s, description=%s, icon=%s, system_prompt=%s, user_prompt=%s,
              input_type=%s, output_type=%s, output_webhook_url=%s,
              output_webhook_secret=%s, tools=%s
            WHERE id=%s
        """, (name, description, icon, system_prompt, user_prompt, input_type, output_type,
              output_webhook_url, output_webhook_secret, json.dumps(tools), agent_id))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        put_db(conn)


def delete_custom_agent(agent_id: int) -> bool:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM custom_agents WHERE id=%s", (agent_id,))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        put_db(conn)


def list_custom_agents(created_by: int = None) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if created_by:
            cur.execute("SELECT * FROM custom_agents WHERE created_by=%s ORDER BY created_at DESC", (created_by,))
        else:
            cur.execute("SELECT * FROM custom_agents ORDER BY created_at DESC")
        return [_parse_custom_agent(dict(r)) for r in cur.fetchall()]
    finally:
        put_db(conn)


def get_custom_agent(agent_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM custom_agents WHERE id=%s", (agent_id,))
        row = cur.fetchone()
        return _parse_custom_agent(dict(row)) if row else None
    finally:
        put_db(conn)


def get_custom_agent_by_key(api_key: str) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM custom_agents WHERE api_key=%s AND is_active=1", (api_key,))
        row = cur.fetchone()
        return _parse_custom_agent(dict(row)) if row else None
    finally:
        put_db(conn)


def _parse_custom_agent(d: dict) -> dict:
    if isinstance(d.get("tools"), str):
        try:
            d["tools"] = json.loads(d["tools"])
        except Exception:
            d["tools"] = []
    if not isinstance(d.get("tools"), list):
        d["tools"] = []
    return d


# ── Agent Schedules ──────────────────────────────────────────────────────────

def _compute_next_run(schedule_type: str, interval_minutes: int,
                      daily_time: str, weekly_day: int) -> str:
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    if schedule_type == 'interval':
        nxt = now + timedelta(minutes=max(1, interval_minutes))
    elif schedule_type == 'daily':
        try:
            h, m = [int(x) for x in (daily_time or '09:00').split(':')]
        except Exception:
            h, m = 9, 0
        nxt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
    elif schedule_type == 'weekly':
        try:
            h, m = [int(x) for x in (daily_time or '09:00').split(':')]
        except Exception:
            h, m = 9, 0
        target_dow = int(weekly_day or 1) % 7  # 0=Mon
        days_ahead = (target_dow - now.weekday()) % 7
        nxt = (now + timedelta(days=days_ahead)).replace(hour=h, minute=m, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(weeks=1)
    else:
        nxt = now + timedelta(hours=1)
    return nxt.strftime('%Y-%m-%d %H:%M:%S')


def create_agent_schedule(agent_id: int, name: str, input_text: str,
                          schedule_type: str, interval_minutes: int,
                          daily_time: str, weekly_day: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        next_run = _compute_next_run(schedule_type, interval_minutes, daily_time, weekly_day)
        cur.execute("""
            INSERT INTO agent_schedules
              (agent_id, name, input_text, schedule_type, interval_minutes,
               daily_time, weekly_day, next_run_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *
        """, (agent_id, name, input_text, schedule_type, interval_minutes,
              daily_time, weekly_day, next_run))
        row = dict(cur.fetchone())
        conn.commit()
        return row
    finally:
        put_db(conn)


def list_agent_schedules(agent_id: int) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM agent_schedules WHERE agent_id=%s ORDER BY created_at DESC",
                    (agent_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        put_db(conn)


def toggle_agent_schedule(schedule_id: int) -> bool:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE agent_schedules SET is_active = CASE WHEN is_active=1 THEN 0 ELSE 1 END
            WHERE id=%s
        """, (schedule_id,))
        conn.commit()
        return True
    finally:
        put_db(conn)


def delete_agent_schedule(schedule_id: int) -> bool:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM agent_schedules WHERE id=%s", (schedule_id,))
        conn.commit()
        return True
    finally:
        put_db(conn)


def get_due_schedules() -> list:
    """Return active schedules whose next_run_at is due, atomically claiming them."""
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT s.*, ca.system_prompt, ca.user_prompt, ca.tools,
                   ca.output_type, ca.output_webhook_url, ca.output_webhook_secret, ca.api_key
            FROM agent_schedules s
            JOIN custom_agents ca ON ca.id = s.agent_id
            WHERE s.is_active = 1 AND s.next_run_at <= NOW()
        """)
        rows = [dict(r) for r in cur.fetchall()]
        # Immediately push next_run_at forward to prevent double-execution
        for row in rows:
            next_run = _compute_next_run(
                row['schedule_type'], row.get('interval_minutes', 60),
                row.get('daily_time', '09:00'), row.get('weekly_day', 1))
            cur.execute("""
                UPDATE agent_schedules
                SET last_run_at=NOW(), next_run_at=%s, run_count=run_count+1
                WHERE id=%s
            """, (next_run, row['id']))
        conn.commit()
        return rows
    finally:
        put_db(conn)


def save_custom_agent_run(agent_id: int, input_text: str, output_text: str,
                          sources: list, confidence: str, input_src: str) -> int:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO custom_agent_runs (agent_id, input_text, output_text, sources, confidence, input_src, status)
            VALUES (%s,%s,%s,%s,%s,%s,'done') RETURNING id
        """, (agent_id, input_text, output_text, json.dumps(sources), confidence, input_src))
        row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    finally:
        put_db(conn)


def get_custom_agent_runs(agent_id: int, limit: int = 50) -> list:
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT * FROM custom_agent_runs WHERE agent_id=%s ORDER BY created_at DESC LIMIT %s
        """, (agent_id, limit))
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            if isinstance(d.get("sources"), str):
                try:
                    d["sources"] = json.loads(d["sources"])
                except Exception:
                    d["sources"] = []
            rows.append(d)
        return rows
    finally:
        put_db(conn)
