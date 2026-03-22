# DDQ Platform

AI-powered Due Diligence Questionnaire platform for fund managers. Upload your fund documents (LPA, presentations, policies, memos) and let the AI answer investor questions with source citations and draft responses — all tracked for analytics.

---

## Features

- **Document-grounded Q&A** — answers sourced from your uploaded fund documents
- **Source citations** — every answer references the exact document and section
- **Draft responses** — professional, copy-ready email drafts for each answer
- **Investor sessions** — upload investor-specific docs (entity structure, tax forms) for cross-referencing
- **Analytics** — track question themes, topic frequency, and documentation gaps
- **Team management** — admin and analyst roles

---

## Tech Stack

- **Backend** — Python / Flask
- **AI** — OpenAI GPT-4o with function calling
- **Database** — SQLite with FTS5 full-text search
- **Frontend** — HTML + Vanilla JavaScript

---

## Setup

### 1. Clone / navigate to the project

```bash
cd /path/to/RAgdata
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set:

```env
OPENAI_API_KEY=sk-...        # Your OpenAI API key
SECRET_KEY=change-me-random  # Any random string for Flask sessions
FUND_NAME=Your Fund Name     # Displayed throughout the UI
```

### 5. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Default Login

| Email | Password | Role |
|---|---|---|
| admin@fund.com | admin123 | Admin |

Change this immediately via **Team → Add Member** after first login.

---

## Loading Fund Documents

### Option A — Upload via UI (recommended)

1. Log in and go to **Fund Documents**
2. Click **Upload Document**
3. Select the file type (LPA, Presentation, Policy, Memo, Tax)
4. Upload — the document is chunked and indexed automatically

### Option B — Bulk ingest from folder

1. Drop your files into the `/documents` folder
2. Log in as admin
3. Go to **Fund Documents** and click **bulk ingest**

**Supported formats:** PDF, DOCX, DOC, TXT, Markdown

---

## Project Structure

```
RAgdata/
├── app.py                   # Flask routes and SSE streaming
├── agent.py                 # OpenAI GPT-4o RAG agent with tool use
├── database.py              # SQLite schema and query helpers
├── document_processor.py    # File parsing and text chunking
├── requirements.txt
├── .env                     # Your config (not committed)
├── .env.example             # Config template
├── documents/               # Drop fund docs here for bulk ingest
├── uploads/                 # Runtime upload storage (auto-created)
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   ├── chat.html            # Q&A interface with streaming
│   ├── documents.html
│   ├── investors.html
│   ├── investor_detail.html
│   ├── analytics.html
│   └── team.html
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## Deactivating the Virtual Environment

When you're done:

```bash
deactivate
```

---

## Restarting Later

```bash
cd /path/to/RAgdata
source venv/bin/activate      # macOS/Linux
# or: venv\Scripts\activate   # Windows
python app.py
```
