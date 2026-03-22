"""
Document ingestion: parse files, chunk text, build FTS index.
Supports .txt, .md, .pdf, .docx
"""
import os
import re
from database import get_db

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200


# ── File readers ──────────────────────────────────────────────────────────────

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf(path: str) -> tuple[str, list[tuple[int, str]]]:
    """Returns (full_text, [(page_num, page_text), ...])"""
    try:
        import fitz
        doc = fitz.open(path)
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
        return "\n\n".join(t for _, t in pages), pages
    except ImportError:
        text = f"[PDF support requires PyMuPDF. File: {os.path.basename(path)}]"
        return text, [(1, text)]
    except Exception as e:
        text = f"[Error reading PDF: {e}]"
        return text, [(1, text)]


def _read_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        return f"[DOCX support requires python-docx. File: {os.path.basename(path)}]"
    except Exception as e:
        return f"[Error reading DOCX: {e}]"


def read_file(path: str) -> tuple:
    """Returns (content, file_type, pages_or_None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        content, pages = _read_pdf(path)
        return content, "pdf", pages
    elif ext in (".docx", ".doc"):
        return _read_docx(path), "docx", None
    elif ext == ".md":
        return _read_txt(path), "markdown", None
    else:
        return _read_txt(path), "text", None


# ── Chunking ──────────────────────────────────────────────────────────────────

def _detect_section(text: str) -> str:
    """Try to detect a section/heading from the start of a chunk."""
    lines = text.strip().splitlines()
    for line in lines[:3]:
        line = line.strip()
        if re.match(r'^(section|article|clause|schedule|appendix|part)\s+[\d\.]+', line, re.I):
            return line[:80]
        if re.match(r'^[\d]+[\.\)]\s+[A-Z]', line):
            return line[:80]
    return None


def chunk_text_simple(text: str, size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk text into overlapping segments. Returns list of {text, section_ref}."""
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]

        # Prefer breaking at double newline
        if end < len(text):
            bp = chunk.rfind('\n\n')
            if bp > size // 3:
                end = start + bp
                chunk = text[start:end]

        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "section_ref": _detect_section(chunk),
            })

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def chunk_by_pages(pages: list[tuple[int, str]],
                   size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk with page references for PDFs."""
    all_chunks = []
    for page_num, page_text in pages:
        page_chunks = chunk_text_simple(page_text, size, overlap)
        for c in page_chunks:
            c["page_ref"] = f"p.{page_num}"
            all_chunks.append(c)
    return all_chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_fund_document(filepath: str, name: str, doc_type: str,
                          uploaded_by: int) -> int:
    """
    Parse file, store in DB, build FTS index.
    Returns fund_documents.id
    """
    content, file_type, pages = read_file(filepath)

    if pages:
        chunks = chunk_by_pages(pages)
    else:
        chunks = chunk_text_simple(content)
        for c in chunks:
            c["page_ref"] = None

    conn = get_db()
    c = conn.cursor()

    # Insert document record
    c.execute("""
        INSERT INTO fund_documents (name, doc_type, filepath, content, uploaded_by)
        VALUES (?, ?, ?, ?, ?)
    """, (name, doc_type, filepath, content[:10000], uploaded_by))
    doc_id = c.lastrowid

    # Insert chunks and update FTS
    for i, chunk in enumerate(chunks):
        c.execute("""
            INSERT INTO document_chunks
                (document_id, chunk_text, chunk_index, page_ref, section_ref)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, chunk["text"], i,
              chunk.get("page_ref"), chunk.get("section_ref")))
        chunk_rowid = c.lastrowid
        conn.execute(
            "INSERT INTO chunks_fts(rowid, chunk_text) VALUES (?, ?)",
            (chunk_rowid, chunk["text"])
        )

    conn.commit()
    conn.close()
    return doc_id


def ingest_investor_document(filepath: str, name: str, doc_type: str,
                              investor_session_id: int) -> int:
    """Ingest an investor-specific document. Returns investor_documents.id"""
    content, file_type, pages = read_file(filepath)

    if pages:
        chunks = chunk_by_pages(pages)
    else:
        chunks = chunk_text_simple(content)
        for c in chunks:
            c["page_ref"] = None

    conn = get_db()
    c = conn.cursor()

    c.execute("""
        INSERT INTO investor_documents
            (investor_session_id, name, filepath, doc_type, content)
        VALUES (?, ?, ?, ?, ?)
    """, (investor_session_id, name, filepath, doc_type, content[:10000]))
    inv_doc_id = c.lastrowid

    for i, chunk in enumerate(chunks):
        c.execute("""
            INSERT INTO investor_doc_chunks (investor_doc_id, chunk_text, chunk_index)
            VALUES (?, ?, ?)
        """, (inv_doc_id, chunk["text"], i))
        chunk_rowid = c.lastrowid
        conn.execute(
            "INSERT INTO inv_chunks_fts(rowid, chunk_text) VALUES (?, ?)",
            (chunk_rowid, chunk["text"])
        )

    conn.commit()
    conn.close()
    return inv_doc_id


def ingest_from_folder(folder_path: str, uploaded_by: int = 1) -> list[dict]:
    """Bulk ingest all supported files from a folder."""
    supported = {".txt", ".md", ".pdf", ".docx", ".doc"}
    type_map = {
        "lpa": "LPA",
        "presentation": "Presentation",
        "policy": "Policy",
        "memo": "Memo",
        "tax": "Tax",
    }
    results = []
    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported:
            continue
        lower = fname.lower()
        doc_type = "Other"
        for key, val in type_map.items():
            if key in lower:
                doc_type = val
                break

        filepath = os.path.join(folder_path, fname)
        try:
            doc_id = ingest_fund_document(filepath, fname, doc_type, uploaded_by)
            results.append({"file": fname, "doc_id": doc_id, "status": "ok"})
        except Exception as e:
            results.append({"file": fname, "status": "error", "error": str(e)})

    return results
