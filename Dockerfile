# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — OBFUSCATE
# Runs PyArmor to encrypt/obfuscate all Python source files.
# This stage is discarded — the final image never contains source .py files.
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS obfuscator

WORKDIR /build

# Install PyArmor
RUN pip install --no-cache-dir pyarmor==8.*

# Copy only the Python source files we want to obfuscate
COPY agent.py app.py company_research.py database.py \
     document_processor.py fund_research.py llm_crypto.py \
     license_manager.py plugins.py ./

# Obfuscate all .py files → /dist
# --pack: bundle runtime inside each file (no separate runtime dir needed)
RUN pyarmor gen --output /dist \
    agent.py app.py company_research.py database.py \
    document_processor.py fund_research.py llm_crypto.py \
    license_manager.py plugins.py


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — PRODUCTION IMAGE
# Only obfuscated files, assets, and dependencies. Zero source code.
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS production

LABEL maintainer="your@email.com"
LABEL description="DDQ Platform — Protected Distribution"

# System deps for Playwright, pdfplumber, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc wget gnupg curl ca-certificates \
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir playwright yfinance pdfplumber \
    && playwright install chromium --with-deps

# Copy obfuscated Python files from Stage 1
COPY --from=obfuscator /dist/ ./

# Copy non-Python assets (these don't need obfuscation)
COPY templates/   ./templates/
COPY static/      ./static/
COPY prompts/     ./prompts/

# License directory — clients mount their license.lic here
RUN mkdir -p /app/license
VOLUME ["/app/license"]

# Data directories
RUN mkdir -p /app/uploads /app/company_docs /app/documents
VOLUME ["/app/uploads", "/app/company_docs", "/app/documents"]

# Environment defaults (clients override via docker-compose.yml)
ENV FLASK_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    LICENSE_FILE=/app/license/license.lic \
    WORKERS=2

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with gunicorn — source code is obfuscated, not readable
CMD gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers ${WORKERS} \
    --worker-class sync \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    "app:app"
