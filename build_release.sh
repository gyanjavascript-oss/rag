#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_release.sh — Build a protected Docker image for client distribution
#
# Usage:
#   ./build_release.sh [version]
#   ./build_release.sh 1.2.0
#
# What it does:
#   1. Verifies private key exists (so you don't ship an unprotected build)
#   2. Embeds the public key into license_manager.py
#   3. Builds the Docker image (obfuscation happens inside Docker build)
#   4. Exports the image as a .tar.gz for offline distribution
# ─────────────────────────────────────────────────────────────────────────────
set -e

VERSION=${1:-"1.0.0"}
IMAGE_NAME="ddq-platform"
FULL_TAG="${IMAGE_NAME}:${VERSION}"
LATEST_TAG="${IMAGE_NAME}:latest"

echo "═══════════════════════════════════════════"
echo "  Building DDQ Platform v${VERSION}"
echo "  Protected release — source code hidden"
echo "═══════════════════════════════════════════"

# ── 1. Check private key exists ──────────────────────────────────────────────
if [ ! -f "keys/private.pem" ]; then
  echo ""
  echo "ERROR: keys/private.pem not found."
  echo "Run first: python generate_license.py --init"
  exit 1
fi

if [ ! -f "keys/public.pem" ]; then
  echo "ERROR: keys/public.pem not found."
  exit 1
fi

echo "✓ Keys found"

# ── 2. Embed public key into license_manager.py ──────────────────────────────
PUBLIC_KEY=$(cat keys/public.pem)

# Escape for sed: replace newlines with \n
ESCAPED_KEY=$(echo "$PUBLIC_KEY" | python3 -c "import sys; print(sys.stdin.read().rstrip().replace('\\n', '\\\\n'))")

# Replace the _FALLBACK_PUBLIC_KEY in license_manager.py
python3 - <<EOF
import re

with open('license_manager.py', 'r') as f:
    content = f.read()

pub_key = open('keys/public.pem').read().strip()

# Replace between triple-quoted markers
new_content = re.sub(
    r'(_FALLBACK_PUBLIC_KEY\s*=\s*""").*?(""")',
    f'_FALLBACK_PUBLIC_KEY = """\n{pub_key}\n"""',
    content,
    flags=re.DOTALL
)

with open('license_manager.py', 'w') as f:
    f.write(new_content)

print("✓ Public key embedded into license_manager.py")
EOF

# ── 3. Build Docker image ─────────────────────────────────────────────────────
echo ""
echo "Building Docker image: ${FULL_TAG}"
echo "(PyArmor obfuscation runs inside Docker — source will not be in final image)"
echo ""

docker build \
  --no-cache \
  --build-arg BUILD_VERSION="${VERSION}" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "${FULL_TAG}" \
  -t "${LATEST_TAG}" \
  .

echo ""
echo "✓ Image built: ${FULL_TAG}"

# ── 4. Verify source is NOT in image ─────────────────────────────────────────
echo ""
echo "Verifying source code is not accessible in image..."

# Try to find readable .py source files in the image
PY_READABLE=$(docker run --rm --entrypoint sh "${FULL_TAG}" -c \
  "find /app -name '*.py' -not -path '*/pyarmor*' 2>/dev/null | head -5" 2>/dev/null || echo "")

if [ -n "$PY_READABLE" ]; then
  echo "⚠  WARNING: Some .py files found in image:"
  echo "$PY_READABLE"
  echo "   These are PyArmor runtime files — original source is still obfuscated."
else
  echo "✓ No readable source .py files in final image"
fi

# ── 5. Export image for distribution ─────────────────────────────────────────
echo ""
EXPORT_FILE="${IMAGE_NAME}-${VERSION}.tar.gz"
echo "Exporting image to ${EXPORT_FILE}..."
docker save "${FULL_TAG}" | gzip > "${EXPORT_FILE}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✓ Release complete!"
echo ""
echo "  Image tag:    ${FULL_TAG}"
echo "  Export file:  ${EXPORT_FILE}  ($(du -sh ${EXPORT_FILE} | cut -f1))"
echo ""
echo "  To ship to a client:"
echo "    1. Send them:  ${EXPORT_FILE}  +  docker-compose.yml"
echo "    2. Generate a license: python generate_license.py --create"
echo "    3. Send them the .lic file"
echo ""
echo "  Client loads the image:"
echo "    docker load < ${EXPORT_FILE}"
echo "    # place license.lic in ./license/"
echo "    docker-compose up -d"
echo "═══════════════════════════════════════════════════════════"
