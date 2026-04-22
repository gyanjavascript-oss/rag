#!/usr/bin/env python3
"""
DEVELOPER TOOL — Never include this file in the Docker image.
Run this locally to:
  1. Generate your RSA key pair (once): python generate_license.py --init
  2. Create a license for a client:    python generate_license.py --create
  3. Inspect a license file:           python generate_license.py --inspect license.lic
"""
import argparse, base64, json, os, sys, datetime
from pathlib import Path

KEYS_DIR     = Path(__file__).parent / "keys"
PRIVATE_KEY  = KEYS_DIR / "private.pem"
PUBLIC_KEY   = KEYS_DIR / "public.pem"


def _load_private_key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    if not PRIVATE_KEY.exists():
        print("ERROR: Private key not found. Run: python generate_license.py --init")
        sys.exit(1)
    with open(PRIVATE_KEY, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())


def cmd_init():
    """Generate RSA-2048 key pair. Run ONCE and keep private.pem safe."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    KEYS_DIR.mkdir(exist_ok=True)
    if PRIVATE_KEY.exists():
        ans = input("Keys already exist. Overwrite? All existing licenses will be INVALIDATED. [y/N]: ")
        if ans.lower() != "y":
            print("Aborted.")
            return

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    with open(PRIVATE_KEY, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()
        ))
    with open(PUBLIC_KEY, "wb") as f:
        f.write(key.public_key().private_bytes if False else
                key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo
                ))

    pub_pem = PUBLIC_KEY.read_text()
    print("✓ Key pair generated.")
    print(f"  Private key: {PRIVATE_KEY}  ← KEEP THIS SECRET, never commit/ship")
    print(f"  Public key:  {PUBLIC_KEY}")
    print()
    print("=" * 60)
    print("NEXT STEP: Open license_manager.py and replace _FALLBACK_PUBLIC_KEY with:")
    print("=" * 60)
    print(pub_pem)
    print("=" * 60)


def cmd_create():
    """Interactively create a license file for a client."""
    priv = _load_private_key()

    print("\n── Create License ──────────────────────────")
    customer = input("Customer name:        ").strip()
    email    = input("Customer email:       ").strip()
    tier     = input("Tier (standard/enterprise) [standard]: ").strip() or "standard"
    seats    = input("Number of seats [1]:  ").strip() or "1"
    days     = input("Valid for (days) [365]: ").strip() or "365"

    print("\nAvailable features: company_research, fund_research, agents, scheduler, marketplace")
    feat_input = input("Features (comma-separated, or 'all'): ").strip()
    all_features = ["company_research", "fund_research", "agents", "scheduler", "marketplace"]
    if feat_input.lower() == "all" or not feat_input:
        features = all_features
    else:
        features = [f.strip() for f in feat_input.split(",")]

    issued  = datetime.date.today().isoformat()
    expires = (datetime.date.today() + datetime.timedelta(days=int(days))).isoformat()

    payload = {
        "customer": customer,
        "email":    email,
        "tier":     tier,
        "seats":    int(seats),
        "features": features,
        "issued":   issued,
        "expires":  expires,
    }

    # Sign
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    signature = priv.sign(payload_bytes, padding.PKCS1v15(), hashes.SHA256())

    license_str = base64.b64encode(payload_bytes).decode() + "." + base64.b64encode(signature).decode()

    # Save
    safe_name = customer.lower().replace(" ", "_").replace("/", "_")
    out_file  = Path(f"license_{safe_name}_{expires}.lic")
    out_file.write_text(license_str)

    print(f"\n✓ License created: {out_file}")
    print(f"  Customer : {customer}")
    print(f"  Email    : {email}")
    print(f"  Tier     : {tier}")
    print(f"  Seats    : {seats}")
    print(f"  Features : {', '.join(features)}")
    print(f"  Issued   : {issued}")
    print(f"  Expires  : {expires}  ({days} days)")
    print(f"\nSend {out_file} to the client. They mount it at /app/license/license.lic")


def cmd_inspect(path: str):
    """Inspect and validate a license file without verifying signature."""
    import base64, json
    content = Path(path).read_text().strip()
    parts = content.split(".")
    if len(parts) != 2:
        print("ERROR: Not a valid license file.")
        return
    try:
        payload = json.loads(base64.b64decode(parts[0]).decode())
        print("License contents:")
        for k, v in payload.items():
            print(f"  {k:12}: {v}")
        # Check expiry
        expires = datetime.date.fromisoformat(payload.get("expires","9999-12-31"))
        days_left = (expires - datetime.date.today()).days
        if days_left < 0:
            print(f"\n  STATUS: EXPIRED ({-days_left} days ago)")
        else:
            print(f"\n  STATUS: VALID ({days_left} days remaining)")
    except Exception as e:
        print(f"ERROR: {e}")


def cmd_list():
    """List all .lic files in the current directory."""
    files = list(Path(".").glob("*.lic"))
    if not files:
        print("No .lic files found.")
        return
    for f in files:
        print(f"\n{f}:")
        cmd_inspect(str(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License management tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init",    action="store_true", help="Generate RSA key pair (run once)")
    group.add_argument("--create",  action="store_true", help="Create a new license for a client")
    group.add_argument("--inspect", metavar="FILE",      help="Inspect a .lic file")
    group.add_argument("--list",    action="store_true", help="List all .lic files")
    args = parser.parse_args()

    if args.init:    cmd_init()
    elif args.create: cmd_create()
    elif args.inspect: cmd_inspect(args.inspect)
    elif args.list:  cmd_list()
