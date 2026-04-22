"""
License validation module.
The PUBLIC key is embedded here — only the developer holds the private key.
Clients cannot forge or modify licenses.
"""
import base64, json, os, datetime, hashlib, socket
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# ── Embedded public key (generated once, private key stays on dev machine) ───
# Replace this with the output of: python generate_license.py --init
_PUBLIC_KEY_PEM = os.getenv("LICENSE_PUBLIC_KEY", "")  # override via env for rotation

_FALLBACK_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
REPLACE_WITH_YOUR_PUBLIC_KEY_AFTER_RUNNING_generate_license.py_--init
-----END PUBLIC KEY-----"""


class LicenseError(Exception):
    pass


def _get_public_key():
    pem = _PUBLIC_KEY_PEM.strip() or _FALLBACK_PUBLIC_KEY.strip()
    return serialization.load_pem_public_key(pem.encode(), backend=default_backend())


def validate_license(license_path: str = None) -> dict:
    """
    Load and validate a .lic file. Returns the license payload dict.
    Raises LicenseError with a human-readable message on failure.
    """
    if license_path is None:
        license_path = os.environ.get("LICENSE_FILE", "/app/license/license.lic")

    if not os.path.exists(license_path):
        raise LicenseError(
            f"License file not found at {license_path}.\n"
            "Please place a valid license.lic file and restart."
        )

    try:
        with open(license_path, "r") as f:
            content = f.read().strip()
    except Exception as e:
        raise LicenseError(f"Could not read license file: {e}")

    # Format: base64(payload_json) . base64(signature)
    try:
        parts = content.split(".")
        if len(parts) != 2:
            raise ValueError("bad format")
        payload_b64, sig_b64 = parts
        payload_bytes = base64.b64decode(payload_b64)
        signature     = base64.b64decode(sig_b64)
    except Exception:
        raise LicenseError("License file is corrupted or tampered.")

    # Verify signature
    try:
        pub = _get_public_key()
        pub.verify(signature, payload_bytes, padding.PKCS1v15(), hashes.SHA256())
    except Exception:
        raise LicenseError("License signature is invalid. The file may have been modified.")

    # Decode payload
    try:
        payload = json.loads(payload_bytes.decode())
    except Exception:
        raise LicenseError("License payload is unreadable.")

    # Check expiry
    expires_str = payload.get("expires", "")
    if expires_str:
        try:
            expires = datetime.date.fromisoformat(expires_str)
            if datetime.date.today() > expires:
                raise LicenseError(
                    f"License expired on {expires_str}. "
                    "Please renew your license."
                )
        except LicenseError:
            raise
        except Exception:
            raise LicenseError(f"Invalid expiry date in license: {expires_str}")

    return payload


def get_license_info(license_path: str = None) -> dict:
    """Return license info dict, or error dict — never raises."""
    try:
        payload = validate_license(license_path)
        expires_str = payload.get("expires", "")
        days_left = None
        if expires_str:
            try:
                days_left = (datetime.date.fromisoformat(expires_str) - datetime.date.today()).days
            except Exception:
                pass
        return {
            "valid": True,
            "customer": payload.get("customer", ""),
            "email": payload.get("email", ""),
            "tier": payload.get("tier", "standard"),
            "seats": payload.get("seats", 1),
            "expires": expires_str,
            "days_left": days_left,
            "features": payload.get("features", []),
            "issued": payload.get("issued", ""),
        }
    except LicenseError as e:
        return {"valid": False, "error": str(e)}
    except Exception as e:
        return {"valid": False, "error": f"Unexpected error: {e}"}


def require_license(app=None):
    """
    Call this at Flask app startup. If the license is invalid, the app
    registers a catch-all route that shows a license error page.
    """
    info = get_license_info()
    if info["valid"]:
        print(f"[LICENSE] ✓ Valid — Customer: {info['customer']} | "
              f"Tier: {info['tier']} | Expires: {info['expires']} ({info['days_left']} days left)")
        # Warn if expiring soon
        if info["days_left"] is not None and info["days_left"] <= 14:
            print(f"[LICENSE] ⚠ WARNING: License expires in {info['days_left']} days. Please renew.")
        return info
    else:
        msg = info["error"]
        print(f"[LICENSE] ✗ INVALID — {msg}")
        if app is not None:
            _lock_app(app, msg)
        return info


def _lock_app(app, error_msg: str):
    """Replace all routes with a license error page."""
    from flask import Flask

    # Remove existing routes
    app.url_map._rules.clear()
    app.url_map._rules_by_endpoint.clear()
    app.view_functions.clear()

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def _license_error(path):
        from flask import make_response
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>License Required</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#0f172a;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}}
  .box{{background:#1e293b;border:1px solid #c9a84c;border-radius:12px;padding:2.5rem 3rem;max-width:520px;text-align:center}}
  h1{{color:#c9a84c;font-size:1.4rem;margin-bottom:.5rem}}
  p{{color:#94a3b8;font-size:.9rem;line-height:1.6}}
  code{{background:#0f172a;color:#f1f5f9;padding:.2rem .5rem;border-radius:4px;font-size:.82rem}}
</style></head>
<body><div class="box">
  <div style="font-size:2.5rem;margin-bottom:.75rem">🔐</div>
  <h1>License Required</h1>
  <p>{error_msg}</p>
  <p style="margin-top:1.5rem;font-size:.8rem;color:#64748b">
    Place your <code>license.lic</code> file in the license volume<br>
    and restart the container.<br><br>
    Contact <strong style="color:#c9a84c">support@yourcompany.com</strong> for a license.
  </p>
</div></body></html>"""
        return make_response(html, 402)
