"""
Encryption helpers for LLM API keys using Fernet (AES-128-CBC + HMAC).
Derives a stable Fernet key from the app's SECRET_KEY env var.

WARNING: If SECRET_KEY changes, all stored encrypted keys become unreadable.
"""
import os
import hashlib
import base64
from cryptography.fernet import Fernet


def _fernet() -> Fernet:
    secret = os.getenv("SECRET_KEY", "dev-secret-change-me")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt_key(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_key(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()
