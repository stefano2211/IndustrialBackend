"""Encryption utilities for sensitive fields (e.g. DB connection strings)."""

import base64
import hashlib
from functools import lru_cache
from cryptography.fernet import Fernet
from app.core.config import settings


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    """Derives a 32-byte Fernet key from the app SECRET_KEY via SHA-256 (cached)."""
    raw_key = settings.secret_key.encode("utf-8")
    digest = hashlib.sha256(raw_key).digest()
    b64_key = base64.urlsafe_b64encode(digest)
    return Fernet(b64_key)


def encrypt(plain_text: str) -> str:
    """Encrypts a string and returns a base64-safe ciphertext string."""
    f = _get_fernet()
    token = f.encrypt(plain_text.encode("utf-8"))
    return token.decode("utf-8")


def decrypt(cipher_text: str) -> str:
    """Decrypts a Fernet-encrypted string back to plain text."""
    f = _get_fernet()
    plain = f.decrypt(cipher_text.encode("utf-8"))
    return plain.decode("utf-8")
