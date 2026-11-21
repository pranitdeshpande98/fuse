"""Encrypt secrets before they are stored in the database.

Uses Fernet (from the `cryptography` library): authenticated encryption, so anything
tampered with or encrypted under a different key fails to decrypt instead of returning junk.
"""

from cryptography.fernet import Fernet, InvalidToken


class SecretBox:
    def __init__(self, key: str) -> None:
        if not key:
            raise RuntimeError(
                "ENCRYPTION_KEY is not set. Generate one with:\n"
                '  uv run python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            )
        self._fernet = Fernet(key.encode())

    def encrypt(self, plaintext: str) -> bytes:
        return self._fernet.encrypt(plaintext.encode())

    def decrypt(self, ciphertext: bytes) -> str:
        try:
            return self._fernet.decrypt(ciphertext).decode()
        except InvalidToken as exc:
            raise RuntimeError("Could not decrypt: wrong ENCRYPTION_KEY or corrupted data") from exc
