import pytest
from cryptography.fernet import Fernet

from fuse.crypto import SecretBox


def test_round_trip():
    box = SecretBox(Fernet.generate_key().decode())
    secret = '{"refresh_token": "1//abc"}'
    ciphertext = box.encrypt(secret)
    assert b"refresh_token" not in ciphertext  # not readable when stored
    assert box.decrypt(ciphertext) == secret


def test_wrong_key_fails_loudly():
    ciphertext = SecretBox(Fernet.generate_key().decode()).encrypt("secret")
    with pytest.raises(RuntimeError, match="wrong ENCRYPTION_KEY"):
        SecretBox(Fernet.generate_key().decode()).decrypt(ciphertext)


def test_missing_key_explains_how_to_make_one():
    with pytest.raises(RuntimeError, match="generate_key"):
        SecretBox("")
