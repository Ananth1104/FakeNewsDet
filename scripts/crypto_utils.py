from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import base64

# Use a fixed 32-byte key for AES-256 encryption (Must be kept secret)
secret_key_env = "mysecretaeskey123456789012345678"  # Default 32-byte key
SECRET_KEY = secret_key_env.encode("utf-8")
if len(SECRET_KEY) not in [16, 24, 32]:  # AES only supports 16, 24, or 32-byte keys
    raise ValueError("AES_SECRET_KEY must be exactly 16, 24, or 32 bytes.")

def pad(text):
    """Pad the text using PKCS7 to make it a multiple of 16 bytes."""
    padding_size = 16 - (len(text) % 16)
    return text + (chr(padding_size) * padding_size)

def unpad(text):
    """Remove PKCS7 padding from the decrypted text."""
    return text[:-ord(text[-1])]

def encrypt_query(query):
    """Encrypt a search query using AES encryption (CBC mode)."""
    iv = os.urandom(16)  # Generate a random IV for each encryption
    cipher = Cipher(algorithms.AES(SECRET_KEY), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_bytes = encryptor.update(pad(query).encode("utf-8")) + encryptor.finalize()
    return base64.b64encode(iv + encrypted_bytes).decode("utf-8")

def decrypt_query(encrypted_query):
    """Decrypt an AES-encrypted search query (CBC mode)."""
    encrypted_data = base64.b64decode(encrypted_query)
    iv = encrypted_data[:16]  # Extract IV from the encrypted data
    cipher = Cipher(algorithms.AES(SECRET_KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_bytes = decryptor.update(encrypted_data[16:]) + decryptor.finalize()
    return unpad(decrypted_bytes.decode("utf-8"))
