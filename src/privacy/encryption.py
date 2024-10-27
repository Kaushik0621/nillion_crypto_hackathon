# src/privacy/encryption.py

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os
import config

class AESCipher:
    def __init__(self, key: bytes = config.AES_KEY):
        """Initialize AES cipher with a given key."""
        self.key = key
        self.block_size = 128  # AES block size in bits

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using AES encryption with PKCS7 padding."""
        iv = os.urandom(16)  # Initialization Vector (16 bytes for AES)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Apply PKCS7 padding to data
        padder = padding.PKCS7(self.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()

        # Encrypt padded data and prepend the IV
        encrypted_data = iv + encryptor.update(padded_data) + encryptor.finalize()
        return encrypted_data

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES decryption with PKCS7 unpadding."""
        iv = encrypted_data[:16]  # Extract the IV from the beginning
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        # Decrypt the data and remove padding
        padded_data = decryptor.update(encrypted_data[16:]) + decryptor.finalize()
        unpadder = padding.PKCS7(self.block_size).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        return data

# Instantiate AESCipher with the persistent key
aes_cipher = AESCipher()
