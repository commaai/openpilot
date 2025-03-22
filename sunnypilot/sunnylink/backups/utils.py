"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import base64
import hashlib
import zlib
import re
import json
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from sunnypilot.sunnylink.backups.AESCipher import AESCipher
from openpilot.system.hardware.hw import Paths


class KeyDerivation:
  @staticmethod
  def _load_key(file_path: str) -> bytes:
    with open(file_path, 'rb') as f:
      return f.read()

  @staticmethod
  def derive_aes_key_iv_from_rsa(key_path: str, use_aes_256: bool) -> tuple[bytes, bytes]:
    rsa_key_pem: bytes = KeyDerivation._load_key(key_path)
    key_plain = rsa_key_pem.decode(errors="ignore")

    if "private" in key_plain.lower():
      private_key = serialization.load_pem_private_key(rsa_key_pem, password=None, backend=default_backend())
      if not isinstance(private_key, rsa.RSAPrivateKey):
        raise ValueError("Invalid RSA key format: Unable to determine if key is public or private.")

      der_data = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
      )
    elif "public" in key_plain.lower():
      public_key = serialization.load_pem_public_key(rsa_key_pem, backend=default_backend())
      if not isinstance(public_key, rsa.RSAPublicKey):
        raise ValueError("Invalid RSA key format: Unable to determine if key is public or private.")

      der_data = public_key.public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.PKCS1)
    else:
      raise ValueError("Unknown key format: Unable to determine if key is public or private.")

    sha256_hash = hashlib.sha256(der_data).digest()
    aes_key = sha256_hash[:32] if use_aes_256 else sha256_hash[:16]
    aes_iv = sha256_hash[16:32]

    return aes_key, aes_iv


def qUncompress(data):
  """
  Decompress data using zlib.

  Args:
      data (bytes): Compressed data

  Returns:
      bytes: Decompressed data
  """
  data_stripped_4 = data[4:]
  return zlib.decompress(data_stripped_4)


def qCompress(data):
  """
  Compress data using zlib.

  Args:
      data (bytes): Data to compress

  Returns:
      bytes: Compressed data
  """
  compressed_data = zlib.compress(data, level=9)
  return b"ZLIB" + compressed_data


def decrypt_compressed_data(encrypted_base64, use_aes_256=False):
  """
  Decrypt and decompress data from base64 string.

  Args:
      encrypted_base64 (str): Base64 encoded encrypted data
      key_path (str, optional): Path to RSA public key

  Returns:
      str: Decrypted and decompressed string
  """
  key_path = Path(f"{Paths.persist_root()}/comma/id_rsa") if use_aes_256 else Path(f"{Paths.persist_root()}/comma/id_rsa.pub")
  try:
    # Decode base64
    encrypted_data = base64.b64decode(encrypted_base64)

    # Decrypt
    key, iv = KeyDerivation.derive_aes_key_iv_from_rsa(str(key_path), use_aes_256)
    cipher = AESCipher(key, iv)
    decrypted_data = cipher.decrypt(encrypted_data)

    # Decompress
    decompressed_data = qUncompress(decrypted_data)

    # Decode UTF-8
    result = decompressed_data.decode('utf-8')
    return result
  except Exception as e:
    print(f"Decryption and decompression failed: {e}")
    return ""


def encrypt_compress_data(text, use_aes_256=True):
  """
  Compress and encrypt string data to base64.

  Args:
      text (str): Text to compress and encrypt
      key_path (str, optional): Path to RSA public key

  Returns:
      str: Base64 encoded encrypted data
  """
  key_path = Path(f"{Paths.persist_root()}/comma/id_rsa") if use_aes_256 else Path(f"{Paths.persist_root()}/comma/id_rsa.pub")
  try:
    # Encode to UTF-8
    text_bytes = text.encode('utf-8')

    # Compress
    compressed_data = qCompress(text_bytes)

    # Encrypt
    key, iv = KeyDerivation.derive_aes_key_iv_from_rsa(str(key_path), use_aes_256)
    cipher = AESCipher(key, iv)
    encrypted_data = cipher.encrypt(compressed_data)

    # Encode to base64
    result = base64.b64encode(encrypted_data).decode('utf-8')
    return result
  except Exception as e:
    print(f"Compression and encryption failed: {e}")
    return ""


def camel_to_snake(name):
  """Convert camelCase to snake_case."""
  name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def transform_dict(obj):
  """Recursively transform dictionary keys from camelCase to snake_case."""
  if isinstance(obj, dict):
    return {camel_to_snake(k): transform_dict(v) for k, v in obj.items()}
  elif isinstance(obj, list):
    return [transform_dict(item) for item in obj]
  return obj


class SnakeCaseEncoder(json.JSONEncoder):
  def encode(self, obj):
    transformed_obj = transform_dict(obj)
    return super().encode(transformed_obj)
