"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from Crypto.Cipher import AES


class AESCipher:
  def __init__(self, key: bytes, iv: bytes):
    if len(key) not in (16, 32):
      raise ValueError("Key must be 16 bytes (AES-128) or 32 bytes (AES-256).")
    if len(iv) != 16:
      raise ValueError("IV must be 16 bytes.")

    self.key = key
    self.iv = iv

  def encrypt(self, data: bytes) -> bytes:
    block_size = 16
    padding_length = block_size - (len(data) % block_size)
    padding = bytes([padding_length]) * padding_length
    padded_data = data + padding

    cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
    return cipher.encrypt(padded_data)

  def decrypt(self, encrypted_data: bytes) -> bytes:
    cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
    decrypted_data = cipher.decrypt(encrypted_data)
    padding_length = decrypted_data[-1]
    return decrypted_data[:-padding_length]
