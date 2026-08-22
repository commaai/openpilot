import unittest

from openpilot.common.nm_keyfile import decode_nm_keyfile_ssid


class TestNmKeyfile(unittest.TestCase):
  def test_decodes_escaped_semicolon(self):
    self.assertEqual(decode_nm_keyfile_ssid(r"Cafe\;Guest"), "Cafe;Guest")

  def test_decodes_escaped_backslash(self):
    self.assertEqual(decode_nm_keyfile_ssid(r"Cafe\\Guest"), r"Cafe\Guest")

  def test_preserves_literal_backslash_before_semicolon(self):
    self.assertEqual(decode_nm_keyfile_ssid(r"Cafe\\;Guest"), r"Cafe\;Guest")

  def test_decodes_boundary_spaces(self):
    self.assertEqual(decode_nm_keyfile_ssid(r"\sCafe\s"), " Cafe ")

  def test_decodes_decimal_byte_list(self):
    self.assertEqual(decode_nm_keyfile_ssid("65;66;67;"), "ABC")

  def test_preserves_non_utf8_decimal_byte_list(self):
    self.assertEqual(
      decode_nm_keyfile_ssid("255;65;"),
      b"\xffA".decode("utf-8", errors="surrogateescape"),
    )

  def test_preserves_unknown_escape(self):
    self.assertEqual(decode_nm_keyfile_ssid(r"Cafe\q"), r"Cafe\q")


if __name__ == "__main__":
  unittest.main()
