#!/usr/bin/env python3
import unittest

from opendbc.safety.tests.libsafety.libsafety_py import _build_libsafety


class TestBuild(unittest.TestCase):
  def test_development_build(self):
    _build_libsafety(release=False)

  def test_release_build(self):
    _build_libsafety(release=True)


if __name__ == "__main__":
  unittest.main()
