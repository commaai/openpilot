#!/usr/bin/env python3
import json
import os
import unittest
import requests

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(TEST_DIR, "../agnos.json")


class TestAgnosUpdater(unittest.TestCase):

  def test_manifest(self):
    with open(MANIFEST) as f:
      m = json.load(f)

    for img in m:
      r = requests.head(img['url'], timeout=10)
      r.raise_for_status()
      self.assertEqual(r.headers['Content-Type'], "application/x-xz")
      if not img['sparse']:
        assert img['hash'] == img['hash_raw']


if __name__ == "__main__":
  unittest.main()
