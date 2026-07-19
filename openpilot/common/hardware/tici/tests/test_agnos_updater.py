import json
import os
import requests

from openpilot.selfdrive.test.helpers import OpenpilotTestCase

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(TEST_DIR, "../agnos.json")


class TestAgnosUpdater(OpenpilotTestCase):

  def test_manifest(self):
    with open(MANIFEST) as f:
      m = json.load(f)

    for img in m:
      r = requests.head(img['url'], timeout=10)
      r.raise_for_status()
      assert r.headers['Content-Type'] == "application/x-xz"
      if not img['sparse']:
        assert img['hash'] == img['hash_raw']
