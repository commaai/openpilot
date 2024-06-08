import os

from openpilot.common.basedir import BASEDIR
from openpilot.common.markdown import parse_markdown


class TestMarkdown:
  def test_all_release_notes(self):
    with open(os.path.join(BASEDIR, "RELEASES.md")) as f:
      release_notes = f.read().split("\n\n")
      assert len(release_notes) > 10

      for rn in release_notes:
        md = parse_markdown(rn)
        assert len(md) > 0
