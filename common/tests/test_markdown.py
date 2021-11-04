from markdown_it import MarkdownIt
import os
import unittest

from common.basedir import BASEDIR
from common.markdown import parse_markdown

class TestMarkdown(unittest.TestCase):
  # validate that our simple markdown parser produces the same output as `markdown_it` from pip
  def test_current_release_notes(self):
    with open(os.path.join(BASEDIR, "RELEASES.md")) as f:
      r = f.read().split("\n\n", 1)[0]
    self.maxDiff = None
    self.assertEqual(MarkdownIt().render(r), parse_markdown(r))


if __name__ == "__main__":
  unittest.main()
