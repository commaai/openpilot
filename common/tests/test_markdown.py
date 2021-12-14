#!/usr/bin/env python3
from markdown_it import MarkdownIt
import os
import unittest

from common.basedir import BASEDIR
from common.markdown import parse_markdown


class TestMarkdown(unittest.TestCase):
  # validate that our simple markdown parser produces the same output as `markdown_it` from pip
  def test_current_release_notes(self):
    self.maxDiff = None

    with open(os.path.join(BASEDIR, "RELEASES.md")) as f:
      for r in f.read().split("\n\n"):

        # No hyperlink support is ok
        if '[' in r:
          continue

        self.assertEqual(MarkdownIt().render(r), parse_markdown(r))


if __name__ == "__main__":
  unittest.main()
