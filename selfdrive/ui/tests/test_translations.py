#!/usr/bin/env python3
from collections import defaultdict
import json
import os
import unittest

from selfdrive.ui.update_translations import TRANSLATIONS_DIR, LANGUAGES_FILE, update_translations


class TestTranslations(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    with open(LANGUAGES_FILE, "r") as f:
      cls.translation_files = json.load(f)
    cls.prev_translations = cls._get_previous_translations(cls)

  def _get_previous_translations(self):
    prev_translations = defaultdict(lambda: defaultdict(str))

    for file in self.translation_files.values():
      if len(file):
        for file_ext in ["ts", "qm"]:
          tr_file = os.path.join(TRANSLATIONS_DIR, f"{file}.{file_ext}")

          if os.path.exists(tr_file):
            with open(tr_file, "rb") as f:
              prev_translations[file][file_ext] = f.read()

    return prev_translations

  def test_missing_translation_files(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        if not len(file):
          self.skipTest(f"{name} translation has no file")

        self.assertTrue(len(self.prev_translations[file]["ts"]),
                        f"{name} has no XML translation file, run selfdrive/ui/update_translations.py")
        self.assertTrue(len(self.prev_translations[file]["qm"]),
                        f"{name} has no compiled QM translation file, run selfdrive/ui/update_translations.py --release")

  def test_translations_updated(self):
    update_translations(release=True)

    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        for file_ext in ["ts", "qm"]:
          with self.subTest(file_ext=file_ext):
            new_file = os.path.join(TRANSLATIONS_DIR, f"{file}.{file_ext}")

            # caught by test_missing_translation_files
            if not len(file):
              self.skipTest(f"{name} translation has no file")
            elif not len(self.prev_translations[file][file_ext]):
              self.skipTest(f"{name} missing translation file")

            with open(new_file, "rb") as f:
              new_translations = f.read()

            self.assertEqual(self.prev_translations[file][file_ext], new_translations,
                             f"{file} ({name}) {file_ext.upper()} translation file out of date. Run selfdrive/ui/update_translations.py --release to update the translation files")


if __name__ == "__main__":
  unittest.main()
