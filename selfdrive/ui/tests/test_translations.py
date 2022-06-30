#!/usr/bin/env python3
import json
import os
import unittest

from selfdrive.ui.update_translations import TRANSLATIONS_DIR, LANGUAGES_FILE, update_translations


class TestTranslations(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    with open(LANGUAGES_FILE, "r") as f:
      cls.translation_files = json.load(f)

  def test_missing_translation_files(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        if not len(file):
          self.skipTest(f"{name} translation has no file")

        self.assertTrue(os.path.exists(os.path.join(TRANSLATIONS_DIR, f"{file}.ts")),
                        f"{name} has no XML translation file, run selfdrive/ui/update_translations.py")
        self.assertTrue(os.path.exists(os.path.join(TRANSLATIONS_DIR, f"{file}.qm")),
                        f"{name} has no compiled QM translation file, run selfdrive/ui/update_translations.py --release")

  def test_translations_updated(self):
    suffix = "_test"
    update_translations(suffix=suffix)

    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        cur_tr_file = os.path.join(TRANSLATIONS_DIR, f"{file}.ts")
        new_tr_file = os.path.join(TRANSLATIONS_DIR, f"{file}{suffix}.ts")

        if not len(file):
          self.skipTest(f"{name} translation has no file")
        elif not os.path.exists(cur_tr_file):
          self.skipTest(f"{name} missing translation file")  # caught by test_missing_translation_files

        with open(cur_tr_file, "r") as f:
          cur_translations = f.read()
        with open(new_tr_file, "r") as f:
          new_translations = f.read()

        self.assertEqual(cur_translations, new_translations,
                         f"{name} translation file out of date. Run selfdrive/ui/update_translations.py to update the translation files")


if __name__ == "__main__":
  unittest.main()
