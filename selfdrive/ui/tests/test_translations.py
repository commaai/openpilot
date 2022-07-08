#!/usr/bin/env python3
import json
import os
import shutil
import unittest

from selfdrive.ui.update_translations import TRANSLATIONS_DIR, LANGUAGES_FILE, update_translations

TMP_TRANSLATIONS_DIR = os.path.join(TRANSLATIONS_DIR, "tmp")


class TestTranslations(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    with open(LANGUAGES_FILE, "r") as f:
      cls.translation_files = json.load(f)

    # Set up temp directory
    shutil.copytree(TRANSLATIONS_DIR, TMP_TRANSLATIONS_DIR, dirs_exist_ok=True)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(TMP_TRANSLATIONS_DIR, ignore_errors=True)

  @staticmethod
  def _read_translation_file(path, file, file_ext):
    tr_file = os.path.join(path, f"{file}.{file_ext}")
    with open(tr_file, "rb") as f:
      # fix relative path depth
      return f.read().replace(b"filename=\"../../", b"filename=\"../")

  def test_missing_translation_files(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        if not len(file):
          self.skipTest(f"{name} translation has no defined file")

        self.assertTrue(os.path.exists(os.path.join(TRANSLATIONS_DIR, f"{file}.ts")),
                        f"{name} has no XML translation file, run selfdrive/ui/update_translations.py")
        self.assertTrue(os.path.exists(os.path.join(TRANSLATIONS_DIR, f"{file}.qm")),
                        f"{name} has no compiled QM translation file, run selfdrive/ui/update_translations.py --release")

  def test_translations_updated(self):
    update_translations(release=True, translations_dir=TMP_TRANSLATIONS_DIR)

    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        if not len(file):
          self.skipTest(f"{name} translation has no defined file")

        for file_ext in ["ts", "qm"]:
          with self.subTest(file_ext=file_ext):

            # caught by test_missing_translation_files
            if not os.path.exists(os.path.join(TRANSLATIONS_DIR, f"{file}.{file_ext}")):
              self.skipTest(f"{name} missing translation file")

            cur_translations = self._read_translation_file(TRANSLATIONS_DIR, file, file_ext)
            new_translations = self._read_translation_file(TMP_TRANSLATIONS_DIR, file, file_ext)

            self.assertEqual(cur_translations, new_translations,
                             f"{file} ({name}) {file_ext.upper()} translation file out of date. Run selfdrive/ui/update_translations.py --release to update the translation files")

  def test_unfinished_translations(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        if not len(file):
          raise self.skipTest(f"{name} translation has no defined file")

        cur_translations = self._read_translation_file(TRANSLATIONS_DIR, file, "ts")
        self.assertTrue(b"<translation type=\"unfinished\">" not in cur_translations,
                        f"{file} ({name}) translation file has unfinished translations. Finish translations or mark them as completed in Qt Linguist")


if __name__ == "__main__":
  unittest.main()
