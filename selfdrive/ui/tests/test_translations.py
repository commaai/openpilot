#!/usr/bin/env python3
import json
import os
import re
import shutil
import unittest
import xml.etree.ElementTree as ET

from openpilot.selfdrive.ui.update_translations import TRANSLATIONS_DIR, LANGUAGES_FILE, update_translations

TMP_TRANSLATIONS_DIR = os.path.join(TRANSLATIONS_DIR, "tmp")
UNFINISHED_TRANSLATION_TAG = "<translation type=\"unfinished\""  # non-empty translations can be marked unfinished
LOCATION_TAG = "<location "
FORMAT_ARG = re.compile("%[0-9]+")


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
  def _read_translation_file(path, file):
    tr_file = os.path.join(path, f"{file}.ts")
    with open(tr_file, "r") as f:
      return f.read()

  def test_missing_translation_files(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        self.assertTrue(os.path.exists(os.path.join(TRANSLATIONS_DIR, f"{file}.ts")),
                        f"{name} has no XML translation file, run selfdrive/ui/update_translations.py")

  def test_translations_updated(self):
    update_translations(plural_only=["main_en"], translations_dir=TMP_TRANSLATIONS_DIR)

    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        # caught by test_missing_translation_files
        if not os.path.exists(os.path.join(TRANSLATIONS_DIR, f"{file}.ts")):
          self.skipTest(f"{name} missing translation file")

        cur_translations = self._read_translation_file(TRANSLATIONS_DIR, file)
        new_translations = self._read_translation_file(TMP_TRANSLATIONS_DIR, file)
        self.assertEqual(cur_translations, new_translations,
                         f"{file} ({name}) XML translation file out of date. Run selfdrive/ui/update_translations.py to update the translation files")

  @unittest.skip("Only test unfinished translations before going to release")
  def test_unfinished_translations(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        cur_translations = self._read_translation_file(TRANSLATIONS_DIR, file)
        self.assertTrue(UNFINISHED_TRANSLATION_TAG not in cur_translations,
                        f"{file} ({name}) translation file has unfinished translations. Finish translations or mark them as completed in Qt Linguist")

  def test_vanished_translations(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        cur_translations = self._read_translation_file(TRANSLATIONS_DIR, file)
        self.assertTrue("<translation type=\"vanished\">" not in cur_translations,
                        f"{file} ({name}) translation file has obsolete translations. Run selfdrive/ui/update_translations.py --vanish to remove them")

  def test_finished_translations(self):
    """
      Tests ran on each translation marked "finished"
      Plural:
      - that any numerus (plural) translations have all plural forms non-empty
      - that the correct format specifier is used (%n)
      Non-plural:
      - that translation is not empty
      - that translation format arguments are consistent
    """
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        tr_xml = ET.parse(os.path.join(TRANSLATIONS_DIR, f"{file}.ts"))

        for context in tr_xml.getroot():
          for message in context.iterfind("message"):
            translation = message.find("translation")
            source_text = message.find("source").text

            # Do not test unfinished translations
            if translation.get("type") == "unfinished":
              continue

            if message.get("numerus") == "yes":
              numerusform = [t.text for t in translation.findall("numerusform")]

              for nf in numerusform:
                self.assertIsNotNone(nf, f"Ensure all plural translation forms are completed: {source_text}")
                self.assertIn("%n", nf, "Ensure numerus argument (%n) exists in translation.")
                self.assertIsNone(FORMAT_ARG.search(nf), "Plural translations must use %n, not %1, %2, etc.: {}".format(numerusform))

            else:
              self.assertIsNotNone(translation.text, f"Ensure translation is completed: {source_text}")

              source_args = FORMAT_ARG.findall(source_text)
              translation_args = FORMAT_ARG.findall(translation.text)
              self.assertEqual(sorted(source_args), sorted(translation_args),
                               f"Ensure format arguments are consistent: `{source_text}` vs. `{translation.text}`")

  def test_no_locations(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        for line in self._read_translation_file(TRANSLATIONS_DIR, file).splitlines():
          self.assertFalse(line.strip().startswith(LOCATION_TAG),
                           f"Line contains location tag: {line.strip()}, remove all line numbers.")

  def test_entities_error(self):
    for name, file in self.translation_files.items():
      with self.subTest(name=name, file=file):
        cur_translations = self._read_translation_file(TRANSLATIONS_DIR, file)
        matches = re.findall(r'@(\w+);', cur_translations)
        self.assertEqual(len(matches), 0, f"The string(s) {matches} were found with '@' instead of '&'")


if __name__ == "__main__":
  unittest.main()
