#!/usr/bin/env python3
import pytest
import json
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import string
import requests
from parameterized import parameterized_class

from openpilot.selfdrive.ui.update_translations import TRANSLATIONS_DIR, LANGUAGES_FILE, update_translations

with open(LANGUAGES_FILE) as f:
  translation_files = json.load(f)

UNFINISHED_TRANSLATION_TAG = "<translation type=\"unfinished\""  # non-empty translations can be marked unfinished
LOCATION_TAG = "<location "
FORMAT_ARG = re.compile("%[0-9]+")


@parameterized_class(("name", "file"), translation_files.items())
class TestTranslations:
  name: str
  file: str

  @staticmethod
  def _read_translation_file(path, file):
    tr_file = os.path.join(path, f"{file}.ts")
    with open(tr_file) as f:
      return f.read()

  def test_missing_translation_files(self):
    assert os.path.exists(os.path.join(TRANSLATIONS_DIR, f"{self.file}.ts")), \
                    f"{self.name} has no XML translation file, run selfdrive/ui/update_translations.py"

  def test_translations_updated(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      shutil.copytree(TRANSLATIONS_DIR, tmpdir, dirs_exist_ok=True)
      update_translations(translation_files=[self.file], translations_dir=tmpdir)

      cur_translations = self._read_translation_file(TRANSLATIONS_DIR, self.file)
      new_translations = self._read_translation_file(tmpdir, self.file)
      assert cur_translations == new_translations, \
                       f"{self.file} ({self.name}) XML translation file out of date. Run selfdrive/ui/update_translations.py to update the translation files"

  @pytest.mark.skip("Only test unfinished translations before going to release")
  def test_unfinished_translations(self):
    cur_translations = self._read_translation_file(TRANSLATIONS_DIR, self.file)
    assert UNFINISHED_TRANSLATION_TAG not in cur_translations, \
                    f"{self.file} ({self.name}) translation file has unfinished translations. Finish translations or mark them as completed in Qt Linguist"

  def test_vanished_translations(self):
    cur_translations = self._read_translation_file(TRANSLATIONS_DIR, self.file)
    assert "<translation type=\"vanished\">" not in cur_translations, \
                    f"{self.file} ({self.name}) translation file has obsolete translations. Run selfdrive/ui/update_translations.py --vanish to remove them"

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
    tr_xml = ET.parse(os.path.join(TRANSLATIONS_DIR, f"{self.file}.ts"))

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
            assert nf is not None, f"Ensure all plural translation forms are completed: {source_text}"
            assert "%n" in nf, "Ensure numerus argument (%n) exists in translation."
            assert FORMAT_ARG.search(nf) is None, f"Plural translations must use %n, not %1, %2, etc.: {numerusform}"

        else:
          assert translation.text is not None, f"Ensure translation is completed: {source_text}"

          source_args = FORMAT_ARG.findall(source_text)
          translation_args = FORMAT_ARG.findall(translation.text)
          assert sorted(source_args) == sorted(translation_args), \
                           f"Ensure format arguments are consistent: `{source_text}` vs. `{translation.text}`"

  def test_no_locations(self):
    for line in self._read_translation_file(TRANSLATIONS_DIR, self.file).splitlines():
      assert not line.strip().startswith(LOCATION_TAG), \
                       f"Line contains location tag: {line.strip()}, remove all line numbers."

  def test_entities_error(self):
    cur_translations = self._read_translation_file(TRANSLATIONS_DIR, self.file)
    matches = re.findall(r'@(\w+);', cur_translations)
    assert len(matches) == 0, f"The string(s) {matches} were found with '@' instead of '&'"

  def test_bad_language(self):
    IGNORED_WORDS = {'pédale'}

    match = re.search(r'_([a-zA-Z]{2,3})', self.file)
    assert match, f"{self.name} - could not parse language"

    response = requests.get(f"https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/{match.group(1)}")
    response.raise_for_status()

    banned_words = {line.strip() for line in response.text.splitlines()}

    for context in ET.parse(os.path.join(TRANSLATIONS_DIR, f"{self.file}.ts")).getroot():
      for message in context.iterfind("message"):
        translation = message.find("translation")
        if translation.get("type") == "unfinished":
          continue

        translation_text = " ".join([t.text for t in translation.findall("numerusform")]) if message.get("numerus") == "yes" else translation.text

        if not translation_text:
          continue

        words = set(translation_text.translate(str.maketrans('', '', string.punctuation + '%n')).lower().split())
        bad_words_found = words & (banned_words - IGNORED_WORDS)
        assert not bad_words_found, f"Bad language found in {self.name}: '{translation_text}'. Bad word(s): {', '.join(bad_words_found)}"
