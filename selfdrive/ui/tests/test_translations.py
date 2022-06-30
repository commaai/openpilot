#!/usr/bin/env python3
import os
import json
import unittest
from selfdrive.ui.update_translations import TRANSLATIONS_DIR, LANGUAGES_FILE, update_translations


class TestTranslations(unittest.TestCase):

  def test_missing_language_files(self):
    with open(LANGUAGES_FILE, 'r') as f:
      translation_files = json.load(f)

    for name, file in translation_files.items():
      with self.subTest(name=name, file=file):
        if not len(file):
          self.skipTest(f'{name} translation has no file, skipping...')

        file_path = os.path.join(TRANSLATIONS_DIR, f'{file}.ts')
        self.assertTrue(os.path.exists(file_path), f'{name} has no language file, run selfdrive/ui/update_translations.py')


if __name__ == '__main__':
  unittest.main()
