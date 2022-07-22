#!/usr/bin/env python3
import os
import json

from selfdrive.ui.update_translations import TRANSLATIONS_DIR, LANGUAGES_FILE

TRANSLATION_TAG = "<translation"
UNFINISHED_TRANSLATION_TAG = '<translation type="unfinished"'

if __name__ == "__main__":
  with open(LANGUAGES_FILE, "r") as f:
    translation_files = json.load(f)

  unfinished_translations = 0
  for name, file in translation_files.items():
    if not len(file):
      continue

    tr_file = os.path.join(TRANSLATIONS_DIR, f"{file}.ts")
    with open(tr_file, "r") as f:
      tr_file = f.read()

    total_translations = 0  # TODO: this assumes equal translations for all files, not super explicit but works
    for line in tr_file.splitlines():
      if TRANSLATION_TAG in line:
        total_translations += 1
      if UNFINISHED_TRANSLATION_TAG in line:
        unfinished_translations += 1

  percent_finished = (1 - unfinished_translations / total_translations) * 100.
  percent_finished = int(percent_finished * 10) / 10  # round down (99.99% shouldn't be 100%)
  print(percent_finished)
