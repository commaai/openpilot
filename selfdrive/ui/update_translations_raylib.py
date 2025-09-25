#!/usr/bin/env python3
import argparse
import json
import os

from openpilot.common.basedir import BASEDIR

UI_DIR = os.path.join(BASEDIR, "selfdrive", "ui")
TRANSLATIONS_DIR = os.path.join(UI_DIR, "translations")
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, "languages.json")


def update_translations():
  files = []
  for root, _, filenames in os.walk(os.path.join(UI_DIR, "widgets")):
    for filename in filenames:
      if filename.endswith(".py"):
        files.append(os.path.join(root, filename))

  # Create main translation file
  print(files)
  cmd = ("xgettext -L Python --keyword=tr --keyword=trn:1,2 --keyword=pgettext:1c,2 --from-code=UTF-8 " +
         "--flag=tr:1:python-brace-format --flag=trn:1:python-brace-format --flag=trn:2:python-brace-format " +
         "-o translations/app.pot {}").format(" ".join(files))
  print(cmd)

  ret = os.system(cmd)
  assert ret == 0

  # Generate/update translation files for each language
  with open(LANGUAGES_FILE) as f:
    translation_files = json.load(f).values()

  for file in translation_files:
    name = file.replace("main_", "")
    if os.path.exists(os.path.join(TRANSLATIONS_DIR, f"app_{name}.po")):
      cmd = "msgmerge --update --backup=none --sort-output translations/app.pot translations/app_{}.po".format(name)
      ret = os.system(cmd)
      assert ret == 0
    else:
      cmd = "msginit -l es --no-translator --input translations/app.pot --output-file translations/app_{}.po".format(name)
      ret = os.system(cmd)
      assert ret == 0


if __name__ == "__main__":
  update_translations()
