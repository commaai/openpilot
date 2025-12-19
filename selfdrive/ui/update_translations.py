#!/usr/bin/env python3
from itertools import chain
import os
from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.multilang import SYSTEM_UI_DIR, UI_DIR, TRANSLATIONS_DIR, multilang

LANGUAGES_FILE = os.path.join(str(TRANSLATIONS_DIR), "languages.json")
POT_FILE = os.path.join(str(TRANSLATIONS_DIR), "app.pot")


def update_translations():
  files = []
  for root, _, filenames in chain(os.walk(SYSTEM_UI_DIR),
                                  os.walk(os.path.join(UI_DIR, "widgets")),
                                  os.walk(os.path.join(UI_DIR, "layouts")),
                                  os.walk(os.path.join(UI_DIR, "onroad"))):
    for filename in filenames:
      if filename.endswith(".py"):
        files.append(os.path.relpath(os.path.join(root, filename), BASEDIR))

  # Create main translation file
  cmd = ("xgettext -L Python --keyword=tr --keyword=trn:1,2 --keyword=tr_noop --from-code=UTF-8 " +
         "--flag=tr:1:python-brace-format --flag=trn:1:python-brace-format --flag=trn:2:python-brace-format " +
         f"-D {BASEDIR} -o {POT_FILE} {' '.join(files)}")

  ret = os.system(cmd)
  assert ret == 0

  # Generate/update translation files for each language
  for name in multilang.languages.values():
    if os.path.exists(os.path.join(TRANSLATIONS_DIR, f"app_{name}.po")):
      cmd = f"msgmerge --update --no-fuzzy-matching --backup=none --sort-output {TRANSLATIONS_DIR}/app_{name}.po {POT_FILE}"
      ret = os.system(cmd)
      assert ret == 0
    else:
      cmd = f"msginit -l {name} --no-translator --input {POT_FILE} --output-file {TRANSLATIONS_DIR}/app_{name}.po"
      ret = os.system(cmd)
      assert ret == 0


if __name__ == "__main__":
  update_translations()
