#!/usr/bin/env python3
from itertools import chain
import os
from openpilot.system.ui.lib.multilang import SYSTEM_UI_DIR, UI_DIR, TRANSLATIONS_DIR, multilang


def update_translations():
  files = []
  for root, _, filenames in chain(os.walk(SYSTEM_UI_DIR),
                                  os.walk(os.path.join(UI_DIR, "widgets")),
                                  os.walk(os.path.join(UI_DIR, "layouts")),
                                  os.walk(os.path.join(UI_DIR, "onroad"))):
    for filename in filenames:
      if filename.endswith(".py"):
        files.append(os.path.join(root, filename))

  # Create main translation file
  cmd = ("xgettext -L Python --keyword=tr --keyword=trn:1,2 --keyword=tr_noop --from-code=UTF-8 " +
         "--flag=tr:1:python-brace-format --flag=trn:1:python-brace-format --flag=trn:2:python-brace-format " +
         "-o translations/app.pot {}").format(" ".join(files))

  ret = os.system(cmd)
  assert ret == 0

  # Generate/update translation files for each language
  for name in multilang.languages.values():
    if os.path.exists(os.path.join(TRANSLATIONS_DIR, f"app_{name}.po")):
      cmd = f"msgmerge --update --no-fuzzy-matching --backup=none --sort-output translations/app_{name}.po translations/app.pot"
      ret = os.system(cmd)
      assert ret == 0

      # cmd = f"msgattrib --no-fuzzy --no-obsolete -o translations/app_{name}.po translations/app_{name}.po"
      # ret = os.system(cmd)
      # assert ret == 0
    else:
      cmd = f"msginit -l {name} --no-translator --input translations/app.pot --output-file translations/app_{name}.po"
      ret = os.system(cmd)
      assert ret == 0


if __name__ == "__main__":
  update_translations()
