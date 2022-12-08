#!/usr/bin/env python3
import argparse
import json
import os

from common.basedir import BASEDIR

UI_DIR = os.path.join(BASEDIR, "selfdrive", "ui")
TRANSLATIONS_DIR = os.path.join(UI_DIR, "translations")
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, "languages.json")


def update_translations(vanish=False, plural_only=None, translations_dir=TRANSLATIONS_DIR):
  if plural_only is None:
    plural_only = []

  with open(LANGUAGES_FILE, "r") as f:
    translation_files = json.load(f)

  for file in translation_files.values():
    tr_file = os.path.join(translations_dir, f"{file}.ts")
    args = f"lupdate -locations none -recursive {UI_DIR} -ts {tr_file}"
    if vanish:
      args += " -no-obsolete"
    if file in plural_only:
      args += " -pluralonly"
    ret = os.system(args)
    assert ret == 0


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Update translation files for UI",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--vanish", action="store_true", help="Remove translations with source text no longer found")
  parser.add_argument("--plural-only", type=str, nargs="*", default=["main_en"], help="Translation codes to only create plural translations for (ie. the base language)")
  args = parser.parse_args()

  update_translations(args.vanish, args.plural_only)
