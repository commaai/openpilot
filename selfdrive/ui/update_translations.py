#!/usr/bin/env python3
import argparse
import os
import json

from common.basedir import BASEDIR

UI_DIR = os.path.join(BASEDIR, 'selfdrive', 'ui')
TRANSLATIONS_DIR = os.path.join(UI_DIR, 'translations')
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, 'languages.json')


def update_translations(no_update=False, release=False, suffix=''):
  with open(LANGUAGES_FILE, 'r') as f:
    translation_files = json.load(f)

  for name, file in translation_files.items():
    if not len(file):
      print(f'{name} translation has no file, skipping...')
      continue
    tr_file = os.path.join(TRANSLATIONS_DIR, f'{file}{suffix}.ts')
    if not no_update:
      # if os.path.exists(tr_file):
      #   print(f'Updating {name} translation: {tr_file}')
      # else:
      #   print(f'Creating {name} translation: {tr_file}')
      ret = os.system(f'lupdate -recursive {UI_DIR} -ts {tr_file}')
      assert ret == 0
    if release:
      ret = os.system(f'lrelease {tr_file}')
      assert ret == 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Update translation files for UI',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--no-update', action='store_true', help='Skip updating ts translation files')
  parser.add_argument('--release', action='store_true', help='Create final qm translation files used by UI')
  args = parser.parse_args()

  update_translations(args.no_update, args.release)
