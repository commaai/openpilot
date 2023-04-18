#!/usr/bin/env python3
import argparse
import os
import time
from tqdm import tqdm
import xml.etree.ElementTree as ET

import translators as ts

from selfdrive.ui.update_translations import TRANSLATIONS_DIR


def chunks(l, n=128):
  for i in range(0, len(l), n):
    yield l[i:i + n]


def view_translations(file):
  sources = []
  translations = []
  tr_xml = ET.parse(os.path.join(TRANSLATIONS_DIR, f"{file}.ts"))
  for context in tqdm(tr_xml.getroot()):
    for message in context.iterfind("message"):
      sources.append(message.find('source').text)
      if message.get("numerus") == "yes":
        translations.append(', '.join([t.text for t in message.find('translation').findall("numerusform")]))
      else:
        translations.append(message.find('translation').text)

  for chunk in chunks(list(zip(sources, translations)), 10):
    time.sleep(0.1)
    src, trans = zip(*chunk)
    translated = ts.translate_text(' || '.join(trans), translator='google')
    translated = [i.strip() for i in translated.split('||')]
    for s, t, t_en in zip(src, trans, translated):
      print(f'English         : {s}')
      print(f'Translation (en): {t_en}')
      print(f'Translation ({file.replace("main_", "")}): {t}')
      print()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("fn", type=str, help="Name of translations file to translate/view (eg. main_es)")
  args = parser.parse_args()

  view_translations(args.fn)
