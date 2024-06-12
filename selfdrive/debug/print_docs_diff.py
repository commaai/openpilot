#!/usr/bin/env python3
import argparse
import os
import subprocess

from collections import defaultdict

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car.docs_definitions import Column

STAR_ICON = '<a href="##"><img valign="top" ' + \
            'src="https://media.githubusercontent.com/media/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
COLUMN_HEADER = "|---|---|---|{}|".format("|".join([":---:"] * (len(Column) - 3)))
ARROW_SYMBOL = "➡️"

def column_change_format(line1, line2):
  info1, info2 = line1.split('|'), line2.split('|')
  return "|".join([f"{i1} {ARROW_SYMBOL} {i2}|" if i1 != i2 else f"{i1}|" for i1, i2 in zip(info1, info2, strict=True)])

def get_detail_sentence(data):
  if len(data.split("|")) == 11:
    return data.split("|")[8]

def process_detail_sentences(info):
  detail_sentences = []
  ind = info.index("---")
  for line1, line2 in zip(info[1:ind], info[ind+1:], strict=True):
    name = ' '.join(line1[:2].split("|")[:2])
    detail1 = get_detail_sentence(line1[2:])
    detail2 = get_detail_sentence(line2[2:])
    if detail1 != detail2:
      detail_sentences.append(f"- Sentence for {name} changed!\n" +
                                 "  ```diff\n" +
                                 f"  - {detail1}\n" +
                                 f"  + {detail2}\n" +
                                 "  ```")
  return detail_sentences

def process_diff_information(info):
  header = info[0]
  category = None
  final_strings = []
  if "c" in header:
    category = "column"
    ind = info.index("---")
    for line1, line2 in zip(info[1:ind], info[ind+1:], strict=True):
      final_strings.append(column_change_format(line1[2:], line2[2:]))
  else:
    category = "additions" if "a" in header else "removals"
    final_strings = [x[2:] for x in info[1:]]
  return category, final_strings

def print_markdown(changes):
  markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]
  for title, category in (("## 🔀 Column Changes", "column"), ("## ❌ Removed", "removals"),
                          ("## ➕ Added", "additions"), ("## 📖 Detail Sentence Changes", "detail")):
    # TODO: Add details for detail changes
    if len(changes[category]):
      markdown_builder.append(title)
      if category not in ("detail",):
        markdown_builder.append(COLUMNS)
        markdown_builder.append(COLUMN_HEADER)
      markdown_builder.extend(changes[category])
  print("\n".join(markdown_builder))

def print_car_docs_diff(path):
  CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
  MASTER_CARS_MD = os.path.join(path, "CARS.md")
  changes = subprocess.run(['diff', MASTER_CARS_MD, CARS_MD_OUT], capture_output=True, text=True).stdout.split('\n')

  changes_markdown = defaultdict(list)
  ind = 0
  while ind < len(changes):
    if changes[ind] and changes[ind][0].isdigit():
      start = ind
      ind += 1
      while ind < len(changes) and changes[ind] and not changes[ind][0].isdigit():
        ind += 1
      category, strings = process_diff_information(changes[start:ind])
      changes_markdown[category] += strings
      if category == "column":
        changes_markdown["detail"] += process_detail_sentences(changes[start:ind])
    else:
      ind += 1

  if any(len(c) for c in changes_markdown.values()):
    print_markdown(changes_markdown)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  args = parser.parse_args()
  print_car_docs_diff(args.path)
