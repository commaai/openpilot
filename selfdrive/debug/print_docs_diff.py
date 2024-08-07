#!/usr/bin/env python3
import argparse
from collections import defaultdict
import os
import subprocess

from openpilot.common.basedir import BASEDIR
from openpilot.common.conversions import Conversions as CV
from openpilot.common.enums import Column
from openpilot.common.detail_sentence import get_detail_sentence

COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
COLUMN_HEADER = "|---|---|---|{}|".format("|".join([":---:"] * (len(Column) - 3)))
ARROW_SYMBOL = "➡️"

def process_detail_sentence(old, new):
  name = ' '.join(old[3:].split("|")[:2]) # Make Model Year
  cur_sentence = []
  for line in [old, new]:
    make, model, _, longitudinal, fsr_longitudinal, fsr_steering, steering_torque, auto_resume, _, _ = line.split("|")[1:-1]
    min_steer_speed = float(fsr_steering[:fsr_steering.index('mph')]) / CV.MS_TO_MPH
    min_enable_speed = float(fsr_longitudinal[:fsr_longitudinal.index('mph')]) / CV.MS_TO_MPH
    cur_sentence.append(get_detail_sentence(make, model, longitudinal, min_enable_speed, min_steer_speed, auto_resume, steering_torque))
  if cur_sentence[0] != cur_sentence[1]:
    return f"- Sentence for {name} changed!\n" + \
            "  ```diff\n" + \
            f"  - {cur_sentence[0]}\n" + \
            f"  + {cur_sentence[1]}\n" + \
            "  ```"

def column_change_format(line1, line2):
  line1, line2 = line1[3:], line2[3:]
  info1, info2 = line1.split('|'), line2.split('|')
  return "".join([f"{i1} {ARROW_SYMBOL} {i2}|" if i1 != i2 else f"{i1}|" for i1, i2 in zip(info1, info2, strict=True)])

def process_diff_information(info):
  header = info[0]
  output = [] # (category, strings)
  if any("> <sup>" in x for x in info):
    return [] # Changes in footnotes ignored
  if any("* " in x for x in info):
    return [] # Changes in Toyota Security ignored
  if any(" Supported Cars" in x for x in info):
    return [] # Changes in number of supported cars ignored
  if "c" in header:
    categories = []
    final_strings = []
    ind = info.index("---")
    remove = {' '.join(line[3:].split("|")[:2]): line for line in info[1:ind]}
    add = {' '.join(line[3:].split("|")[:2]): line for line in info[ind+1:]}
    makes = set(remove.keys()) | set(add.keys())
    for make in makes:
      if make in remove and make in add:
        categories.append('column')
        final_strings.append(column_change_format(remove[make], add[make]))
        diff_detail_sentence = process_detail_sentence(remove[make], add[make])
        if diff_detail_sentence:
          categories.append('detail')
          final_strings.append(diff_detail_sentence)
        del add[make]
        del remove[make]
      elif make in remove:
        categories.append('removals')
        final_strings.append(remove[make][2:])
        del remove[make]
      elif make in add:
        categories.append('additions')
        final_strings.append(add[make][2:])
        del add[make]
    output = list(zip(categories, final_strings, strict=True))
  else:
    category = "additions" if "a" in header else "removals"
    final_strings = [x[2:] for x in info[1:]]
    output = [(category, string) for string in final_strings]
  return output

def print_markdown(changes):
  markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]
  for title, category in (("## 🔀 Column Changes", "column"), ("## ❌ Removed", "removals"),
                          ("## ➕ Added", "additions"), ("## 📖 Detail Sentence Changes", "detail")):
    if len(changes[category]):
      markdown_builder.append(title)
      if category not in ("detail",):
        markdown_builder.append(COLUMNS)
        markdown_builder.append(COLUMN_HEADER)
      markdown_builder.extend(changes[category])
  print("\n".join(markdown_builder))

def print_car_docs_diff(path):
  CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
  MASTER_CARS_MD = os.path.join(path, "CARS.md") # path to the CARS.md in the master branch

  changes = subprocess.run(['diff', MASTER_CARS_MD, CARS_MD_OUT], capture_output=True, text=True).stdout.split('\n')

  changes_markdown = defaultdict(list)
  ind = 0
  while ind < len(changes):
    if changes[ind] and changes[ind][0].isdigit():
      start = ind
      ind += 1
      while ind < len(changes) and changes[ind] and not changes[ind][0].isdigit():
        ind += 1
      diffs = process_diff_information(changes[start:ind])
      for category, strings in diffs:
        changes_markdown[category].append(strings)
    else:
      ind += 1

  if any(len(c) for c in changes_markdown.values()):
    print_markdown(changes_markdown)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  args = parser.parse_args()
  print_car_docs_diff(args.path)
