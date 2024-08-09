#!/usr/bin/env python3
import argparse
from collections import defaultdict
import os
import subprocess

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car.conversions import Conversions as CV
from openpilot.selfdrive.car.doc_enums import Column, Star
from openpilot.selfdrive.car.detail_sentences import get_detail_sentence

FOOTNOTE_TAG = "<sup>{}</sup>"
STAR_ICON = '<a href="##"><img valign="top" ' + \
            'src="https://media.githubusercontent.com/media/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
VIDEO_ICON = '<a href="{}" target="_blank">' + \
             '<img height="18px" src="https://media.githubusercontent.com/media/commaai/openpilot/master/docs/assets/icon-youtube.svg"></img></a>'
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

def format_line(line1, line2=None):
  """
  Formats the line to show the changes between the two lines.
  If line2 is not provided, it is assumed to be the same as line1.
  This lets us format line1 on its own and handle the stars/video icons.
  """
  if not line2:
    line2 = line1
  line1, line2 = line1[3:], line2[3:]
  info1, info2 = line1.split('|'), line2.split('|')
  row = "|"
  for i1, i2 in zip(info1, info2, strict=True):
    if '![star](assets/icon-star-' in i1 + i2: # Handle the star icons
      for star_type in Star:
        if star_type.value in i1:
          i1 = STAR_ICON.format(star_type.value)
        if star_type.value in i2:
          i2 = STAR_ICON.format(star_type.value)
    if 'icon-youtube.svg' in i1 + i2: # Handle the video icons
      if i1:
        link = i1[i1.index('href="')+6:i1.index('" target')]
        i1 = VIDEO_ICON.format(link)
      if i2:
        link = i2[i2.index('href="')+6:i2.index('" target')]
        i2 = VIDEO_ICON.format(link)
    if i1 != i2:
      row += f"{i1} {ARROW_SYMBOL} {i2}|"
    else:
      row += f"{i1}|"
  return row

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
        final_strings.append(format_line(remove[make], add[make]))
        diff_detail_sentence = process_detail_sentence(remove[make], add[make])
        if diff_detail_sentence:
          categories.append('detail')
          final_strings.append(diff_detail_sentence)
        del add[make]
        del remove[make]
      elif make in remove:
        categories.append('removals')
        final_strings.append(format_line(remove[make]))
        del remove[make]
      elif make in add:
        categories.append('additions')
        final_strings.append(format_line(add[make]))
        del add[make]
    output = list(zip(categories, final_strings, strict=True))
  else:
    category = "additions" if "a" in header else "removals"
    output = [(category, format_line(line)) for line in info[1:]]
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
