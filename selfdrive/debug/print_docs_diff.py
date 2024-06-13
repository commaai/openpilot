#!/usr/bin/env python3
import argparse
from collections import defaultdict
import difflib

from openpilot.selfdrive.car.docs import CARS_MD_OUT
from openpilot.selfdrive.car.docs_definitions import Column

COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
MODEL_INDEX = COLUMNS.split('|').index(Column.MODEL.value)
MAKE_INDEX = COLUMNS.split('|').index(Column.MAKE.value)
COLUMN_HEADER = "|---|---|---|{}|".format("|".join([":---:"] * (len(Column) - 3)))
ARROW_SYMBOL = "➡️"


def get_model_name(row):
  return row.split('|')[MODEL_INDEX]


def find_model_match(find, rows):
  matches = {}
  for row in rows:
    ratio = difflib.SequenceMatcher(a=get_model_name(find), b=get_model_name(row)).ratio()
    if ratio > 0.6:
      matches[row] = ratio
  return max(matches, key=matches.get) if matches else []


# detail sentences are stored in the MAKE column, after the make as `[](## "<sentence>")
def extract_detail_sentence(row: str) -> tuple[str, str]:
  cols = row.split('|')
  split = cols[MAKE_INDEX].split('[](## "')
  cols[MAKE_INDEX] = split[0]
  detail = split[1][:-2] if len(split) > 1 else ''
  return '|'.join(cols), detail


def get_table_changes(table_row_diffs):
  table_changes = defaultdict(list)
  detail_changes = defaultdict(list)

  new_rows = [row_diff[2:] for row_diff in table_row_diffs if row_diff.startswith('+ ')]
  old_rows = [row_diff[2:] for row_diff in table_row_diffs if row_diff.startswith('- ')]

  for old_row in old_rows:
    new_row = find_model_match(old_row, new_rows)
    if new_row:
      # if we found a match, new_row is not new
      new_rows.remove(new_row)
      new_row, new_detail = extract_detail_sentence(new_row)
      old_row, old_detail = extract_detail_sentence(old_row)

      if new_detail != old_detail:
        detail_changes[get_model_name(new_row)] = (f'- {old_detail}', f'+ {new_detail}')

      if new_row != old_row:
        table_changes['changes'].append('|'.join([a if a == b else f'{a} {ARROW_SYMBOL} {b}' \
                                                 for a, b in zip(old_row.split('|'), new_row.split('|'), strict=True)]))
    else:
      table_changes['removals'].append(old_row)

    table_changes['additions'] = new_rows

  return table_changes, detail_changes


def print_diff(table_changes: dict, detail_changes: dict, other_diffs: list):
  if table_changes:
    print("### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️")
    for change_type in table_changes:
      print(f'## {change_type.capitalize()}\n{COLUMNS}\n{COLUMN_HEADER}')
      for row in table_changes[change_type]:
        print(row)

  if detail_changes:
    print('## Detail Sentence Changes')
    for car_model in detail_changes:
      print(f'- Sentence for {car_model} changed:\n```')
      for sentence in detail_changes[car_model]:
        print(sentence)
      print('```')

  if other_diffs:
    print('## Other Changes\n```diff\n' + '\n'.join(other_diffs) + '\n```')


def check_diff(new_cars_md, old_cars_md):
  differ = difflib.Differ()
  diff = differ.compare(new_cars_md, old_cars_md)

  changed_lines = [line for line in diff if line.startswith(('- ', '+ '))]

  if len(changed_lines):
    table_row_diffs = [line.strip() for line in changed_lines if '|' in line]
    other_diffs = [line.strip() for line in changed_lines if '|' not in line]

    table_changes, detail_changes = get_table_changes(table_row_diffs)

    print_diff(table_changes, detail_changes, other_diffs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--new", default=CARS_MD_OUT)
  parser.add_argument("--old", required=True)
  args = parser.parse_args()

  with open(args.new) as new_file, open(args.old) as old_file:
    new_cars_md = new_file.readlines()
    old_cars_md = old_file.readlines()

  check_diff(new_cars_md, old_cars_md)
