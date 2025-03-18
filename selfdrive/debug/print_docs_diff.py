#!/usr/bin/env python3
import argparse
from collections import defaultdict
import difflib
import pickle

from opendbc.car.docs import get_all_car_docs
from opendbc.car.docs_definitions import Column

FOOTNOTE_TAG = "<sup>{}</sup>"
STAR_ICON = '<a href="##"><img valign="top" ' + \
            'src="https://media.githubusercontent.com/media/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
VIDEO_ICON = '<a href="{}" target="_blank">' + \
             '<img height="18px" src="https://media.githubusercontent.com/media/commaai/openpilot/master/docs/assets/icon-youtube.svg"></img></a>'
COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
COLUMN_HEADER = "|---|---|---|{}|".format("|".join([":---:"] * (len(Column) - 3)))
ARROW_SYMBOL = "➡️"


def load_base_car_docs(path):
  with open(path, "rb") as f:
    return pickle.load(f)


def match_cars(base_cars, new_cars):
  changes = []
  additions = []
  for new in new_cars:
    # Addition if no close matches or close match already used
    # Change if close match and not already used
    matches = difflib.get_close_matches(new.name, [b.name for b in base_cars], cutoff=0.)
    if not len(matches) or matches[0] in [c[1].name for c in changes]:
      additions.append(new)
    else:
      changes.append((new, next(car for car in base_cars if car.name == matches[0])))

  # Removal if base car not in changes
  removals = [b for b in base_cars if b.name not in [c[1].name for c in changes]]
  return changes, additions, removals


def build_column_diff(base_car, new_car):
  row_builder = []
  for column in Column:
    base_column = base_car.get_column(column, STAR_ICON, VIDEO_ICON, FOOTNOTE_TAG)
    new_column = new_car.get_column(column, STAR_ICON, VIDEO_ICON, FOOTNOTE_TAG)

    if base_column != new_column:
      row_builder.append(f"{base_column} {ARROW_SYMBOL} {new_column}")
    else:
      row_builder.append(new_column)

  return format_row(row_builder)


def format_row(builder):
  return "|" + "|".join(builder) + "|"


def print_car_docs_diff(path):
  base_car_docs = defaultdict(list)
  new_car_docs = defaultdict(list)

  for car in load_base_car_docs(path):
    base_car_docs[car.car_fingerprint].append(car)
  for car in get_all_car_docs():
    new_car_docs[car.car_fingerprint].append(car)

  # Add new platforms to base cars so we can detect additions and removals in one pass
  base_car_docs.update({car: [] for car in new_car_docs if car not in base_car_docs})

  changes = defaultdict(list)
  for base_car_model, base_cars in base_car_docs.items():
    # Match car info changes, and get additions and removals
    new_cars = new_car_docs[base_car_model]
    car_changes, car_additions, car_removals = match_cars(base_cars, new_cars)

    # Removals
    for car_docs in car_removals:
      changes["removals"].append(format_row([car_docs.get_column(column, STAR_ICON, VIDEO_ICON, FOOTNOTE_TAG) for column in Column]))

    # Additions
    for car_docs in car_additions:
      changes["additions"].append(format_row([car_docs.get_column(column, STAR_ICON, VIDEO_ICON, FOOTNOTE_TAG) for column in Column]))

    for new_car, base_car in car_changes:
      # Column changes
      row_diff = build_column_diff(base_car, new_car)
      if ARROW_SYMBOL in row_diff:
        changes["column"].append(row_diff)

      # Detail sentence changes
      if base_car.detail_sentence != new_car.detail_sentence:
        changes["detail"].append(f"- Sentence for {base_car.name} changed!\n" +
                                 "  ```diff\n" +
                                 f"  - {base_car.detail_sentence}\n" +
                                 f"  + {new_car.detail_sentence}\n" +
                                 "  ```")

  # Print diff
  if any(len(c) for c in changes.values()):
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  args = parser.parse_args()
  print_car_docs_diff(args.path)
