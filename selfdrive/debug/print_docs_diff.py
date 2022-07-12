#!/usr/bin/env python3
import os
import pickle
from collections import defaultdict

from common.basedir import BASEDIR
from selfdrive.car.docs import get_all_car_info
from selfdrive.car.docs_definitions import Column

STAR_ICON = '<a href="##"><img valign="top" src="https://raw.githubusercontent.com/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
COLUMNS = "|" + "|".join([column.value for column in Column] + ["Tier"]) + "|"
COLUMN_HEADER = "|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|"
ARROW_SYMBOL = "➡️"
EXCLUDE_COLUMNS = [Column.MAKE, Column.MODEL]  # these are used as keys, so exclude diffs


def pretty_row(row, exclude=[Column.MAKE, Column.MODEL]):
  return {k.value: v for k, v in row.items() if k not in exclude}


def load_base_car_info():
  with open(os.path.join(BASEDIR, "../openpilot_cache/old_car_info"), "rb") as f:  # TODO: rename to base
    return pickle.load(f)


def get_diff(base_car, new_car):
  diff = []
  for column, value in base_car.row.items():
    if value != new_car.row[column]:
      diff.append(column)
  if base_car.tier != new_car.tier:
    diff.append("tier")
  return diff


def format_row(builder):
  return "|" + "|".join(builder) + "|"


def print_car_info_diff():
  base_car_info = {f"{i.make} {i.model}": i for i in load_base_car_info()}
  new_car_info = {f"{i.make} {i.model}": i for i in get_all_car_info()}

  changes = []
  removals = []
  additions = []

  # Changes
  for base_car_model, base_car in base_car_info.items():
    if base_car_model not in new_car_info:
      continue

    new_car = new_car_info[base_car_model]
    diff = get_diff(base_car, new_car)
    if not len(diff):
      continue

    row_builder = []
    for column in list(Column) + ["tier"]:
      if column not in diff:
        row_builder.append(new_car.get_column(column, STAR_ICON, "{}"))
      else:
        row_builder.append(base_car.get_column(column, STAR_ICON, "{}") + ARROW_SYMBOL + new_car.get_column(column, STAR_ICON, "{}"))

    changes.append(format_row(row_builder))

  # Changes
  for model in set(base_car_info) - set(new_car_info):
    car_info = base_car_info[model]
    removals.append(format_row([car_info.get_column(column, STAR_ICON, "{}") for column in Column]))

  # Additions
  for model in set(new_car_info) - set(base_car_info):
    car_info = new_car_info[model]
    additions.append(format_row([car_info.get_column(column, STAR_ICON, "{}") for column in Column]))

  # Print diff
  if len(changes) or len(removals) or len(additions):
    markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]
    for title, category in (("## 🔀 Changes", changes), ("## ❌ Removed", removals), ("## ➕ Added", additions)):
      if len(category):
        markdown_builder.append(title)
        markdown_builder.append(COLUMNS)
        markdown_builder.append(COLUMN_HEADER)
        markdown_builder.extend(category)

    print("\n".join(markdown_builder))


if __name__ == "__main__":
  print_car_info_diff()
