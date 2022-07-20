#!/usr/bin/env python3
import argparse
from collections import defaultdict
import difflib
import pickle

from selfdrive.car.docs import get_all_car_info
from selfdrive.car.docs_definitions import Column

STAR_ICON = '<a href="##"><img valign="top" src="https://raw.githubusercontent.com/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
COLUMN_HEADER = "|---|---|---|:---:|:---:|:---:|:---:|"
ARROW_SYMBOL = "➡️"


def match_cars(old_cars, new_cars):
  changes = []
  additions = []
  for new in new_cars:
    closest_match = difflib.get_close_matches(new.name, [c.name for c in old_cars])[0]
    if closest_match not in [i[1].name for i in changes]:
      changes.append((new, next(car for car in old_cars if car.name == closest_match)))
      # changes[new.name] = next(car for car in old_cars if car.name == closest_match)
    else:
      additions.append(new)
    # print(new, difflib.get_close_matches(new, old_cars))
  removals = [i for i in old_cars if i.name not in [k[1].name for k in changes]]
  return changes, additions, removals


def load_base_car_info(path):
  with open(path, "rb") as f:
    return pickle.load(f)


def get_column_diff(base_car, new_car):
  return [column for column in Column if base_car.get_column(column, STAR_ICON, "{}") != new_car.get_column(column, STAR_ICON, "{}")]


def format_row(builder):
  return "|" + "|".join(builder) + "|"


def print_car_info_diff(path):
  base_car_info = defaultdict(list)
  new_car_info = defaultdict(list)
  for car in load_base_car_info(path):
    base_car_info[car.car_fingerprint].append(car)
  for car in get_all_car_info():
    new_car_info[car.car_fingerprint].append(car)

  # Create empty lists so we can check additions and removals in one pass
  base_car_info.update({f: [] for f in new_car_info if f not in base_car_info})
  new_car_info.update({f: [] for f in base_car_info if f not in new_car_info})

  tier_changes = []
  column_changes = []
  removals = []
  additions = []

  for base_car_model, base_cars in base_car_info.items():
    # Match car info changes, and get additions and removals
    new_cars = new_car_info[base_car_model]
    car_changes, car_additions, car_removals = match_cars(base_cars, new_cars)

    # Removals
    for car_info in car_removals:
      removals.append(format_row([car_info.get_column(column, STAR_ICON, "{}") for column in Column]))

    # Additions
    for car_info in car_additions:
      additions.append(format_row([car_info.get_column(column, STAR_ICON, "{}") for column in Column]))

    for new_car, base_car in car_changes:
      # Tier changes
      if base_car.tier != new_car.tier:
        tier_changes.append(f"- Tier for {base_car.make} {base_car.model} changed! ({base_car.tier.name.title()} {ARROW_SYMBOL} {new_car.tier.name.title()})")

      # Column changes
      diff = get_column_diff(base_car, new_car)
      if not len(diff):
        continue

      # TODO: combine with above get_column_diff
      row_builder = []
      for column in list(Column):
        if column not in diff:
          row_builder.append(new_car.get_column(column, STAR_ICON, "{}"))
        else:
          row_builder.append(base_car.get_column(column, STAR_ICON, "{}") + ARROW_SYMBOL + new_car.get_column(column, STAR_ICON, "{}"))

      column_changes.append(format_row(row_builder))

  # Print diff
  if len(column_changes) or len(tier_changes) or len(removals) or len(additions):
    markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]

    for title, category in (("## 🏅 Tier Changes", tier_changes), ("## 🔀 Column Changes", column_changes), ("## ❌ Removed", removals), ("## ➕ Added", additions)):
      if len(category):
        markdown_builder.append(title)
        if "Tier" not in title:
          markdown_builder.append(COLUMNS)
          markdown_builder.append(COLUMN_HEADER)
        markdown_builder.extend(category)

    print("\n".join(markdown_builder))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  args = parser.parse_args()
  # path = '/home/batman/base_car_info'
  print_car_info_diff(args.path)
