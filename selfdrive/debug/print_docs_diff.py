#!/usr/bin/env python3
import argparse
from collections import defaultdict
import difflib
import pickle

from selfdrive.car.docs import get_all_car_info
from selfdrive.car.docs_definitions import Column

STAR_ICON = '<a href="##"><img valign="top" src="https://raw.githubusercontent.com/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
COLUMN_HEADER = "|---|---|---|:---:|:---:|:---:|:---:|:---:|"
ARROW_SYMBOL = "➡️"


def match_cars(old_cars, new_cars):
  changes = []
  for new in new_cars:
    closest_match = difflib.get_close_matches(new.name, [c.name for c in old_cars])[0]
    if closest_match not in [i[1].name for i in changes]:
      changes.append((new, next(car for car in old_cars if car.name == closest_match)))
      # changes[new.name] = next(car for car in old_cars if car.name == closest_match)
    # else:
    #   additions.append(new.name)
    # print(new, difflib.get_close_matches(new, old_cars))
  return changes  #  , additions

def load_base_car_info(path):
  with open(path, "rb") as f:
    return pickle.load(f)


def get_star_diff(base_car, new_car):
  # print(base_car.row)
  # print(new_car.row)
  # print([base_car.get_column(i, STAR_ICON, "{}") for i in Column])
  return [column for column in Column if base_car.get_column(column, STAR_ICON, "{}") != new_car.get_column(column, STAR_ICON, "{}")]
  # return [column for column, value in base_car.row.items() if value != new_car.row[column]]


def format_row(builder):
  return "|" + "|".join(builder) + "|"


def print_car_info_diff(path):
  # base_car_info = {f"{i.make} {i.model}": i for i in load_base_car_info(path)}
  # new_car_info = {f"{i.make} {i.model}": i for i in get_all_car_info()}

  base_car_info = defaultdict(list)
  new_car_info = defaultdict(list)

  for car in load_base_car_info(path):
    base_car_info[car.car_fingerprint].append(car)

  for car in get_all_car_info():
    new_car_info[car.car_fingerprint].append(car)

  tier_changes = []
  star_changes = []  # TODO: rename column changes?
  removals = []
  additions = []

  # Changes (tier + columns)
  for base_car_model, base_cars in base_car_info.items():
    if base_car_model not in new_car_info:
      continue

    new_cars = new_car_info[base_car_model]

    if not any(['RAV4' in i.name for i in base_cars]):
      continue

    # print(base_cars, new_cars)
    matches = match_cars(base_cars, new_cars)
    # print(matches)
    # changes = get_column_changes()

    # Tier changes
    for base_car, new_car in matches:
      if base_car.tier != new_car.tier:
        tier_changes.append(f"- Tier for {base_car.make} {base_car.model} changed! ({base_car.tier.name.title()} {ARROW_SYMBOL} {new_car.tier.name.title()})")

      # Star changes
      print(base_car, new_car)
      diff = get_star_diff(base_car, new_car)
      print('Diff', diff)
      if not len(diff):
        continue

      # TODO: combine with above get_star_diff
      row_builder = []
      for column in list(Column):
        if column not in diff:
          row_builder.append(new_car.get_column(column, STAR_ICON, "{}"))
        else:
          row_builder.append(base_car.get_column(column, STAR_ICON, "{}") + ARROW_SYMBOL + new_car.get_column(column, STAR_ICON, "{}"))

      star_changes.append(format_row(row_builder))

  # Removals
  for model in set(base_car_info) - set(new_car_info):
    car_info = base_car_info[model]
    removals.append(format_row([car_info.get_column(column, STAR_ICON, "{}") for column in Column]))

  # Additions
  for model in set(new_car_info) - set(base_car_info):
    car_info = new_car_info[model]
    additions.append(format_row([car_info.get_column(column, STAR_ICON, "{}") for column in Column]))

  # Print diff
  if len(star_changes) or len(tier_changes) or len(removals) or len(additions):
    markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]

    for title, category in (("## 🏅 Tier Changes", tier_changes), ("## 🔀 Star Changes", star_changes), ("## ❌ Removed", removals), ("## ➕ Added", additions)):
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
  print_car_info_diff(args.path)
