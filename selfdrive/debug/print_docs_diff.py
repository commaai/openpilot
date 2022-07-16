#!/usr/bin/env python3
import argparse
import pickle

from selfdrive.car.docs import get_all_car_info
from selfdrive.car.docs_definitions import Column

STAR_ICON = '<a href="##"><img valign="top" src="https://raw.githubusercontent.com/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
COLUMN_HEADER = "|---|---|---|:---:|:---:|:---:|:---:|:---:|"
ARROW_SYMBOL = "➡️"


def load_base_car_info(path):
  with open(path, "rb") as f:
    return pickle.load(f)


def get_star_diff(base_car, new_car):
  return [column for column, value in base_car.row.items() if value != new_car.row[column]]


def format_row(builder):
  return "|" + "|".join(builder) + "|"


def print_car_info_diff(path):
  base_car_info = {f"{i.make} {i.model}": i for i in load_base_car_info(path)}
  new_car_info = {f"{i.make} {i.model}": i for i in get_all_car_info()}

  tier_changes = []
  star_changes = []
  removals = []
  additions = []

  # Changes (tier + stars)
  for base_car_model, base_car in base_car_info.items():
    if base_car_model not in new_car_info:
      continue

    new_car = new_car_info[base_car_model]

    # Tier changes
    if base_car.tier != new_car.tier:
      tier_changes.append(f"- Tier for {base_car.make} {base_car.model} changed! ({base_car.tier.name.title()} {ARROW_SYMBOL} {new_car.tier.name.title()})")

    # Star changes
    diff = get_star_diff(base_car, new_car)
    if not len(diff):
      continue

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
