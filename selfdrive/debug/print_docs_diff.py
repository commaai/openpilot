#!/usr/bin/env python3
import argparse
import requests
import re

ARROW_SYMBOL = "➡️"


def get_cars_docs_in_markdown(docs_content):
  def _parse_markdown_table(md_table):
    lines = md_table.strip().split('\n')
    headers = [h.strip() for h in lines[0].split('|') if h]
    assert len(headers) == len(set(headers)), 'Duplicate headers found in the markdown table'
    return headers, [dict(zip(headers, [cell.strip() for cell in line.split('|')[1:-1]], strict=True)) for line in lines[2:]]

  match = re.search(r"(\d+)\s+supported\s+cars\s*\n([\s\S]*?\|.*?\|[\s\S]*?\|[-:]+\|[\s\S]*?\n(?:[^\S\r\n]*\S.*\n)*\n?)", docs_content, re.IGNORECASE)
  if not match:
    raise RuntimeError("Couldn't find the car docs Markdown table.")
  num_cars, md_table = match.groups()
  num_cars = int(num_cars)
  headers, results = _parse_markdown_table(md_table)
  results = {f'{r['Make']} {r['Model']}': r for r in results}
  assert len(results) == num_cars
  return headers, results


def find_changed_cars(base_cars, new_cars):
  def _limited_to_keys(ks, d):
    return {k: d[k] for k in d if k in ks}
  cars_in_common = [(base_cars[name], new_cars[name]) for name in new_cars.keys() if name in base_cars]
  return [(base, new) for base, new in cars_in_common if base != _limited_to_keys(base.keys(), new)]


def find_removed_cars(base_cars, new_cars):
  return [base_cars[k] for k in base_cars.keys() if k not in new_cars]


def find_added_cars(base_cars, new_cars):
  return [new_cars[k] for k in new_cars.keys() if k not in base_cars]


def build_row(headers, base_car, new_car=None):
  def _preprocess_cell(v):
    v = re.sub(r'\bassets/', 'https://media.githubusercontent.com/media/commaai/openpilot/master/docs/assets/', v)  # replace 'assets/' with the full URL
    v = re.sub(r'\[<sup>(.*?)<\/sup>\]\(#footnotes\)', r'<sup>\1</sup>', v)  # remove links to #footnotes
    return v
  row_builder = []
  for header in headers:
    if new_car and new_car.get(header) != base_car.get(header):
      row_builder.append(f"{_preprocess_cell(base_car.get(header, ''))} {ARROW_SYMBOL} {_preprocess_cell(new_car.get(header, ''))}")
    else:
      row_builder.append(_preprocess_cell(base_car[header]))
  return "|" + "|".join(row_builder) + "|"


def print_car_docs_diff(new_docs_path):
  base_docs_content = requests.get("https://raw.githubusercontent.com/commaai/openpilot/master/docs/CARS.md").text

  with open(new_docs_path) as new_docs_file:
    new_docs_content = new_docs_file.read()

  base_headers, base_cars = get_cars_docs_in_markdown(base_docs_content)
  new_headers, new_cars = get_cars_docs_in_markdown(new_docs_content)

  changes = {
    'column': [build_row(base_headers, base_car, new_car) for base_car, new_car in find_changed_cars(base_cars, new_cars)],
    'removals': [build_row(base_headers, car) for car in find_removed_cars(base_cars, new_cars)],
    'additions': [build_row(new_headers, car) for car in find_added_cars(base_cars, new_cars)],
  }

  # Print diff
  if any(len(c) for c in changes.values()):
    markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]

    for title, category in (("## 🔀 Column Changes", "column"), ("## ❌ Removed", "removals"), ("## ➕ Added", "additions")):
      ordered_headers = base_headers if category != 'additions' else new_headers
      if changes[category]:
        markdown_builder.append(title)
        markdown_builder.append("|" + "|".join(ordered_headers) + "|")
        markdown_builder.append("|---|---|---|{}|".format("|".join([":---:"] * (len(ordered_headers) - 3))))
        markdown_builder.extend(changes[category])

    print("\n".join(markdown_builder))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--new-docs-path", required=True)
  args = parser.parse_args()
  print_car_docs_diff(args.new_docs_path)
