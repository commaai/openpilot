#!/usr/bin/env python3
import argparse
from collections import defaultdict
import requests
import os


def download_file(url, local_path):
  response = requests.get(url)
  response.raise_for_status()
  with open(local_path, 'w') as f:
    f.write(response.text)


def load_file(path):
  with open(path, encoding='utf-8', errors='ignore') as file:
    return file.read().splitlines()


def parse_markdown(lines):
  cars = {}
  for line in lines:
    if line.startswith('|') and not line.startswith('|---'):
      columns = line.split('|')
      platform = columns[1].strip()
      model_year = columns[2].strip()
      key = (platform, model_year)
      cars[key] = line
  return cars


def extract_platform_model_year(line):
  parts = line.split('|')
  platform = parts[1].strip()
  model_year = parts[2].strip()
  return platform, model_year


def generate_diff(old_cars, new_cars):
  changes = defaultdict(list)
  all_keys = set(old_cars.keys()) | set(new_cars.keys())

  for key in all_keys:
    old_line = old_cars.get(key)
    new_line = new_cars.get(key)

    if old_line and new_line:
      if old_line != new_line:
        changes["column"].append(f"|{new_line}")
    elif old_line:
      platform, _ = key
      for new_key in new_cars:
        new_platform, new_model_year = new_key
        if new_platform == platform:
          changes["column"].append(f"|{new_cars[new_key]}")
          break
      else:
        changes["removals"].append(f"|{old_line}")
    elif new_line:
      changes["additions"].append(f"|{new_line}")

  markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]

  for title, category in (("## 🔀 Column Changes", "column"), ("## ❌ Removed", "removals"), ("## ➕ Added", "additions")):
    if len(changes[category]):
      markdown_builder.append(title)
      markdown_builder.extend(changes[category])

  return "\n".join(markdown_builder)


def print_car_docs_diff(path):
  base_url = "https://raw.githubusercontent.com/commaai/openpilot/master/docs/CARS.md"
  base_path = 'old_CARS.md'
  download_file(base_url, base_path)

  old_lines = load_file(base_path)
  new_lines = load_file(path)

  old_cars = parse_markdown(old_lines)
  new_cars = parse_markdown(new_lines)

  diff = generate_diff(old_cars, new_cars)
  if diff:
    print(diff)
  else:
    print("No differences found in CARS.md")

  os.remove(base_path)


def main():
  parser = argparse.ArgumentParser(description="Compare two CARS.md files and generate a diff.")
  parser.add_argument('--path', default='docs/CARS.md', help="Path to the new CARS.md file")
  args = parser.parse_args()

  print_car_docs_diff(args.path)


if __name__ == "__main__":
  main()
