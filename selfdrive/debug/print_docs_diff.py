#!/usr/bin/env python3
import argparse
import os
import requests


def load_file(path):
  with open(path) as file:
    return file.readlines()


def download_file(url, output_path):
  response = requests.get(url)
  response.raise_for_status()
  with open(output_path, 'w') as file:
    file.write(response.text)


def parse_markdown(lines):
  cars = {}
  for line in lines:
    if line.startswith('|') and not line.startswith('|---'):
      columns = line.split('|')
      platform = columns[1].strip()
      cars[platform] = line
  return cars


def generate_diff(old_cars, new_cars):
  changes = {"column": [], "additions": [], "removals": [], "detail": []}

  for platform, new_line in new_cars.items():
    old_line = old_cars.get(platform)
    if old_line:
      if old_line != new_line:
        changes["column"].append(f"| {old_line} | ➡️ | {new_line} |")
    else:
      changes["additions"].append(f"| {new_line} |")

  for platform in old_cars.keys() - new_cars.keys():
    changes["removals"].append(f"| {old_cars[platform]} |")

  return changes


def print_changes(changes):
  markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]
  sections = [("## 🔀 Column Changes", "column"), ("## ➕ Added", "additions"), ("## ❌ Removed", "removals"), ("## 📖 Detail Sentence Changes", "detail")]

  for title, category in sections:
    if changes[category]:
      markdown_builder.append(title)
      markdown_builder.extend(changes[category])

  print("\n".join(markdown_builder))


def main():
  parser = argparse.ArgumentParser(description="Compare two CARS.md files and generate a diff.")
  parser.add_argument('--path', default='docs/CARS.md', help="Path to the new CARS.md file")
  args = parser.parse_args()

  base_url = "https://raw.githubusercontent.com/commaai/openpilot/master/docs/CARS.md"
  base_path = 'old_CARS.md'
  download_file(base_url, base_path)

  old_lines = load_file(base_path)
  new_lines = load_file(args.path)

  old_cars = parse_markdown(old_lines)
  new_cars = parse_markdown(new_lines)

  changes = generate_diff(old_cars, new_cars)

  if any(changes.values()):
    print_changes(changes)
  else:
    print("No differences found in CARS.md")

  os.remove(base_path)


if __name__ == "__main__":
  main()
