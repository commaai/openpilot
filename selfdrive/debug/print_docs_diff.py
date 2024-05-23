#!/usr/bin/env python3
import argparse
import requests
import os


def download_file(url, local_path):
  response = requests.get(url)
  response.raise_for_status()
  with open(local_path, 'wb') as f:
    f.write(response.content)


def load_file(path):
  with open(path, 'rb') as file:
    return file.read().decode('utf-8', errors='ignore').splitlines()


def parse_markdown(lines):
  cars = {}
  for line in lines:
    if line.startswith('|') and not line.startswith('|---'):
      columns = line.split('|')
      platform = columns[1].strip()
      cars[platform] = line
  return cars


def generate_diff(old_cars, new_cars):
  diffs = []
  for platform, new_line in new_cars.items():
    old_line = old_cars.get(platform)
    if old_line:
      if old_line != new_line:
        diffs.append(f"### Change in {platform}:\n```diff\n- {old_line}+ {new_line}```")
    else:
      diffs.append(f"### Addition: {platform}:\n{new_line}")

  for platform in old_cars.keys() - new_cars.keys():
    diffs.append(f"### Removal: {platform}:\n{old_cars[platform]}")

  return '\n'.join(diffs)


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

  diff = generate_diff(old_cars, new_cars)
  if diff:
    print("### Differences found in CARS.md:\n")
    print(diff)
  else:
    print("No differences found in CARS.md")

  os.remove(base_path)


if __name__ == "__main__":
  main()
