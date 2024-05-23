#!/usr/bin/env python3
import argparse
import difflib
import subprocess
import os


def load_file(path):
  with open(path) as file:
    return file.readlines()


def get_last_commit_file(path, output_path):
  result = subprocess.run(['git', 'show', f'HEAD~1:{path}'], capture_output=True, text=True, check=True)
  with open(output_path, 'w') as file:
    file.write(result.stdout)


def generate_diff(base_file, new_file):
  diff = difflib.unified_diff(base_file, new_file, fromfile='old_CARS.md', tofile='new_CARS.md', lineterm='')
  return '\n'.join(diff)


def main():
  parser = argparse.ArgumentParser(description="Compare two CARS.md files and generate a diff.")
  parser.add_argument('--path', default='docs/CARS.md', help="Path to the new CARS.md file")
  args = parser.parse_args()

  base_path = 'old_CARS.md'
  get_last_commit_file(args.path, base_path)

  base_file = load_file(base_path)
  new_file = load_file(args.path)

  diff = generate_diff(base_file, new_file)
  if diff:
    print("### Differences found in CARS.md:\n")
    print(diff)
  else:
    print("No differences found in CARS.md")

  os.remove(base_path)


if __name__ == "__main__":
  main()
