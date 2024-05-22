#!/usr/bin/env python3
import argparse
import difflib

def load_file(path):
    with open(path) as file:
        return file.readlines()

def generate_diff(base_file, new_file):
    diff = difflib.unified_diff(
        base_file,
        new_file,
        fromfile='base/CARS.md',
        tofile='new/CARS.md',
        lineterm=''
    )
    return '\n'.join(diff)

def main():
    parser = argparse.ArgumentParser(description="Compare two CARS.md files and generate a diff.")
    parser.add_argument('base_path', help="Path to the base CARS.md file")
    parser.add_argument('new_path', help="Path to the new CARS.md file")
    args = parser.parse_args()

    base_file = load_file(args.base_path)
    new_file = load_file(args.new_path)

    diff = generate_diff(base_file, new_file)
    print(diff)

if __name__ == "__main__":
    main()
