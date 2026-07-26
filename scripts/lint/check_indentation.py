#!/usr/bin/env python3
import argparse
import tokenize


# TODO: remove this once https://github.com/astral-sh/ruff/issues/8705 is closed
def check_indentation(filename: str, indent_width: int = 2) -> bool:
  failed = False
  indent_stack = [0]

  with tokenize.open(filename) as f:
    tokens = tokenize.generate_tokens(f.readline)
    for token in tokens:
      if token.type == tokenize.INDENT:
        indentation = token.string
        width = len(indentation)
        expected = indent_stack[-1] + indent_width

        if indentation != " " * expected:
          found = "indentation containing tabs" if "\t" in indentation else f"{width} spaces"
          print(f"{filename}:{token.start[0]}:1: expected {expected} spaces, found {found}")
          failed = True
        indent_stack.append(width)
      elif token.type == tokenize.DEDENT:
        indent_stack.pop()

  return failed


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Check Python block indentation.")
  parser.add_argument("filenames", nargs="+")
  args = parser.parse_args()

  failed = False
  for filename in args.filenames:
    failed |= check_indentation(filename)
  raise SystemExit(failed)
