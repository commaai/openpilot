#!/usr/bin/env python3
import argparse
import tokenize
from collections.abc import Iterable


def check_indentation(filename: str, indent_width: int = 2) -> bool:
  failed = False
  indent_stack = [0]

  try:
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
  except (IndentationError, SyntaxError, tokenize.TokenError):
    # Ruff reports syntax and structurally invalid indentation errors. This
    # checker only adds the style check that Ruff cannot configure for 2 spaces.
    pass

  return failed


def check_files(filenames: Iterable[str], indent_width: int = 2) -> int:
  failed = False
  for filename in filenames:
    failed |= check_indentation(filename, indent_width)
  return int(failed)


def main() -> int:
  parser = argparse.ArgumentParser(description="Check Python block indentation.")
  parser.add_argument("filenames", nargs="+")
  parser.add_argument("--indent-width", type=int, default=2)
  args = parser.parse_args()

  if args.indent_width < 1:
    parser.error("--indent-width must be greater than zero")

  return check_files(args.filenames, args.indent_width)


if __name__ == "__main__":
  raise SystemExit(main())
