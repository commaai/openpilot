#!/usr/bin/env python3
"""A minimal shellcheck-like static analysis tool for shell scripts.

Covers syntax, unquoted expansions, scalar $@ assignments, and read -r.
"""
import re
import argparse
import subprocess
from pathlib import Path

VARIABLE = re.compile(r"\$(?:\{[^}\n]*\}|[A-Za-z_]\w*|[@*0-9])")
ASSIGNMENT = re.compile(r"[A-Za-z_]\w*(?:\[[^]]*\])?\+?=")
OPAQUE = re.compile(r"\$?\(\([^\n]*?\)\)|\[\[.*?\]\]", re.DOTALL)
HEREDOC = re.compile(r"<<(-?)\s*('[^']+'|\"[^\"]+\"|\\?[A-Za-z_]\w*)")


def commands(text):
  def scan(i=0, end="", pattern_group=False):
    start, quote, words, expansions, documents = i, False, [], [], []
    cases = []
    while i < len(text):
      c = text[i]
      if c == "\\":
        i += 2
        continue
      if c == "'" and not quote:
        i = text.find("'", i + 1) + 1 or len(text)
        continue
      if c == '"':
        quote = not quote
      elif (not quote or text.startswith("$((", i)) and (match := OPAQUE.match(text, i)):
        i = match.end()
        continue
      elif text.startswith("$(", i):
        if not quote:
          expansions.append(i)
        i = yield from scan(i + 2, ")")
        continue
      elif match := VARIABLE.match(text, i):
        if not quote and not match[0].startswith("${#"):
          expansions.append(i)
        i = match.end()
        continue
      elif not quote:
        if text.startswith("<<<", i):
          i += 3
          continue
        if c == "#" and i == start:
          i = text.find("\n", i)
          i = len(text) if i < 0 else i
          start = i
          continue
        if not text.startswith("<<<", i) and (match := HEREDOC.match(text, i)):
          documents.append((match[2].strip("'\"").lstrip("\\"), bool(match[1])))
          i = match.end()
          continue
        if c in " \t\r\n;|&()":
          if start < i:
            words.append((text[start:i], start, expansions))
          if len(words) >= 3 and words[0][0] == "case" and words[-1][0] == "in":
            cases.append(True)
            words = []
          if words and words[0][0] == "esac" and cases:
            cases.pop()
            words = []
          pattern = bool(cases and cases[-1])
          expansions = []
          if c in "\n;|&()":
            if words and not (pattern or pattern_group) and (c != ")" or end):
              yield words
            words = []
          if c == end and not pattern:
            return i + 1
          if c == ")" and pattern:
            cases[-1] = False
          if cases and (terminator := re.match(r";(?:;&|;|&)", text[i:])):
            cases[-1] = True
            i += len(terminator[0]) - 1
          if c == "(":
            if pattern and i == start:
              cases[-1] = False
            i = yield from scan(i + 1, ")", pattern or pattern_group)
            start = i
            continue
          if c == "\n":
            for delimiter, strip_tabs in documents:
              while i < len(text):
                stop = text.find("\n", i + 1)
                stop = len(text) if stop < 0 else stop
                line, i = text[i + 1:stop], stop
                if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                  break
            documents = []
          start = i + 1
      i += 1
    if start < i:
      words.append((text[start:i], start, expansions))
    if words:
      yield words
    return i
  yield from scan()


def check_text(text):
  for words in commands(text):
    command = next((w for w, _, _ in words if not ASSIGNMENT.match(w) and w not in {"if", "then", "elif", "while", "until", "do", "!"}), "")
    prefix = True
    for index, (word, start, expansions) in enumerate(words):
      assignment = ASSIGNMENT.match(word) and (prefix or command in {"export", "local", "declare", "readonly", "typeset"})
      prefix = prefix and (bool(assignment) or word in {"if", "then", "elif", "while", "until", "do", "!"})
      for offset in expansions:
        array = word in {"$@", "$*"} or "[@]" in word or "[*]" in word
        if not assignment and (command not in {"case", "for", "select"} or array) and not (index and words[index - 1][0] == "<<<"):
          yield text.count("\n", 0, offset) + 1, "Quote this expansion to prevent word splitting and globbing"
      if assignment and re.fullmatch(r'"[^"\n]*\$@[^"\n]*"', ASSIGNMENT.sub("", word, count=1)):
        yield text.count("\n", 0, start) + 1, 'Use an array for "$@", or "$*" to join arguments'
    if command == "read" and not any(re.fullmatch(r"-[A-Za-z]*r[A-Za-z0-9]*", w) for w, _, _ in words):
      yield text.count("\n", 0, next(start for w, start, _ in words if w == "read")) + 1, "Use read -r to preserve backslashes"


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("filenames", nargs="+")
  failed = False
  for filename in parser.parse_args().filenames:
    syntax = subprocess.run(["bash", "-n", "--", filename], check=False)
    failed |= syntax.returncode != 0
    if syntax.returncode == 0:
      for line, message in check_text(Path(filename).read_text()):
        print(f"{filename}:{line}: {message}")
        failed = True
  raise SystemExit(failed)
