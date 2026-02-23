#!/usr/bin/env python3
import argparse
import ast
import os
import re
import struct
from dataclasses import dataclass, field
from itertools import chain

from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.multilang import SYSTEM_UI_DIR, UI_DIR, TRANSLATIONS_DIR, multilang

POT_FILE = os.path.join(str(TRANSLATIONS_DIR), "app.pot")

# Functions whose first arg gets the python-format flag
FORMAT_FUNCS = {'tr', 'trn'}

# Plural forms for all supported languages
PLURAL_FORMS = {
  'de': 'nplurals=2; plural=(n != 1);',
  'en': 'nplurals=2; plural=(n != 1);',
  'es': 'nplurals=2; plural=(n != 1);',
  'fr': 'nplurals=2; plural=(n > 1);',
  'ja': 'nplurals=1; plural=0;',
  'ko': 'nplurals=1; plural=0;',
  'pt-BR': 'nplurals=2; plural=(n > 1);',
  'th': 'nplurals=1; plural=0;',
  'tr': 'nplurals=2; plural=(n != 1);',
  'uk': 'nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);',
  'zh-CHS': 'nplurals=1; plural=0;',
  'zh-CHT': 'nplurals=1; plural=0;',
}


@dataclass
class POEntry:
  msgid: str
  msgid_plural: str | None = None
  locations: list[str] = field(default_factory=list)
  has_format_flag: bool = False
  msgstr: str = ""
  plural_msgstrs: dict[int, str] = field(default_factory=dict)
  # Raw pre-formatted lines from parsed .po file, preserved to avoid re-wrapping churn
  raw_msgid_lines: list[str] | None = None
  raw_msgid_plural_lines: list[str] | None = None
  raw_msgstr_lines: list[str] | None = None
  raw_plural_msgstr_lines: dict[int, list[str]] | None = None


# --- String helpers ---

def _po_escape(s: str) -> str:
  return s.replace('\\', '\\\\').replace('"', '\\"').replace('\t', '\\t').replace('\n', '\\n')


def _strip_quotes(s: str) -> str:
  """Remove exactly one pair of surrounding double quotes."""
  if s.startswith('"') and s.endswith('"'):
    return s[1:-1]
  return s


def _po_unescape(s: str) -> str:
  result = []
  i = 0
  while i < len(s):
    if s[i] == '\\' and i + 1 < len(s):
      c = s[i + 1]
      if c == 'n':
        result.append('\n')
      elif c == 't':
        result.append('\t')
      elif c == '\\':
        result.append('\\')
      elif c == '"':
        result.append('"')
      else:
        result.append(s[i:i + 2])
      i += 2
    else:
      result.append(s[i])
      i += 1
  return ''.join(result)


def _format_po_string(keyword: str, s: str) -> str:
  """Format a msgid/msgstr/msgid_plural string with proper line wrapping."""
  escaped = _po_escape(s)

  # If empty string
  if not escaped:
    return f'{keyword} ""'

  # Check if we need multiline format
  # Use multiline if: contains \n (newlines), or the single line would be too long
  has_newlines = '\\n' in escaped
  single_line = f'{keyword} "{escaped}"'

  if not has_newlines and len(single_line) <= 79:
    return single_line

  # Multiline format: keyword "" followed by continuation lines
  lines = [f'{keyword} ""']

  if has_newlines:
    # Split at \n boundaries, keeping \n at end of each segment
    parts = escaped.split('\\n')
    for i, part in enumerate(parts):
      if i < len(parts) - 1:
        segment = part + '\\n'
      else:
        segment = part
      if segment:  # skip empty trailing segment
        # Wrap long segments
        for wrapped in _wrap_line(segment, 77):
          lines.append(f'"{wrapped}"')
  else:
    # Just wrap long line
    for wrapped in _wrap_line(escaped, 77):
      lines.append(f'"{wrapped}"')

  return '\n'.join(lines)


def _wrap_line(s: str, max_width: int) -> list[str]:
  """Wrap a string at word boundaries to fit within max_width chars."""
  if len(s) <= max_width:
    return [s]

  result = []
  while len(s) > max_width:
    # Find a space to break at
    break_at = s.rfind(' ', 0, max_width)
    if break_at == -1:
      break_at = s.find(' ')
      if break_at == -1:
        break
    result.append(s[:break_at + 1])
    s = s[break_at + 1:]
  if s:
    result.append(s)
  return result


# --- AST extraction (replaces xgettext) ---

def _resolve_string(node: ast.expr) -> str | None:
  """Resolve an AST node to a string, handling concatenation."""
  if isinstance(node, ast.Constant) and isinstance(node.value, str):
    return node.value
  if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
    left = _resolve_string(node.left)
    right = _resolve_string(node.right)
    if left is not None and right is not None:
      return left + right
  if isinstance(node, ast.JoinedStr):
    return None  # f-strings are not translatable
  return None


def _get_func_name(node: ast.Call) -> str | None:
  """Get the function name from a Call node (handles Name and Attribute)."""
  if isinstance(node.func, ast.Name):
    return node.func.id
  if isinstance(node.func, ast.Attribute):
    return node.func.attr
  return None


def extract_strings(files: list[str]) -> dict[str, POEntry]:
  """Extract translatable strings from Python source files.

  Entries are ordered by first occurrence (file order, then line number),
  matching xgettext's default output order.
  """
  entries: dict[str, POEntry] = {}

  for filepath in files:
    abs_path = os.path.join(BASEDIR, filepath)
    with open(abs_path, encoding='utf-8') as f:
      source = f.read()

    try:
      tree = ast.parse(source, filename=filepath)
    except SyntaxError:
      continue

    # Collect all tr/trn/tr_noop calls and sort by line number
    calls = []
    for node in ast.walk(tree):
      if isinstance(node, ast.Call):
        func_name = _get_func_name(node)
        if func_name in ('tr', 'trn', 'tr_noop') and node.args:
          calls.append((node.lineno, func_name, node))
    calls.sort(key=lambda x: x[0])

    for lineno, func_name, node in calls:
      msgid = _resolve_string(node.args[0])
      if msgid is None:
        continue

      location = f"{_normalize_path(filepath)}:{lineno}"
      has_format = func_name in FORMAT_FUNCS

      if func_name == 'trn' and len(node.args) >= 2:
        msgid_plural = _resolve_string(node.args[1])
        if msgid_plural is None:
          continue

        if msgid in entries:
          entry = entries[msgid]
          entry.locations.append(location)
          entry.has_format_flag = entry.has_format_flag or has_format
        else:
          entries[msgid] = POEntry(
            msgid=msgid,
            msgid_plural=msgid_plural,
            locations=[location],
            has_format_flag=has_format,
          )
      else:
        if msgid in entries:
          entries[msgid].locations.append(location)
          entries[msgid].has_format_flag = entries[msgid].has_format_flag or has_format
        else:
          entries[msgid] = POEntry(
            msgid=msgid,
            locations=[location],
            has_format_flag=has_format,
          )

  return entries


# --- POT writing ---

POT_HEADER = """\
# SOME DESCRIPTIVE TITLE.
# Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER
# This file is distributed under the same license as the PACKAGE package.
# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
#
#, fuzzy
msgid ""
msgstr ""
"Project-Id-Version: PACKAGE VERSION\\n"
"Report-Msgid-Bugs-To: \\n"
"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\\n"
"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"
"Language-Team: LANGUAGE <LL@li.org>\\n"
"Language: \\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"
"Plural-Forms: nplurals=INTEGER; plural=EXPRESSION;\\n"
"""


def _write_entry(entry: POEntry) -> str:
  """Format a single PO entry as a string."""
  lines = []

  # Location comments
  loc_line = "#: " + " ".join(entry.locations)
  # Wrap location lines at ~79 chars
  if len(loc_line) > 79:
    parts = []
    current = "#:"
    for loc in entry.locations:
      test = current + " " + loc
      if len(test) > 79 and current != "#:":
        parts.append(current)
        current = "#: " + loc
      else:
        current = test
    parts.append(current)
    lines.extend(parts)
  else:
    lines.append(loc_line)

  # Flags
  if entry.has_format_flag:
    lines.append("#, python-format")

  # msgid - use raw lines if available to preserve formatting
  if entry.raw_msgid_lines:
    lines.append('\n'.join(entry.raw_msgid_lines))
  else:
    lines.append(_format_po_string("msgid", entry.msgid))

  if entry.msgid_plural is not None:
    if entry.raw_msgid_plural_lines:
      lines.append('\n'.join(entry.raw_msgid_plural_lines))
    else:
      lines.append(_format_po_string("msgid_plural", entry.msgid_plural))
    # Plural msgstr entries - use raw lines if available to preserve formatting
    if entry.plural_msgstrs:
      for idx in sorted(entry.plural_msgstrs.keys()):
        if entry.raw_plural_msgstr_lines and idx in entry.raw_plural_msgstr_lines:
          lines.append('\n'.join(entry.raw_plural_msgstr_lines[idx]))
        else:
          lines.append(_format_po_string(f"msgstr[{idx}]", entry.plural_msgstrs[idx]))
    else:
      lines.append('msgstr[0] ""')
      lines.append('msgstr[1] ""')
  else:
    # Use raw lines if available to preserve original formatting
    if entry.raw_msgstr_lines:
      lines.append('\n'.join(entry.raw_msgstr_lines))
    else:
      lines.append(_format_po_string("msgstr", entry.msgstr))

  return '\n'.join(lines)


def write_pot(entries: dict[str, POEntry], output_path: str):
  """Write a .pot file from extracted entries."""
  parts = [POT_HEADER.rstrip()]
  for entry in entries.values():
    parts.append(_write_entry(entry))

  with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(parts) + '\n')


# --- PO parsing ---

def _parse_po_file(text: str) -> tuple[str, dict[str, POEntry], str]:
  """Parse a .po/.pot file, returning (header_text, entries_dict, header_comments).

  header_text is the raw msgstr value of the empty-msgid header entry.
  entries_dict maps msgid -> POEntry with translations preserved.
  header_comments is the comment block before the header entry.
  """
  entries: dict[str, POEntry] = {}
  header = ""
  lines = text.splitlines()
  i = 0

  # Extract header comments (everything before the first msgid)
  header_comment_lines = []
  while i < len(lines):
    line = lines[i].strip()
    if line.startswith('msgid '):
      break
    header_comment_lines.append(lines[i])
    i += 1
  header_comments = '\n'.join(header_comment_lines).rstrip()

  while i < len(lines):
    line = lines[i].strip()

    # Skip comments and blank lines (but track comment blocks for entries)
    if not line or line.startswith('#'):
      i += 1
      continue

    # Parse msgid
    if line.startswith('msgid '):
      # Collect location and flag comments above this entry
      locations = []
      has_format = False
      j = i - 1
      while j >= 0:
        prev = lines[j].strip()
        if prev.startswith('#: '):
          locs = prev[3:].split()
          locations = locs + locations
        elif prev.startswith('#,'):
          if 'python-format' in prev:
            has_format = True
        elif prev.startswith('#') or prev == '':
          pass
        else:
          break
        j -= 1

      raw_msgid = [line]
      msgid = _po_unescape(_strip_quotes(line[len('msgid '):]))
      i += 1
      while i < len(lines) and lines[i].strip().startswith('"'):
        msgid += _po_unescape(_strip_quotes(lines[i].strip()))
        raw_msgid.append(lines[i].strip())
        i += 1

      # Check for msgid_plural
      msgid_plural = None
      raw_msgid_plural: list[str] | None = None
      if i < len(lines) and lines[i].strip().startswith('msgid_plural '):
        raw_msgid_plural = [lines[i].strip()]
        msgid_plural = _po_unescape(_strip_quotes(lines[i].strip()[len('msgid_plural '):]))
        i += 1
        while i < len(lines) and lines[i].strip().startswith('"'):
          msgid_plural += _po_unescape(_strip_quotes(lines[i].strip()))
          raw_msgid_plural.append(lines[i].strip())
          i += 1

      if msgid_plural is not None:
        plural_strs: dict[int, str] = {}
        raw_plural_lines: dict[int, list[str]] = {}
        while i < len(lines):
          m = re.match(r'\s*msgstr\[(\d+)\]\s+"(.*)"', lines[i])
          if not m:
            break
          idx = int(m.group(1))
          val = _po_unescape(m.group(2))
          raw = [lines[i].strip()]
          i += 1
          while i < len(lines) and lines[i].strip().startswith('"'):
            val += _po_unescape(_strip_quotes(lines[i].strip()))
            raw.append(lines[i].strip())
            i += 1
          plural_strs[idx] = val
          raw_plural_lines[idx] = raw

        if msgid == '':
          i += 1
          continue

        entries[msgid] = POEntry(
          msgid=msgid,
          msgid_plural=msgid_plural,
          locations=locations,
          has_format_flag=has_format,
          plural_msgstrs=plural_strs,
          raw_msgid_lines=raw_msgid,
          raw_msgid_plural_lines=raw_msgid_plural,
          raw_plural_msgstr_lines=raw_plural_lines,
        )
      else:
        # Regular msgstr
        msgstr = ""
        raw_lines: list[str] = []
        if i < len(lines) and lines[i].strip().startswith('msgstr '):
          msgstr = _po_unescape(_strip_quotes(lines[i].strip()[len('msgstr '):]))
          raw_lines = [lines[i].strip()]
          i += 1
          while i < len(lines) and lines[i].strip().startswith('"'):
            msgstr += _po_unescape(_strip_quotes(lines[i].strip()))
            raw_lines.append(lines[i].strip())
            i += 1
        else:
          i += 1

        if msgid == '':
          header = msgstr
        else:
          entries[msgid] = POEntry(
            msgid=msgid,
            locations=locations,
            has_format_flag=has_format,
            msgstr=msgstr,
            raw_msgid_lines=raw_msgid,
            raw_msgstr_lines=raw_lines,
          )
    else:
      i += 1

  return header, entries, header_comments


def _get_nplurals(header: str) -> int:
  """Extract nplurals from a PO header string."""
  m = re.search(r'nplurals\s*=\s*(\d+)', header)
  return int(m.group(1)) if m else 2


# --- PO merge (replaces msgmerge) ---

def merge_po(po_path: str, pot_entries: dict[str, POEntry]):
  """Merge new .pot entries into an existing .po file."""
  with open(po_path, encoding='utf-8') as f:
    po_text = f.read()

  header, old_entries, header_comments = _parse_po_file(po_text)
  nplurals = _get_nplurals(header)

  merged: list[POEntry] = []
  for pot_entry in sorted(pot_entries.values(), key=lambda e: e.msgid):
    old = old_entries.get(pot_entry.msgid)

    entry = POEntry(
      msgid=pot_entry.msgid,
      msgid_plural=pot_entry.msgid_plural,
      locations=pot_entry.locations,
      has_format_flag=pot_entry.has_format_flag,
    )

    if old is not None:
      # Preserve existing translations and raw formatting
      entry.raw_msgid_lines = old.raw_msgid_lines
      entry.raw_msgid_plural_lines = old.raw_msgid_plural_lines
      if pot_entry.msgid_plural is not None:
        for idx in range(nplurals):
          entry.plural_msgstrs[idx] = old.plural_msgstrs.get(idx, "")
        entry.raw_plural_msgstr_lines = old.raw_plural_msgstr_lines
      else:
        entry.msgstr = old.msgstr
        entry.raw_msgstr_lines = old.raw_msgstr_lines
    else:
      # New entry - empty translations
      if pot_entry.msgid_plural is not None:
        for idx in range(nplurals):
          entry.plural_msgstrs[idx] = ""

    merged.append(entry)

  _write_po(po_path, header, merged, header_comments=header_comments)


# --- PO init (replaces msginit) ---

def init_po(po_path: str, pot_entries: dict[str, POEntry], lang: str):
  """Create a new .po file for a language from .pot entries."""
  plural_form = PLURAL_FORMS.get(lang, 'nplurals=2; plural=(n != 1);')
  nplurals = _get_nplurals(plural_form)

  header = (
    "Project-Id-Version: PACKAGE VERSION\n"
    + "Report-Msgid-Bugs-To: \n"
    + "PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\n"
    + "Last-Translator: Automatically generated\n"
    + "Language-Team: none\n"
    + f"Language: {lang}\n"
    + "MIME-Version: 1.0\n"
    + "Content-Type: text/plain; charset=UTF-8\n"
    + "Content-Transfer-Encoding: 8bit\n"
    + f"Plural-Forms: {plural_form}\n"
  )

  entries: list[POEntry] = []
  for pot_entry in sorted(pot_entries.values(), key=lambda e: e.msgid):
    entry = POEntry(
      msgid=pot_entry.msgid,
      msgid_plural=pot_entry.msgid_plural,
      locations=pot_entry.locations,
      has_format_flag=pot_entry.has_format_flag,
    )
    if pot_entry.msgid_plural is not None:
      for idx in range(nplurals):
        entry.plural_msgstrs[idx] = ""
    entries.append(entry)

  _write_po(po_path, header, entries)


def _write_po(path: str, header: str, entries: list[POEntry], header_comments: str | None = None):
  """Write a .po file with header and entries."""
  parts = []

  # Header comments
  if header_comments:
    parts.append(header_comments.rstrip())
  else:
    parts.append(
      "# Translations for PACKAGE package.\n"
      + "# This file is distributed under the same license as the PACKAGE package.\n"
      + "# Automatically generated.\n"
      + "#"
    )

  # Header entry
  parts.append(
    'msgid ""\n'
    + 'msgstr ""\n'
    + '\n'.join(f'"{_po_escape(line)}\\n"' for line in header.split('\n') if line)
  )

  for entry in entries:
    parts.append(_write_entry(entry))

  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(parts) + '\n')


# --- MO compilation (replaces msgfmt) ---

def compile_mo(po_path: str, mo_path: str):
  """Compile a .po file to a .mo binary file."""
  with open(po_path, encoding='utf-8') as f:
    header_text, entries, _ = _parse_po_file(f.read())

  # Build message pairs: (msgid_bytes, msgstr_bytes)
  # The .mo format requires entries sorted by msgid bytes
  messages: list[tuple[bytes, bytes]] = []

  # Header entry: empty msgid -> header metadata
  messages.append((b'', header_text.encode('utf-8')))

  for entry in entries.values():
    if entry.msgid_plural is not None:
      # Plural: msgid is "singular\0plural", msgstr is "form0\0form1\0..."
      msgid_bytes = entry.msgid.encode('utf-8') + b'\x00' + entry.msgid_plural.encode('utf-8')
      forms = [entry.plural_msgstrs.get(i, '') for i in range(max(entry.plural_msgstrs.keys()) + 1)] if entry.plural_msgstrs else ['']
      msgstr_bytes = b'\x00'.join(f.encode('utf-8') for f in forms)
    else:
      msgid_bytes = entry.msgid.encode('utf-8')
      msgstr_bytes = entry.msgstr.encode('utf-8')
    messages.append((msgid_bytes, msgstr_bytes))

  # Sort by msgid bytes (required by .mo format for binary search)
  messages.sort(key=lambda x: x[0])

  # Build .mo file
  n = len(messages)
  # Header: magic, revision, nstrings, offset_orig, offset_trans, hash_size, hash_offset
  header_size = 7 * 4  # 28 bytes
  # String descriptor tables follow the header
  originals_offset = header_size
  translations_offset = originals_offset + n * 8  # each entry is (length, offset) = 2 uint32
  # String data starts after both descriptor tables
  data_offset = translations_offset + n * 8

  # Collect string data and offsets
  orig_descriptors = []
  trans_descriptors = []
  string_data = bytearray()

  for msgid_bytes, msgstr_bytes in messages:
    # Original string
    orig_descriptors.append((len(msgid_bytes), data_offset + len(string_data)))
    string_data.extend(msgid_bytes)
    string_data.append(0)  # null terminator

    # Translation string
    trans_descriptors.append((len(msgstr_bytes), data_offset + len(string_data)))
    string_data.extend(msgstr_bytes)
    string_data.append(0)  # null terminator

  # Write .mo file
  with open(mo_path, 'wb') as f:
    # Header
    f.write(struct.pack(
      '<7I',
      0x950412de,  # magic number (little-endian)
      0,           # revision
      n,           # number of strings
      originals_offset,
      translations_offset,
      0,           # hash table size (unused)
      0,           # hash table offset (unused)
    ))
    # Original string descriptors
    for length, offset in orig_descriptors:
      f.write(struct.pack('<2I', length, offset))
    # Translation string descriptors
    for length, offset in trans_descriptors:
      f.write(struct.pack('<2I', length, offset))
    # String data
    f.write(string_data)


# --- Main ---

def _normalize_path(path: str) -> str:
  """Strip 'openpilot/' prefix from paths to match expected format.

  importlib.resources.files() resolves to openpilot/selfdrive/ui/ but
  translation location comments should use selfdrive/ui/ paths.
  """
  if path.startswith('openpilot/'):
    return path[len('openpilot/'):]
  return path


def update_translations():
  files = []
  for root, _, filenames in chain(os.walk(SYSTEM_UI_DIR),
                                  os.walk(os.path.join(UI_DIR, "widgets")),
                                  os.walk(os.path.join(UI_DIR, "layouts")),
                                  os.walk(os.path.join(UI_DIR, "onroad"))):
    for filename in filenames:
      if filename.endswith(".py"):
        files.append(os.path.relpath(os.path.join(root, filename), BASEDIR))

  # Extract strings from source files
  entries = extract_strings(files)

  # Write .pot template
  write_pot(entries, POT_FILE)

  # Generate/update translation files for each language
  for name in multilang.languages.values():
    po_path = os.path.join(str(TRANSLATIONS_DIR), f"app_{name}.po")
    if os.path.exists(po_path):
      merge_po(po_path, entries)
    else:
      init_po(po_path, entries, name)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Update translation files")
  parser.add_argument("--vanish", action="store_true",
                      help="Remove obsolete translations (default behavior, kept for compatibility)")
  parser.add_argument("--compile-mo", nargs=2, metavar=("PO_FILE", "MO_FILE"),
                      help="Compile a .po file to .mo binary format")
  args = parser.parse_args()

  if args.compile_mo:
    compile_mo(args.compile_mo[0], args.compile_mo[1])
  else:
    update_translations()
