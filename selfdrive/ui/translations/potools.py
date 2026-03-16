"""Pure Python tools for managing .po translation files.

Replaces GNU gettext CLI tools (xgettext, msginit, msgmerge) with Python
implementations for extracting, creating, and updating .po files.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class POEntry:
  msgid: str = ""
  msgstr: str = ""
  msgid_plural: str = ""
  msgstr_plural: dict[int, str] = field(default_factory=dict)
  comments: list[str] = field(default_factory=list)
  source_refs: list[str] = field(default_factory=list)
  flags: list[str] = field(default_factory=list)

  @property
  def is_plural(self) -> bool:
    return bool(self.msgid_plural)


# ──── PO file parsing ────

def _parse_quoted(s: str) -> str:
  """Parse a PO-format quoted string, handling escape sequences."""
  s = s.strip()
  if not (s.startswith('"') and s.endswith('"')):
    raise ValueError(f"Expected quoted string: {s!r}")
  s = s[1:-1]
  result = []
  i = 0
  while i < len(s):
    if s[i] == '\\' and i + 1 < len(s):
      c = s[i + 1]
      if c == 'n':
        result.append('\n')
      elif c == 't':
        result.append('\t')
      elif c == '"':
        result.append('"')
      elif c == '\\':
        result.append('\\')
      else:
        result.append(s[i:i + 2])
      i += 2
    else:
      result.append(s[i])
      i += 1
  return ''.join(result)


def parse_po(path: str | Path) -> tuple[POEntry | None, list[POEntry]]:
  """Parse a .po/.pot file. Returns (header_entry, entries)."""
  with open(path, encoding='utf-8') as f:
    lines = f.readlines()

  entries: list[POEntry] = []
  header: POEntry | None = None
  cur: POEntry | None = None
  cur_field: str | None = None
  plural_idx = 0

  def finish():
    nonlocal cur, header
    if cur is None:
      return
    if cur.msgid == "" and cur.msgstr:
      header = cur
    elif cur.msgid != "" or cur.is_plural:
      entries.append(cur)
    cur = None

  for raw in lines:
    line = raw.rstrip('\n')
    stripped = line.strip()

    if not stripped:
      finish()
      cur_field = None
      continue

    # Skip obsolete entries
    if stripped.startswith('#~'):
      continue

    if stripped.startswith('#'):
      if cur is None:
        cur = POEntry()
      if stripped.startswith('#:'):
        cur.source_refs.append(stripped[2:].strip())
      elif stripped.startswith('#,'):
        cur.flags.extend(f.strip() for f in stripped[2:].split(',') if f.strip())
      else:
        cur.comments.append(line)
      continue

    if stripped.startswith('msgid_plural '):
      if cur is None:
        cur = POEntry()
      cur.msgid_plural = _parse_quoted(stripped[len('msgid_plural '):])
      cur_field = 'msgid_plural'
      continue

    if stripped.startswith('msgid '):
      if cur is None:
        cur = POEntry()
      cur.msgid = _parse_quoted(stripped[len('msgid '):])
      cur_field = 'msgid'
      continue

    m = re.match(r'msgstr\[(\d+)]\s+(.*)', stripped)
    if m:
      plural_idx = int(m.group(1))
      cur.msgstr_plural[plural_idx] = _parse_quoted(m.group(2))
      cur_field = 'msgstr_plural'
      continue

    if stripped.startswith('msgstr '):
      cur.msgstr = _parse_quoted(stripped[len('msgstr '):])
      cur_field = 'msgstr'
      continue

    if stripped.startswith('"'):
      val = _parse_quoted(stripped)
      if cur_field == 'msgid':
        cur.msgid += val
      elif cur_field == 'msgid_plural':
        cur.msgid_plural += val
      elif cur_field == 'msgstr':
        cur.msgstr += val
      elif cur_field == 'msgstr_plural':
        cur.msgstr_plural[plural_idx] += val

  finish()
  return header, entries


# ──── PO file writing ────

def _quote(s: str) -> str:
  """Quote a string for .po file output."""
  s = s.replace('\\', '\\\\').replace('"', '\\"').replace('\t', '\\t')
  if '\n' in s and s != '\n':
    parts = s.split('\n')
    lines = ['""']
    for i, part in enumerate(parts):
      text = part + ('\\n' if i < len(parts) - 1 else '')
      if text:
        lines.append(f'"{text}"')
    return '\n'.join(lines)
  return f'"{s}"'.replace('\n', '\\n')


def write_po(path: str | Path, header: POEntry | None, entries: list[POEntry]) -> None:
  """Write a .po/.pot file."""
  with open(path, 'w', encoding='utf-8') as f:
    if header:
      for c in header.comments:
        f.write(c + '\n')
      if header.flags:
        f.write('#, ' + ', '.join(header.flags) + '\n')
      f.write(f'msgid {_quote("")}\n')
      f.write(f'msgstr {_quote(header.msgstr)}\n\n')

    for entry in entries:
      for c in entry.comments:
        f.write(c + '\n')
      for ref in entry.source_refs:
        f.write(f'#: {ref}\n')
      if entry.flags:
        f.write('#, ' + ', '.join(entry.flags) + '\n')
      f.write(f'msgid {_quote(entry.msgid)}\n')
      if entry.is_plural:
        f.write(f'msgid_plural {_quote(entry.msgid_plural)}\n')
        for idx in sorted(entry.msgstr_plural):
          f.write(f'msgstr[{idx}] {_quote(entry.msgstr_plural[idx])}\n')
      else:
        f.write(f'msgstr {_quote(entry.msgstr)}\n')
      f.write('\n')


# ──── String extraction (replaces xgettext) ────

def extract_strings(files: list[str], basedir: str) -> list[POEntry]:
  """Extract tr/trn/tr_noop calls from Python source files."""
  seen: dict[str, POEntry] = {}

  for filepath in files:
    full = os.path.join(basedir, filepath)
    with open(full, encoding='utf-8') as f:
      source = f.read()
    try:
      tree = ast.parse(source, filename=filepath)
    except SyntaxError:
      continue

    for node in ast.walk(tree):
      if not isinstance(node, ast.Call):
        continue

      func = node.func
      if isinstance(func, ast.Name):
        name = func.id
      elif isinstance(func, ast.Attribute):
        name = func.attr
      else:
        continue

      if name not in ('tr', 'trn', 'tr_noop'):
        continue

      ref = f'{filepath}:{node.lineno}'
      is_flagged = name in ('tr', 'trn')

      if name in ('tr', 'tr_noop'):
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
          continue
        msgid = node.args[0].value
        if msgid in seen:
          if ref not in seen[msgid].source_refs:
            seen[msgid].source_refs.append(ref)
        else:
          flags = ['python-format'] if is_flagged else []
          seen[msgid] = POEntry(msgid=msgid, source_refs=[ref], flags=flags)

      elif name == 'trn':
        if len(node.args) < 2:
          continue
        a1, a2 = node.args[0], node.args[1]
        if not (isinstance(a1, ast.Constant) and isinstance(a1.value, str)):
          continue
        if not (isinstance(a2, ast.Constant) and isinstance(a2.value, str)):
          continue
        msgid, msgid_plural = a1.value, a2.value
        if msgid in seen:
          if ref not in seen[msgid].source_refs:
            seen[msgid].source_refs.append(ref)
        else:
          flags = ['python-format'] if is_flagged else []
          seen[msgid] = POEntry(
            msgid=msgid, msgid_plural=msgid_plural,
            source_refs=[ref], flags=flags,
            msgstr_plural={0: '', 1: ''},
          )

  return list(seen.values())


# ──── POT generation ────

def generate_pot(entries: list[POEntry], pot_path: str | Path) -> None:
  """Generate a .pot template file from extracted entries."""
  now = datetime.now(UTC).strftime('%Y-%m-%d %H:%M%z')
  header = POEntry(
    comments=[
      '# SOME DESCRIPTIVE TITLE.',
      "# Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER",
      '# This file is distributed under the same license as the PACKAGE package.',
      '# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.',
      '#',
    ],
    flags=['fuzzy'],
    msgstr='Project-Id-Version: PACKAGE VERSION\n' +
      'Report-Msgid-Bugs-To: \n' +
      f'POT-Creation-Date: {now}\n' +
      'PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\n' +
      'Last-Translator: FULL NAME <EMAIL@ADDRESS>\n' +
      'Language-Team: LANGUAGE <LL@li.org>\n' +
      'Language: \n' +
      'MIME-Version: 1.0\n' +
      'Content-Type: text/plain; charset=UTF-8\n' +
      'Content-Transfer-Encoding: 8bit\n' +
      'Plural-Forms: nplurals=INTEGER; plural=EXPRESSION;\n',
  )
  write_po(pot_path, header, entries)


# ──── PO init (replaces msginit) ────

PLURAL_FORMS: dict[str, str] = {
  'en': 'nplurals=2; plural=(n != 1);',
  'de': 'nplurals=2; plural=(n != 1);',
  'fr': 'nplurals=2; plural=(n > 1);',
  'es': 'nplurals=2; plural=(n != 1);',
  'pt-BR': 'nplurals=2; plural=(n > 1);',
  'tr': 'nplurals=2; plural=(n != 1);',
  'uk': 'nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<12 || n%100>14) ? 1 : 2);',
  'th': 'nplurals=1; plural=0;',
  'zh-CHT': 'nplurals=1; plural=0;',
  'zh-CHS': 'nplurals=1; plural=0;',
  'ko': 'nplurals=1; plural=0;',
  'ja': 'nplurals=1; plural=0;',
}


def init_po(pot_path: str | Path, po_path: str | Path, language: str) -> None:
  """Create a new .po file from a .pot template (replaces msginit)."""
  _, entries = parse_po(pot_path)
  plural_forms = PLURAL_FORMS.get(language, 'nplurals=2; plural=(n != 1);')
  now = datetime.now(UTC).strftime('%Y-%m-%d %H:%M%z')

  header = POEntry(
    comments=[
      f'# {language} translations for PACKAGE package.',
      "# Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER",
      '# This file is distributed under the same license as the PACKAGE package.',
      '# Automatically generated.',
      '#',
    ],
    msgstr='Project-Id-Version: PACKAGE VERSION\n' +
      'Report-Msgid-Bugs-To: \n' +
      f'POT-Creation-Date: {now}\n' +
      f'PO-Revision-Date: {now}\n' +
      'Last-Translator: Automatically generated\n' +
      'Language-Team: none\n' +
      f'Language: {language}\n' +
      'MIME-Version: 1.0\n' +
      'Content-Type: text/plain; charset=UTF-8\n' +
      'Content-Transfer-Encoding: 8bit\n' +
      f'Plural-Forms: {plural_forms}\n',
  )

  nplurals = int(re.search(r'nplurals=(\d+)', plural_forms).group(1))
  for e in entries:
    if e.is_plural:
      e.msgstr_plural = dict.fromkeys(range(nplurals), '')

  write_po(po_path, header, entries)


# ──── PO merge (replaces msgmerge) ────

def merge_po(po_path: str | Path, pot_path: str | Path) -> None:
  """Update a .po file with entries from a .pot template (replaces msgmerge --update)."""
  po_header, po_entries = parse_po(po_path)
  _, pot_entries = parse_po(pot_path)

  existing = {e.msgid: e for e in po_entries}
  merged = []

  for pot_e in pot_entries:
    if pot_e.msgid in existing:
      old = existing[pot_e.msgid]
      old.source_refs = pot_e.source_refs
      old.flags = pot_e.flags
      old.comments = pot_e.comments
      if pot_e.is_plural:
        old.msgid_plural = pot_e.msgid_plural
      merged.append(old)
    else:
      merged.append(pot_e)

  merged.sort(key=lambda e: e.msgid)
  write_po(po_path, po_header, merged)
