import json
import re
import string
from pathlib import Path

import pytest

from openpilot.selfdrive.ui.translations.potools import parse_po
from openpilot.system.ui.lib.multilang import LANGUAGES_FILE, TRANSLATIONS_DIR

PERCENT_PLACEHOLDER_RE = re.compile(r"%(?:n|\d+)")
BAD_ENTITY_RE = re.compile(r'@(\w+);')
LINE_NUMBER_REF_RE = re.compile(r'^#:\s+.+:\d+(?:\s|$)')
FORMATTER = string.Formatter()
PO_DIR = Path(str(TRANSLATIONS_DIR))

with LANGUAGES_FILE.open(encoding='utf-8') as f:
  TRANSLATION_LANGUAGES = json.load(f)


def extract_placeholders(text: str) -> list[str]:
  placeholders = PERCENT_PLACEHOLDER_RE.findall(text)

  try:
    parsed = list(FORMATTER.parse(text))
  except ValueError as e:
    raise AssertionError(f"invalid brace formatting in {text!r}: {e}") from e

  for _, field_name, format_spec, conversion in parsed:
    if field_name is None:
      continue

    token = "{"
    token += field_name
    if conversion:
      token += f"!{conversion}"
    if format_spec:
      token += f":{format_spec}"
    token += "}"
    placeholders.append(token)

  return sorted(placeholders)


def load_po_text(po_path: Path) -> str:
  return po_path.read_text(encoding='utf-8')


@pytest.mark.parametrize("language_code", sorted(TRANSLATION_LANGUAGES.values()))
def test_translation_file_exists(language_code: str):
  po_path = PO_DIR / f"app_{language_code}.po"
  assert po_path.exists(), f"missing translation file: {po_path}"


@pytest.mark.parametrize("po_path", sorted(PO_DIR.glob("app_*.po")), ids=lambda p: p.name)
def test_translation_placeholders_are_preserved(po_path: Path):
  _, entries = parse_po(po_path)
  language = po_path.stem.removeprefix("app_")

  for entry in entries:
    source_placeholders = extract_placeholders(entry.msgid)

    if entry.is_plural:
      plural_placeholders = extract_placeholders(entry.msgid_plural)
      message = (
        f"{language}: source plural placeholders do not match singular for "
        + f"{entry.msgid!r}: {source_placeholders} vs {plural_placeholders}"
      )
      assert plural_placeholders == source_placeholders, message

      for idx, msgstr in sorted(entry.msgstr_plural.items()):
        if not msgstr:
          continue

        translated_placeholders = extract_placeholders(msgstr)
        message = (
          f"{language}: plural form {idx} changes placeholders for {entry.msgid!r}: "
          + f"expected {source_placeholders}, got {translated_placeholders}"
        )
        assert translated_placeholders == source_placeholders, message
    else:
      if not entry.msgstr:
        continue

      translated_placeholders = extract_placeholders(entry.msgstr)
      message = (
        f"{language}: translation changes placeholders for {entry.msgid!r}: "
        + f"expected {source_placeholders}, got {translated_placeholders}"
      )
      assert translated_placeholders == source_placeholders, message


@pytest.mark.parametrize("po_path", sorted(PO_DIR.glob("app_*.po")), ids=lambda p: p.name)
def test_translation_refs_do_not_include_line_numbers(po_path: Path):
  for line in load_po_text(po_path).splitlines():
    assert not LINE_NUMBER_REF_RE.match(line), (
      f"{po_path.name}: line-number source reference found: {line}"
    )


@pytest.mark.parametrize("po_path", sorted(PO_DIR.glob("app_*.po")), ids=lambda p: p.name)
def test_translation_entities_are_valid(po_path: Path):
  matches = BAD_ENTITY_RE.findall(load_po_text(po_path))
  assert not matches, (
    f"{po_path.name}: found '@...;' entity typo(s): {', '.join(sorted(set(matches)))}"
  )
