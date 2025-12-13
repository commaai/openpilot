#!/usr/bin/env python3
import json
import os
import requests
import xml.etree.ElementTree as ET

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.ui.update_translations import LANGUAGES_FILE, TRANSLATIONS_DIR

BADGE_HEIGHT = 20 + 8
SHIELDS_URL = "https://img.shields.io/badge"

def parse_po_file(file_path):
  """
  Parse a .po file and count total and unfinished translations.
  Returns: (total_translations, unfinished_translations)
  """
  with open(file_path) as f:
    content = f.read()

  total_translations = 0
  unfinished_translations = 0

  # Split into entries (separated by blank lines)
  entries = content.split('\n\n')

  for entry in entries:
    # Skip header entry (contains Project-Id-Version)
    if 'Project-Id-Version' in entry:
      continue

    # Check if this entry has a msgid (translation entry)
    # After skipping header, any entry with msgid " is a translation
    # (both msgid "content" and msgid "" for multiline contain msgid ")
    if 'msgid "' not in entry:
      continue

    total_translations += 1

    # Check if msgstr is empty (unfinished translation)
    if 'msgstr ""' in entry:
      # Check if there are continuation lines with content after msgstr ""
      lines = entry.split('\n')
      msgstr_idx = None
      for i, line in enumerate(lines):
        if line.strip().startswith('msgstr ""'):
          msgstr_idx = i
          break

      if msgstr_idx is not None:
        # Check if any continuation lines have content
        has_content = False
        for line in lines[msgstr_idx + 1:]:
          stripped = line.strip()
          # Continuation line with content
          if stripped.startswith('"') and len(stripped) > 2:
            has_content = True
            break
          # End of entry
          if stripped.startswith(('msgid', '#')) or not stripped:
            break

        if not has_content:
          unfinished_translations += 1

  return (total_translations, unfinished_translations)

if __name__ == "__main__":
  with open(LANGUAGES_FILE) as f:
    translation_files = json.load(f)

  badge_svg = []
  max_badge_width = 0  # keep track of max width to set parent element
  for idx, (name, file) in enumerate(translation_files.items()):
    po_file_path = os.path.join(str(TRANSLATIONS_DIR), f"app_{file}.po")

    total_translations, unfinished_translations = parse_po_file(po_file_path)

    percent_finished = int(100 - (unfinished_translations / total_translations * 100.)) if total_translations > 0 else 0
    color = f"rgb{(94, 188, 0) if percent_finished == 100 else (248, 255, 50) if percent_finished > 90 else (204, 55, 27)}"

    # Download badge
    badge_label = f"LANGUAGE {name}"
    badge_message = f"{percent_finished}% complete"
    if unfinished_translations != 0:
      badge_message += f" ({unfinished_translations} unfinished)"

    r = requests.get(f"{SHIELDS_URL}/{badge_label}-{badge_message}-{color}", timeout=10)
    assert r.status_code == 200, "Error downloading badge"
    content_svg = r.content.decode("utf-8")

    xml = ET.fromstring(content_svg)
    assert "width" in xml.attrib
    max_badge_width = max(max_badge_width, int(xml.attrib["width"]))

    # Make tag ids in each badge unique to combine them into one svg
    for tag in ("r", "s"):
      content_svg = content_svg.replace(f'id="{tag}"', f'id="{tag}{idx}"')
      content_svg = content_svg.replace(f'"url(#{tag})"', f'"url(#{tag}{idx})"')

    badge_svg.extend([f'<g transform="translate(0, {idx * BADGE_HEIGHT})">', content_svg, "</g>"])

  badge_svg.insert(0, '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ' +
                   f'height="{len(translation_files) * BADGE_HEIGHT}" width="{max_badge_width}">')
  badge_svg.append("</svg>")

  with open(os.path.join(BASEDIR, "translation_badge.svg"), "w") as badge_f:
    badge_f.write("\n".join(badge_svg))
