#!/usr/bin/env python3
import json
import os
import requests
import xml.etree.ElementTree as ET

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.ui.tests.test_translations import UNFINISHED_TRANSLATION_TAG
from openpilot.selfdrive.ui.update_translations import LANGUAGES_FILE, TRANSLATIONS_DIR

TRANSLATION_TAG = "<translation"
BADGE_HEIGHT = 20 + 8
SHIELDS_URL = "https://img.shields.io/badge"

if __name__ == "__main__":
  with open(LANGUAGES_FILE) as f:
    translation_files = json.load(f)

  badge_svg = []
  max_badge_width = 0  # keep track of max width to set parent element
  for idx, (name, file) in enumerate(translation_files.items()):
    with open(os.path.join(TRANSLATIONS_DIR, f"{file}.ts")) as tr_f:
      tr_file = tr_f.read()

    total_translations = 0
    unfinished_translations = 0
    for line in tr_file.splitlines():
      if TRANSLATION_TAG in line:
        total_translations += 1
      if UNFINISHED_TRANSLATION_TAG in line:
        unfinished_translations += 1

    percent_finished = int(100 - (unfinished_translations / total_translations * 100.))
    color = "green" if percent_finished == 100 else "orange" if percent_finished > 90 else "red"

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
