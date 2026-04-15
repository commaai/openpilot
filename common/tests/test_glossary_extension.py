from __future__ import annotations

from pathlib import Path
import sys
from textwrap import dedent

import pytest
from markdown import Markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs"))

from ext.glossary import GlossaryExtension

LinksExtension = pytest.importorskip("zensical.extensions.links").LinksExtension


@pytest.fixture
def glossary_file(tmp_path: Path) -> Path:
  file_path = tmp_path / "glossary.toml"
  file_path.write_text(
    dedent(
      """
      [glossary.onroad]
      title = "Onroad"
      category = "System States"
      description = "openpilot's system state while ignition is on."

      [glossary.route]
      title = "Route"
      category = "Logs"
      description = "A route is a recording of an onroad session."

      [glossary.segment]
      title = "Segment"
      category = "Logs"
      description = "Routes are split into one minute chunks called segments."
      """
    ).strip()
    + "\n"
  )
  return file_path


def render(markdown: str, glossary_file: Path, path: str) -> str:
  md = Markdown(
    extensions=["attr_list", GlossaryExtension(glossary_file=str(glossary_file), glossary_page="concepts/glossary.md")],
  )
  LinksExtension(path=path, use_directory_urls=True).extendMarkdown(md)
  return md.convert(markdown)


def test_glossary_page_is_generated_from_structured_data(glossary_file: Path) -> None:
  html = render("{{GLOSSARY_DEFINITIONS}}", glossary_file, "concepts/glossary.md")

  assert 'id="onroad"' in html
  assert 'id="segment"' in html
  assert "System States" not in html
  assert html.count("<li>") == 3
  assert "{ #onroad }" not in html
  assert "glossary-term" not in html


def test_first_occurrence_of_each_term_is_annotated(glossary_file: Path) -> None:
  html = render(
    "Onroad becomes a route. Later, the onroad route enters the next segment.",
    glossary_file,
    "concepts/logs.md",
  )

  assert html.count('href="../glossary/#onroad"') == 1
  assert html.count('href="../glossary/#route"') == 1
  assert html.count('href="../glossary/#segment"') == 1
  assert 'data-search-exclude=""' in html
