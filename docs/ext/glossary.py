import posixpath
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor

from zensical.extensions.links import LinksProcessor

GlossaryTerm = dict[str, Any]

SKIP_TAGS = {
  "a",
  "code",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "kbd",
  "pre",
  "script",
  "style",
}

def clean_tooltip(description: str) -> str:
  text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", description)
  text = re.sub(r"`([^`]+)`", r"\1", text)
  text = re.sub(r"[*_~]", "", text)
  return re.sub(r"\s+", " ", text).strip()


def load_glossary(file_path: str) -> list[GlossaryTerm]:
  with open(file_path, "rb") as f:
    glossary_data = tomllib.load(f).get("glossary", {})

  glossary: list[GlossaryTerm] = []
  for key, value in glossary_data.items():
    label = str(key).strip().replace("_", " ")
    description = str(value).strip()
    if not description:
      continue

    glossary.append(
      {
        "label": label,
        "slug": label.replace(" ", "-").replace("_", "-").lower(),
        "description": description,
        "tooltip": clean_tooltip(description),
        "pattern": re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE),
      }
    )

  return glossary
class GlossaryPreprocessor(Preprocessor):
  def __init__(self, md, glossary: list[GlossaryTerm], placeholder: str):
    super().__init__(md)
    self.glossary = glossary
    self.placeholder = placeholder

  def run(self, lines: list[str]) -> list[str]:
    markdown = "\n".join(lines)
    if self.placeholder not in markdown:
      return lines
    glossary = "\n".join(
      f'* <span id="{term["slug"]}"></span>**{term["label"]}**: {term["description"]}'
      for term in self.glossary
    )
    return markdown.replace(self.placeholder, glossary).splitlines()


class GlossaryTreeprocessor(Treeprocessor):
  def __init__(self, md, glossary: list[GlossaryTerm], glossary_page: str, match_policy: str):
    super().__init__(md)
    self.glossary = glossary
    self.glossary_page = glossary_page
    self.first_only = match_policy == "first"
    self.seen: set[str] = set()

  def run(self, root: ET.Element) -> None:
    at = self.md.treeprocessors.get_index_for_name("zrelpath")
    processor = self.md.treeprocessors[at]
    if not isinstance(processor, LinksProcessor):
      raise TypeError("Links processor not registered")
    if processor.path == self.glossary_page:
      return

    self.seen.clear()
    self._walk(root, processor.path)

  def _walk(self, element: ET.Element, page_path: str) -> None:
    if element.tag in SKIP_TAGS or element.attrib.get("data-glossary-skip") is not None:
      return

    self._replace(element, page_path)

    idx = 0
    while idx < len(element):
      child = element[idx]
      self._walk(child, page_path)
      idx = self._replace(element, page_path, idx) + 1

  def _replace(self, parent: ET.Element, page_path: str, index: int | None = None) -> int:
    child = None if index is None else parent[index]
    text = parent.text if child is None else child.tail
    pieces = self._pieces(text or "", page_path)
    if not pieces:
      return -1 if index is None else index

    if child is None:
      parent.text = pieces[0] if isinstance(pieces[0], str) else ""
      insert_at = 0 if isinstance(pieces[0], str) else -1
    else:
      assert index is not None
      child.tail = pieces[0] if isinstance(pieces[0], str) else ""
      insert_at = index

    start = 1 if isinstance(pieces[0], str) else 0
    previous = child

    for piece in pieces[start:]:
      if isinstance(piece, str):
        previous.tail = (previous.tail or "") + piece
        continue

      insert_at += 1
      parent.insert(insert_at, piece)
      previous = piece

    return insert_at

  def _pieces(self, text: str, page_path: str) -> list[str | ET.Element]:
    if not text.strip():
      return []

    pieces: list[str | ET.Element] = []
    cursor = 0

    while match := self._next_match(text, cursor):
      term, start, end = match
      if start > cursor:
        pieces.append(text[cursor:start])

      pieces.append(self._anchor(page_path, term, text[start:end]))
      if self.first_only:
        self.seen.add(term["slug"])
      cursor = end

    if not pieces:
      return []
    if cursor < len(text):
      pieces.append(text[cursor:])
    return pieces

  def _next_match(self, text: str, start: int) -> tuple[GlossaryTerm, int, int] | None:
    best: tuple[GlossaryTerm, int, int] | None = None
    for term in self.glossary:
      if self.first_only and term["slug"] in self.seen:
        continue

      found = term["pattern"].search(text, start)
      if found is None:
        continue

      candidate = (term, found.start(), found.end())
      if best is None:
        best = candidate
        continue

      _, best_start, best_end = best
      _, current_start, current_end = candidate
      if current_start < best_start:
        best = candidate
        continue

      if current_start == best_start and current_end - current_start > best_end - best_start:
        best = candidate

    return best

  def _anchor(self, page_path: str, term: GlossaryTerm, label: str) -> ET.Element:
    link = ET.Element(
      "a",
      {
        "class": "glossary-term",
        "data-glossary-term": "",
        "href": f"{posixpath.relpath(self.glossary_page, posixpath.dirname(page_path) or '.')}#{term['slug']}",
      },
    )
    ET.SubElement(link, "span", {"class": "glossary-term__label"}).text = label
    ET.SubElement(
      link,
      "span",
      {
        "class": "glossary-term__tooltip",
        "data-search-exclude": "",
      },
    ).text = term["tooltip"]
    return link

class GlossaryExtension(Extension):
  def __init__(self, **kwargs: Any):
    self.config = {
      "glossary_file": ["docs/ext/glossary.toml", "Path to the glossary data file"],
      "glossary_page": ["concepts/glossary.md", "Glossary page path relative to docs/"],
      "placeholder": ["{{GLOSSARY_DEFINITIONS}}", "Marker to replace in the glossary page"],
      "match_policy": ["first", "Only annotate the first occurrence of each term on a page"],
    }
    super().__init__(**kwargs)

  def extendMarkdown(self, md) -> None:
    md.registerExtension(self)
    glossary = load_glossary(str(Path(self.getConfig("glossary_file")).resolve()))

    md.preprocessors.register(
      GlossaryPreprocessor(md, glossary, self.getConfig("placeholder")),
      "docs-ext-glossary-preprocessor",
      27,
    )
    md.treeprocessors.register(
      GlossaryTreeprocessor(md, glossary, self.getConfig("glossary_page"), self.getConfig("match_policy")),
      "docs-ext-glossary-treeprocessor",
      0,
    )


def makeExtension(**kwargs: Any) -> GlossaryExtension:
  return GlossaryExtension(**kwargs)
