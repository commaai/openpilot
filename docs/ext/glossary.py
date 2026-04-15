import posixpath
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor

from zensical.extensions.links import LinksProcessor

GlossaryTerm = tuple[str, re.Pattern[str], str]

GLOSSARY_FILE = Path(__file__).with_name("glossary.toml")
GLOSSARY_PAGE = "concepts/glossary.md"
GLOSSARY_PLACEHOLDER = "{{GLOSSARY_DEFINITIONS}}"

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


def load_glossary() -> tuple[list[GlossaryTerm], str]:
  with GLOSSARY_FILE.open("rb") as f:
    glossary_data = tomllib.load(f).get("glossary", {})

  glossary: list[GlossaryTerm] = []
  rendered = []
  for key, value in glossary_data.items():
    label = str(key).strip().replace("_", " ")
    description = str(value).strip()
    if not description:
      continue

    slug = label.replace(" ", "-").replace("_", "-").lower()
    glossary.append((slug, re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE), clean_tooltip(description)))
    rendered.append(f'* <span id="{slug}"></span>**{label}**: {description}')

  return glossary, "\n".join(rendered)


class GlossaryPreprocessor(Preprocessor):
  def __init__(self, md, glossary: str):
    super().__init__(md)
    self.glossary = glossary

  def run(self, lines: list[str]) -> list[str]:
    markdown = "\n".join(lines)
    if GLOSSARY_PLACEHOLDER not in markdown:
      return lines
    return markdown.replace(GLOSSARY_PLACEHOLDER, self.glossary).splitlines()


class GlossaryTreeprocessor(Treeprocessor):
  def __init__(self, md, glossary: list[GlossaryTerm]):
    super().__init__(md)
    self.glossary = glossary
    self.seen: set[str] = set()

  def run(self, root: ET.Element) -> None:
    at = self.md.treeprocessors.get_index_for_name("zrelpath")
    processor = self.md.treeprocessors[at]
    if not isinstance(processor, LinksProcessor):
      raise TypeError("Links processor not registered")
    if processor.path == GLOSSARY_PAGE:
      return

    self.seen.clear()
    glossary_href = f"{posixpath.relpath(GLOSSARY_PAGE, posixpath.dirname(processor.path) or '.')}#"
    self._walk(root, glossary_href)

  def _walk(self, element: ET.Element, glossary_href: str) -> None:
    if element.tag in SKIP_TAGS or element.attrib.get("data-glossary-skip") is not None:
      return

    self._replace(element, glossary_href)

    idx = 0
    while idx < len(element):
      child = element[idx]
      self._walk(child, glossary_href)
      idx = self._replace(element, glossary_href, idx) + 1

  def _replace(self, parent: ET.Element, glossary_href: str, index: int | None = None) -> int:
    child = None if index is None else parent[index]
    text = parent.text if child is None else child.tail
    pieces = self._pieces(text or "", glossary_href)
    if not pieces:
      return -1 if index is None else index

    if child is None:
      parent.text = pieces[0] if isinstance(pieces[0], str) else ""
      # Insert replacements for parent.text before the first existing child.
      insert_at = -1
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

  def _pieces(self, text: str, glossary_href: str) -> list[str | ET.Element]:
    if not text.strip():
      return []

    pieces: list[str | ET.Element] = []
    cursor = 0

    while True:
      best = None
      for slug, pattern, tooltip in self.glossary:
        if slug in self.seen:
          continue

        found = pattern.search(text, cursor)
        if found is None:
          continue

        candidate = (slug, tooltip, found.start(), found.end())
        if best is None:
          best = candidate
          continue

        _, _, best_start, best_end = best
        _, _, current_start, current_end = candidate
        if current_start < best_start:
          best = candidate
          continue

        if current_start == best_start and current_end - current_start > best_end - best_start:
          best = candidate

      if best is None:
        break

      slug, tooltip, start, end = best
      if start > cursor:
        pieces.append(text[cursor:start])

      link = ET.Element(
        "a",
        {
          "class": "glossary-term",
          "data-glossary-term": "",
          "href": f"{glossary_href}{slug}",
        },
      )
      ET.SubElement(link, "span", {"class": "glossary-term__label"}).text = text[start:end]
      ET.SubElement(
        link,
        "span",
        {
          "class": "glossary-term__tooltip",
          "data-search-exclude": "",
        },
      ).text = tooltip
      pieces.append(link)
      self.seen.add(slug)
      cursor = end

    if not pieces:
      return []
    if cursor < len(text):
      pieces.append(text[cursor:])
    return pieces


class GlossaryExtension(Extension):
  def extendMarkdown(self, md) -> None:
    md.registerExtension(self)
    glossary, rendered = load_glossary()

    md.preprocessors.register(
      GlossaryPreprocessor(md, rendered),
      "docs-ext-glossary-preprocessor",
      27,
    )
    md.treeprocessors.register(
      GlossaryTreeprocessor(md, glossary),
      "docs-ext-glossary-treeprocessor",
      0,
    )


def makeExtension(**kwargs) -> GlossaryExtension:
  return GlossaryExtension(**kwargs)
