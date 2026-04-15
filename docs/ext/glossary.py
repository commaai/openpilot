from __future__ import annotations

import posixpath
import re
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor

from zensical.extensions.links import LinksProcessor

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


@dataclass(frozen=True)
class GlossaryTerm:
  slug: str
  title: str
  description: str
  category: str
  aliases: tuple[str, ...]

  @property
  def tooltip(self) -> str:
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", self.description)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class GlossaryVariant:
  slug: str
  pattern: re.Pattern[str]


@dataclass(frozen=True)
class MatchResult:
  slug: str
  start: int
  end: int

  @property
  def length(self) -> int:
    return self.end - self.start


class Glossary:
  def __init__(self, terms: list[GlossaryTerm]):
    self.terms = terms
    self.by_slug = {term.slug: term for term in terms}
    self.variants = self._build_variants(terms)

  @classmethod
  def load(cls, file_path: str) -> Glossary:
    with open(file_path, "rb") as f:
      data = tomllib.load(f)

    terms: list[GlossaryTerm] = []
    for key, values in data.get("glossary", {}).items():
      title = str(values.get("title", key.replace("_", " ").title())).strip()
      description = str(values.get("description", "")).strip()
      if not description:
        continue

      aliases = tuple(
        alias.strip()
        for alias in values.get("aliases", [])
        if isinstance(alias, str) and alias.strip()
      )
      category = str(values.get("category", "General")).strip() or "General"
      terms.append(
        GlossaryTerm(
          slug=str(values.get("slug", key.replace("_", "-").lower())),
          title=title,
          description=description,
          category=category,
          aliases=aliases,
        )
      )

    return cls(terms)

  def grouped(self) -> list[tuple[str, list[GlossaryTerm]]]:
    groups: dict[str, list[GlossaryTerm]] = {}
    for term in self.terms:
      groups.setdefault(term.category, []).append(term)
    return list(groups.items())

  @staticmethod
  def _build_variants(terms: list[GlossaryTerm]) -> list[GlossaryVariant]:
    variants: list[GlossaryVariant] = []
    for term in terms:
      names = [term.title, *term.aliases]
      names = sorted(set(names), key=len, reverse=True)
      for name in names:
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        variants.append(GlossaryVariant(term.slug, pattern))
    return variants


class GlossaryPreprocessor(Preprocessor):
  def __init__(self, md, glossary: Glossary, placeholder: str):
    super().__init__(md)
    self.glossary = glossary
    self.placeholder = placeholder

  def run(self, lines: list[str]) -> list[str]:
    markdown = "\n".join(lines)
    if self.placeholder not in markdown:
      return lines
    return markdown.replace(self.placeholder, self._render_glossary()).splitlines()

  def _render_glossary(self) -> str:
    lines = [
      f'* <span id="{term.slug}"></span>**{term.title}**: {term.description}'
      for term in self.glossary.terms
    ]
    return "\n".join(lines)


class GlossaryTreeprocessor(Treeprocessor):
  def __init__(self, md, glossary: Glossary, glossary_page: str, match_policy: str):
    super().__init__(md)
    self.glossary = glossary
    self.glossary_page = glossary_page
    self.match_policy = match_policy
    self.seen: set[str] = set()

  def run(self, root: ET.Element) -> None:
    processor = self._links_processor()
    if processor.path == self.glossary_page:
      return

    self.seen.clear()
    self._walk(root, processor.path)

  def _links_processor(self) -> LinksProcessor:
    at = self.md.treeprocessors.get_index_for_name("zrelpath")
    processor = self.md.treeprocessors[at]
    if not isinstance(processor, LinksProcessor):
      raise TypeError("Links processor not registered")
    return processor

  def _walk(self, element: ET.Element, page_path: str) -> None:
    if element.tag in SKIP_TAGS or element.attrib.get("data-glossary-skip") is not None:
      return

    self._replace_text(element, page_path)

    idx = 0
    while idx < len(element):
      child = element[idx]
      self._walk(child, page_path)
      idx = self._replace_tail(element, idx, page_path) + 1

  def _replace_text(self, element: ET.Element, page_path: str) -> None:
    pieces = self._tokenize(element.text or "", page_path)
    if not pieces:
      return

    element.text = pieces[0] if isinstance(pieces[0], str) else ""
    insert_at = 0 if isinstance(pieces[0], str) else -1
    start = 1 if isinstance(pieces[0], str) else 0

    previous: ET.Element | None = None
    for piece in pieces[start:]:
      if isinstance(piece, str):
        if previous is None:
          element.text = (element.text or "") + piece
        else:
          previous.tail = (previous.tail or "") + piece
        continue

      insert_at += 1
      element.insert(insert_at, piece)
      previous = piece

  def _replace_tail(self, parent: ET.Element, index: int, page_path: str) -> int:
    child = parent[index]
    pieces = self._tokenize(child.tail or "", page_path)
    if not pieces:
      return index

    child.tail = pieces[0] if isinstance(pieces[0], str) else ""
    insert_at = index
    previous = child
    start = 1 if isinstance(pieces[0], str) else 0

    for piece in pieces[start:]:
      if isinstance(piece, str):
        previous.tail = (previous.tail or "") + piece
        continue

      insert_at += 1
      parent.insert(insert_at, piece)
      previous = piece

    return insert_at

  def _tokenize(self, text: str, page_path: str) -> list[str | ET.Element]:
    if not text.strip():
      return []

    pieces: list[str | ET.Element] = []
    cursor = 0
    found_any = False

    while match := self._next_match(text, cursor):
      found_any = True
      if match.start > cursor:
        pieces.append(text[cursor:match.start])

      matched_text = text[match.start:match.end]
      pieces.append(self._build_anchor(page_path, self.glossary.by_slug[match.slug], matched_text))
      if self.match_policy == "first":
        self.seen.add(match.slug)
      cursor = match.end

    if not found_any:
      return []

    if cursor < len(text):
      pieces.append(text[cursor:])

    return pieces

  def _next_match(self, text: str, start: int) -> MatchResult | None:
    best: MatchResult | None = None
    for variant in self.glossary.variants:
      if self.match_policy == "first" and variant.slug in self.seen:
        continue

      found = variant.pattern.search(text, start)
      if found is None:
        continue

      candidate = MatchResult(variant.slug, found.start(), found.end())
      if best is None:
        best = candidate
        continue

      if candidate.start < best.start:
        best = candidate
        continue

      if candidate.start == best.start and candidate.length > best.length:
        best = candidate

    return best

  def _build_anchor(self, page_path: str, term: GlossaryTerm, label: str) -> ET.Element:
    href = f"{self._relative_glossary_path(page_path)}#{term.slug}"
    link = ET.Element(
      "a",
      {
        "class": "glossary-term",
        "data-glossary-term": "",
        "href": href,
      },
    )
    label_el = ET.SubElement(link, "span", {"class": "glossary-term__label"})
    label_el.text = label
    tooltip_el = ET.SubElement(
      link,
      "span",
      {
        "class": "glossary-term__tooltip",
        "data-search-exclude": "",
      },
    )
    tooltip_el.text = term.tooltip
    return link

  def _relative_glossary_path(self, page_path: str) -> str:
    current_dir = posixpath.dirname(page_path) or "."
    return posixpath.relpath(self.glossary_page, current_dir)


class GlossaryExtension(Extension):
  def __init__(self, **kwargs: Any):
    self.config = {
      "glossary_file": ["docs/ext/glossary.toml", "Path to the glossary data file"],
      "glossary_page": ["concepts/glossary.md", "Glossary page path relative to docs/"],
      "placeholder": ["{{GLOSSARY_DEFINITIONS}}", "Marker to replace in the glossary page"],
      "match_policy": ["first", "Only annotate the first occurrence of each term on a page"],
    }
    super().__init__(**kwargs)

  def extendMarkdown(self, md) -> None:  # noqa: N802
    md.registerExtension(self)
    glossary_file = str(Path(self.getConfig("glossary_file")).resolve())
    glossary = Glossary.load(glossary_file)

    md.preprocessors.register(
      GlossaryPreprocessor(md, glossary, self.getConfig("placeholder")),
      "docs-ext-glossary-preprocessor",
      27,
    )
    md.treeprocessors.register(
      GlossaryTreeprocessor(
        md,
        glossary,
        self.getConfig("glossary_page"),
        self.getConfig("match_policy"),
      ),
      "docs-ext-glossary-treeprocessor",
      0,
    )


def makeExtension(**kwargs: Any) -> GlossaryExtension:  # noqa: N802
  return GlossaryExtension(**kwargs)
