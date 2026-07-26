import argparse
import functools
import html
import http.server
import json
import posixpath
import re
import shutil
import threading
import time
import urllib.parse
from pathlib import Path
from xml.etree import ElementTree as ET

import markdown
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor

DOCS_DIR = Path(__file__).resolve().parent
SITE_DIR = DOCS_DIR / "_site"
TEMPLATE_FILE = DOCS_DIR / "template.html"

# Pages whose source lives under docs/ but should not be emitted as pages.
EXCLUDE_DIRS = {"_site"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_URL = "https://github.com/commaai/openpilot/"

# (title, target) pairs. target is a page path or an absolute URL.
# A None target marks a section header.
NAV: list[tuple[str, str | None]] = [
  ("What is openpilot?", "index.md"),
  ("How-to", None),
  ("Turn the speed blue", "how-to/turn-the-speed-blue.md"),
  ("Connect to a comma 3X or four", "how-to/connect-to-comma.md"),
  ("Add support for a car", "how-to/car-port.md"),
  ("Concepts", None),
  ("Logs", "concepts/logs.md"),
  ("Safety", "concepts/safety.md"),
  ("Glossary", "concepts/glossary.md"),
  ("Contributing", None),
  ("Feedback", "contributing/feedback.md"),
  ("Roadmap", "contributing/roadmap.md"),
  ("Contributing Guide →", "https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md"),
  ("Links", None),
  ("Blog →", "https://blog.comma.ai"),
  ("Bounties →", "https://comma.ai/bounties"),
  ("GitHub →", "https://github.com/commaai"),
  ("Discord →", "https://discord.comma.ai"),
  ("X →", "https://x.com/comma_ai"),
]

GlossaryTerm = tuple[str, re.Pattern[str], str]

GLOSSARY_DESCRIPTIONS = {
  "onroad": "openpilot's system state while ignition is on.",
  "offroad": "openpilot's system state while ignition is off.",
  "route": "A route is a recording of an onroad session.",
  "segment": "Routes are split into one minute chunks called segments.",
  "comma connect": "The web viewer for all your routes; check it out at [connect.comma.ai](https://connect.comma.ai).",
  "panda": "The secondary processor on the device that implements the functional safety and directly talks to the car over CAN. See the [panda repo](https://github.com/commaai/panda).",
  "comma four": "The latest hardware by comma.ai for running openpilot. More info at [comma.ai/shop/comma-four](https://www.comma.ai/shop/comma-four).",
}
GLOSSARY_PAGE = "concepts/glossary.md"
GLOSSARY_ROUTE = GLOSSARY_PAGE.removesuffix(".md")
GLOSSARY_SKIP_TAGS = {"a", "code", "h1", "h2", "h3", "h4", "h5", "h6", "kbd", "pre", "script", "style"}

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def page_route(path: str) -> str:
  path = path.removesuffix(".md")
  return posixpath.dirname(path) or "." if posixpath.basename(path) == "index" else path


def page_href(current_page: str, target_page: str) -> str:
  route = posixpath.relpath(page_route(target_page), page_route(current_page))
  return ("." if route == "." else route) + "/"


def rewrite_relative_url(value: str, page: str) -> str | None:
  url = urllib.parse.urlparse(value)
  if value.startswith(("#", "/")) or url.scheme or url.netloc or not url.path:
    return None
  target = posixpath.normpath(posixpath.join(posixpath.dirname(page), url.path))
  if target == ".." or target.startswith("../"):
    return None
  path = page_href(page, target) if target.endswith(".md") else posixpath.relpath(target, page_route(page))
  return url._replace(path=path).geturl()


class RelLinksTreeprocessor(Treeprocessor):
  """Rebase relative links for directory-style page URLs."""

  name = "rellinks"

  def __init__(self, md, path: str):
    super().__init__(md)
    self.path = path

  def run(self, root: ET.Element) -> None:
    for el in root.iter():
      for key in ("href", "src"):
        value = el.get(key)
        if not value:
          continue
        rewritten = self._rewrite(value)
        if rewritten is not None:
          el.set(key, rewritten)

  def _rewrite(self, value: str) -> str | None:
    return rewrite_relative_url(value, self.path)


class RawHtmlLinksPreprocessor(Preprocessor):
  pattern = re.compile(r"""(?P<prefix>\b(?:href|src)=(?P<quote>["']))(?P<url>.*?)(?P=quote)""")

  def __init__(self, md, path: str):
    super().__init__(md)
    self.path = path

  def run(self, lines: list[str]) -> list[str]:
    def replace(match: re.Match[str]) -> str:
      value = match.group("url")
      rewritten = rewrite_relative_url(value, self.path)
      return match.group(0) if rewritten is None else f'{match.group("prefix")}{rewritten}{match.group("quote")}'

    return [self.pattern.sub(replace, line) for line in lines]


def clean_tooltip(description: str) -> str:
  text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", description)
  text = re.sub(r"`([^`]+)`", r"\1", text)
  text = re.sub(r"[*_~]", "", text)
  return re.sub(r"\s+", " ", text).strip()


def glossary_slug(label: str) -> str:
  return label.replace(" ", "-").replace("_", "-").lower()


GLOSSARY_TERMS = [
  (glossary_slug(label), re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE), clean_tooltip(description))
  for label, description in GLOSSARY_DESCRIPTIONS.items()
]
GLOSSARY_DEFINITIONS = "\n".join(
  f'* <span id="{glossary_slug(label)}"></span>**{label}**: {description}'
  for label, description in GLOSSARY_DESCRIPTIONS.items()
)


class GlossaryTreeprocessor(Treeprocessor):
  def __init__(self, md, glossary: list[GlossaryTerm], path: str):
    super().__init__(md)
    self.glossary = glossary
    self.path = path
    self.seen: set[str] = set()

  def run(self, root: ET.Element) -> None:
    if self.path == GLOSSARY_PAGE:
      return

    self.seen.clear()
    current_route = "." if self.path == "index.md" else self.path.removesuffix(".md")
    glossary_href = f"{posixpath.relpath(GLOSSARY_ROUTE, current_route)}/#"
    self._walk(root, glossary_href)

  def _walk(self, element: ET.Element, glossary_href: str) -> None:
    if element.tag in GLOSSARY_SKIP_TAGS or element.attrib.get("data-glossary-skip") is not None:
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
      insert_at = -1
    else:
      assert index is not None
      child.tail = pieces[0] if isinstance(pieces[0], str) else ""
      insert_at = index

    start = 1 if isinstance(pieces[0], str) else 0
    previous = child

    for piece in pieces[start:]:
      if isinstance(piece, str):
        assert previous is not None
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
      for order, (slug, pattern, tooltip) in enumerate(self.glossary):
        if slug in self.seen:
          continue

        found = pattern.search(text, cursor)
        if found is None:
          continue

        candidate = (found.start(), found.start() - found.end(), order, slug, tooltip, found.end())
        if best is None or candidate[:3] < best[:3]:
          best = candidate

      if best is None:
        break

      start, _, _, slug, tooltip, end = best
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


def render_markdown(text: str, page_path: str) -> str:
  md = markdown.Markdown(
    extensions=["tables", "toc", "attr_list", "admonition", "fenced_code", "md_in_html"],
    extension_configs={"toc": {"permalink": "#"}},
    output_format="html5",
  )
  md.preprocessors.register(RawHtmlLinksPreprocessor(md, page_path), "raw-html-links", 21)
  md.treeprocessors.register(RelLinksTreeprocessor(md, page_path), "relative-links", 1)
  md.treeprocessors.register(GlossaryTreeprocessor(md, GLOSSARY_TERMS, page_path), "glossary-links", 0)
  return md.convert(text)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_title(source: str) -> str:
  for line in source.splitlines():
    if line.startswith("# "):
      return line[2:].strip()
  return "openpilot docs"


def write_html_redirect(rel: Path) -> None:
  if rel.name == "index.md":
    return
  target = f"{rel.stem}/"
  out = SITE_DIR / rel.with_suffix(".html")
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text("\n".join([
    "<!doctype html>",
    f'<meta http-equiv="refresh" content="0; url={html.escape(target)}">',
    f'<link rel="canonical" href="{html.escape(target)}">',
    f"<script>location.replace({json.dumps(target)} + location.search + location.hash)</script>",
  ]))


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

def copy_assets() -> None:
  for src in DOCS_DIR.rglob("*"):
    if not src.is_file():
      continue
    rel = src.relative_to(DOCS_DIR)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
      continue
    if src.suffix == ".md" or src in (Path(__file__).resolve(), TEMPLATE_FILE):
      continue
    dest = SITE_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def render_nav_html(current_page: str) -> str:
  parts: list[str] = []
  for title, target in NAV:
    if target is None:
      parts.append(f'<div class="nav-section">{html.escape(title)}</div>')
    elif target.startswith(("http://", "https://")):
      parts.append(f'<a href="{html.escape(target)}">{html.escape(title)}</a>')
    else:
      active = ' class="active"' if target == current_page else ""
      parts.append(f'<a href="{html.escape(page_href(current_page, target))}"{active}>{html.escape(title)}</a>')
  return "\n".join(parts)


def build() -> None:
  template = TEMPLATE_FILE.read_text()
  pages = [
    (path.relative_to(DOCS_DIR), path.read_text()) for path in sorted(DOCS_DIR.rglob("*.md"))
    if path != DOCS_DIR / "README.md"
    and not any(part in EXCLUDE_DIRS for part in path.relative_to(DOCS_DIR).parts)
  ]
  pages.append((Path(GLOSSARY_PAGE), f"# openpilot glossary\n\n{GLOSSARY_DEFINITIONS}"))
  pages.sort()

  if SITE_DIR.exists():
    shutil.rmtree(SITE_DIR)
  SITE_DIR.mkdir(parents=True)

  copy_assets()

  for rel_path, source in pages:
    rel = rel_path.as_posix()
    body = render_markdown(source, rel)
    title = page_title(source)
    route = page_route(rel)
    root = "../" * (0 if route == "." else len(route.split("/")))
    edit_path = "serve.py" if rel == GLOSSARY_PAGE else rel
    edit_url = f"{REPO_URL}blob/master/docs/{edit_path}"
    page_html = template
    for name, value in {
      "TITLE": html.escape(title),
      "ROOT": root,
      "HOME_HREF": page_href(rel, "index.md"),
      "NAV": render_nav_html(rel),
      "BODY": body,
      "EDIT_URL": html.escape(edit_url),
    }.items():
      page_html = page_html.replace(f"{{{{{name}}}}}", value)
    out = SITE_DIR / ("" if route == "." else route) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html)
    write_html_redirect(rel_path)

  print(f"docs: built {len(pages)} pages into {SITE_DIR}")


# ---------------------------------------------------------------------------
# Serve
# ---------------------------------------------------------------------------

def serve() -> None:
  build()
  mtimes = {p: p.stat().st_mtime for p in DOCS_DIR.rglob("*") if p.is_file()}
  handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE_DIR))
  httpd = http.server.ThreadingHTTPServer(("", 0), handler)
  print(f"docs: serving on http://localhost:{httpd.server_port}/ (watching for changes)")
  try:
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    while True:
      time.sleep(0.5)
      new_mtimes = {p: p.stat().st_mtime for p in DOCS_DIR.rglob("*") if p.is_file()}
      if new_mtimes != mtimes:
        mtimes = new_mtimes
        print("docs: change detected, rebuilding...")
        try:
          build()
        except Exception as e:
          print(f"docs: build failed: {e}")
  except KeyboardInterrupt:
    pass
  finally:
    httpd.shutdown()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Build or serve the openpilot documentation site.")
  parser.add_argument("--build", action="store_true", help="Build the site and exit.")
  args = parser.parse_args()

  if args.build:
    build()
  else:
    serve()
