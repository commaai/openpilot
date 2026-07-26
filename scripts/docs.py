import argparse
import functools
import html
import http.server
import json
import posixpath
import re
import shutil
import string
import threading
import time
import tomllib
import urllib.parse
from pathlib import Path
from xml.etree import ElementTree as ET

import markdown
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SITE_DIR = REPO_ROOT / "docs_site"

# Pages whose source lives under docs/ but should not be emitted as pages.
EXCLUDE_DIRS = {"ext"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SITE_NAME = "openpilot docs"
REPO_URL = "https://github.com/commaai/openpilot/"
LOGO = "assets/comma-logo.png"

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

SOCIAL_HTML = """
<a href="https://github.com/commaai">github</a>
<a href="https://discord.comma.ai">discord</a>
<a href="https://x.com/comma_ai">x-twitter</a>
""".strip()

GlossaryTerm = tuple[str, re.Pattern[str], str]

GLOSSARY_FILE = DOCS_DIR / "ext" / "glossary.toml"
GLOSSARY_PAGE = "concepts/glossary.md"
GLOSSARY_ROUTE = GLOSSARY_PAGE.removesuffix(".md")
GLOSSARY_PLACEHOLDER = "{{GLOSSARY_DEFINITIONS}}"
GLOSSARY_SKIP_TAGS = {
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


@functools.cache
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
  glossary, definitions = load_glossary()
  md.treeprocessors.register(GlossaryTreeprocessor(md, glossary, page_path), "glossary-links", 0)
  return md.convert(text.replace(GLOSSARY_PLACEHOLDER, definitions))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_title(source: str) -> str:
  for line in source.splitlines():
    if line.startswith("# "):
      return line[2:].strip()
  return SITE_NAME


def write_html_redirect(src: Path) -> None:
  rel = src.relative_to(DOCS_DIR)
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
    if src.suffix == ".md":
      continue
    dest = SITE_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


COPY_JS = """
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('pre').forEach(function (el) {
    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'copy';
    btn.addEventListener('click', function () {
      var code = el.querySelector('code');
      if (!code) return;
      navigator.clipboard.writeText(code.innerText).then(function () {
        btn.textContent = 'copied';
        setTimeout(function () { btn.textContent = 'copy'; }, 1200);
      });
    });
    el.appendChild(btn);
  });
});
"""

TEMPLATE = string.Template("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$title · $site_name</title>
  <link rel="icon" href="${root}${logo}">
  <link rel="stylesheet" href="${root}stylesheets/extra.css">
</head>
<body>
  <header class="site">
    <a href="$home_href"><img src="${root}${logo}" alt="logo"></a>
    <span class="site-name">$site_name</span>
    <span class="spacer"></span>
    <a class="repo" href="$repo_url">GitHub</a>
  </header>

  <div class="layout">
    <nav class="sidebar">
      $nav_html
    </nav>

    <main class="content">
      $body
      <div class="edit-link">
        <a href="$edit_url">Edit this page on GitHub</a>
      </div>
    </main>
  </div>

  <footer class="site">
    <div class="social">
      $social_html
    </div>
    <div>$site_name</div>
  </footer>

  <script>$copy_js</script>
</body>
</html>
""")


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
  pages = [
    path for path in sorted(DOCS_DIR.rglob("*.md"))
    if not any(part in EXCLUDE_DIRS for part in path.relative_to(DOCS_DIR).parts)
  ]

  if SITE_DIR.exists():
    shutil.rmtree(SITE_DIR)
  SITE_DIR.mkdir(parents=True)

  copy_assets()

  for src in pages:
    rel = src.relative_to(DOCS_DIR).as_posix()
    source = src.read_text()
    body = render_markdown(source, rel)
    title = page_title(source)
    route = page_route(rel)
    root = "../" * (0 if route == "." else len(route.split("/")))
    edit_url = f"{REPO_URL}blob/master/docs/{rel}"
    page_html = TEMPLATE.substitute(
      title=html.escape(title),
      site_name=html.escape(SITE_NAME),
      root=root,
      home_href=page_href(rel, "index.md"),
      logo=LOGO,
      repo_url=html.escape(REPO_URL),
      social_html=SOCIAL_HTML,
      nav_html=render_nav_html(rel),
      body=body,
      edit_url=html.escape(edit_url),
      copy_js=COPY_JS,
    )
    out = SITE_DIR / ("" if route == "." else route) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html)
    write_html_redirect(src)

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
