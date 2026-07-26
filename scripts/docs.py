"""
openpilot docs builder.

A small static site generator for docs/ — Markdown -> HTML with a sidebar nav,
code blocks, and the glossary extension.

Usage:
  python scripts/docs.py build   # write the site to site_dir
  python scripts/docs.py serve   # rebuild on change and serve on a free port
"""
from __future__ import annotations

import functools
import html
import http.server
import posixpath
import re
import shutil
import signal
import string
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from xml.etree import ElementTree as ET

import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SITE_DIR = REPO_ROOT / "docs_site"

# Local docs build helpers live under docs/ so they stay near the content
# source. They are excluded from docs_site/ during the build.
sys.path.insert(0, str(DOCS_DIR))

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

SOCIAL = [
  ("github", "https://github.com/commaai"),
  ("discord", "https://discord.comma.ai"),
  ("x-twitter", "https://x.com/comma_ai"),
]


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


class RelLinksExtension(Extension):
  def __init__(self, path: str):
    self.path = path

  def extendMarkdown(self, md) -> None:
    md.registerExtension(self)
    md.treeprocessors.register(
      RelLinksTreeprocessor(md, self.path),
      RelLinksTreeprocessor.name,
      0,
    )
    md.preprocessors.register(
      RawHtmlLinksPreprocessor(md, self.path),
      "rellinks-raw-html",
      21,
    )


def make_extensions(page_path: str) -> list[str | Extension]:
  from ext.glossary import GlossaryExtension

  return [
    "tables",
    "toc",
    "attr_list",
    "admonition",
    "fenced_code",
    "md_in_html",
    RelLinksExtension(page_path),
    GlossaryExtension(page_path),
  ]


def render_markdown(text: str, page_path: str) -> str:
  md = markdown.Markdown(
    extensions=make_extensions(page_path),
    extension_configs={
      "toc": {"permalink": "#"},
    },
    output_format="html5",
  )
  return md.convert(text)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def source_pages() -> list[Path]:
  return [
    path for path in sorted(DOCS_DIR.rglob("*.md"))
    if not any(part in EXCLUDE_DIRS for part in path.relative_to(DOCS_DIR).parts)
  ]


def page_title(source: str) -> str:
  for line in source.splitlines():
    if line.startswith("# "):
      return line[2:].strip()
  return SITE_NAME


def output_html_path(src: Path) -> Path:
  route = page_route(src.relative_to(DOCS_DIR).as_posix())
  return SITE_DIR / ("" if route == "." else route) / "index.html"


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


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

BASE_CSS = """
:root {
  --bg: #fff;
  --bg-elev: #f5f5f5;
  --bg-hover: #eef0ff;
  --fg: #262626;
  --fg-dim: #666;
  --accent: #4051b5;
  --border: #e5e5e5;
  --max-width: 76rem;
}
* { box-sizing: border-box; }
html { scrollbar-gutter: stable; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  font-size: 15px;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

header.site {
  display: flex; align-items: center; gap: 0.75rem;
  min-height: 3rem;
  padding: 0.5rem max(1.25rem, calc((100% - var(--max-width)) / 2));
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 10;
}
header.site img { display: block; height: 24px; filter: invert(1); }
header.site .site-name { font-weight: 600; font-size: 1.05rem; }
header.site .spacer { flex: 1; }
header.site a.repo { color: var(--fg); font-size: 0.8rem; }

.layout { display: flex; width: 100%; max-width: var(--max-width); margin: 0 auto; flex: 1; }

nav.sidebar {
  width: 14rem; flex-shrink: 0;
  padding: 3.3rem 1.25rem 2rem 0;
  position: sticky; top: 3rem; align-self: start;
  max-height: calc(100vh - 3rem); overflow-y: auto;
}
nav.sidebar .nav-section { font-weight: 600; margin: 1.15rem 0.8rem 0.45rem; }
nav.sidebar .nav-section:first-child { margin-top: 0; }
nav.sidebar a {
  display: block; padding: 0.28rem 0.8rem; border-radius: 0.45rem;
  color: var(--fg); line-height: 1.35;
}
nav.sidebar a:hover { background: var(--bg-hover); text-decoration: none; }
nav.sidebar a.active { background: var(--bg-hover); color: var(--accent); font-weight: 600; }

main.content { width: min(100%, 46.5rem); min-width: 0; padding: 2.7rem 2rem 5rem; }
main.content h1, main.content h2, main.content h3 { line-height: 1.25; margin-top: 1.8rem; }
main.content h1 { margin-top: 0; }
main.content img { max-width: 100%; }
main.content table { border-collapse: collapse; display: block; overflow-x: auto; }
main.content th, main.content td { border: 1px solid var(--border); padding: 0.4rem 0.6rem; text-align: left; }
main.content th { background: var(--bg-elev); }
main.content pre { background: var(--bg-elev); padding: 0.9rem 1rem; border-radius: 0.25rem; overflow-x: auto; }
main.content code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
main.content :not(pre) > code { background: var(--bg-elev); padding: 0.1rem 0.35rem; border-radius: 0.2rem; font-size: 0.88em; }
main.content blockquote { border-left: 3px solid var(--border); margin: 1rem 0; padding: 0.2rem 1rem; color: var(--fg-dim); }
main.content details { margin: 0.5rem 0; }
main.content hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
.headerlink { margin-left: 0.25rem; opacity: 0; font-size: 0.7em; }
h1:hover .headerlink, h2:hover .headerlink, h3:hover .headerlink { opacity: 1; }

.edit-link { margin-top: 3rem; font-size: 0.85rem; color: var(--fg-dim); }

footer.site {
  display: flex; flex-direction: row-reverse; align-items: center; justify-content: space-between;
  min-height: 4rem; padding: 1rem max(1.25rem, calc((100% - var(--max-width)) / 2));
  background: var(--bg-elev); color: var(--fg-dim);
}
footer.site .social { display: flex; gap: 1rem; }

/* code copy buttons */
main.content pre { position: relative; }
.copy-btn {
  position: absolute; top: 0.4rem; right: 0.4rem;
  background: var(--bg); color: var(--fg-dim);
  border: 1px solid var(--border); border-radius: 0.35rem;
  padding: 0.2rem 0.5rem; font-size: 0.78rem; cursor: pointer;
  opacity: 0; transition: opacity 120ms;
}
main.content pre:hover .copy-btn { opacity: 1; }

@media (max-width: 48rem) {
  nav.sidebar { display: none; }
  main.content { width: 100%; padding: 2rem 1.25rem 4rem; }
}
"""

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
  <link rel="stylesheet" href="${root}stylesheets/base.css">
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
    else:
      parts.append(_nav_link(title, target, current_page))
  return "\n".join(parts)


def _nav_link(title: str, target: str, current_page: str) -> str:
  if target.startswith(("http://", "https://")):
    return f'<a href="{html.escape(target)}">{html.escape(title)}</a>'
  active = ' class="active"' if target == current_page else ""
  return f'<a href="{html.escape(page_href(current_page, target))}"{active}>{html.escape(title)}</a>'


def social_html() -> str:
  return "\n".join(
    f'<a href="{html.escape(link)}">{html.escape(icon)}</a>'
    for icon, link in SOCIAL
  )


def build() -> None:
  pages = source_pages()

  if SITE_DIR.exists():
    shutil.rmtree(SITE_DIR)
  SITE_DIR.mkdir(parents=True)

  copy_assets()
  (SITE_DIR / "stylesheets" / "base.css").write_text(BASE_CSS)

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
      social_html=social_html(),
      nav_html=render_nav_html(rel),
      body=body,
      edit_url=html.escape(edit_url),
      copy_js=COPY_JS,
    )
    out = output_html_path(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html)

  print(f"docs: built {len(pages)} pages into {SITE_DIR}")


# ---------------------------------------------------------------------------
# Serve
# ---------------------------------------------------------------------------

def _raise_interrupt(*_):
  raise KeyboardInterrupt


def serve() -> None:
  signal.signal(signal.SIGTERM, _raise_interrupt)
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


def main() -> None:
  cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
  if cmd == "build":
    build()
  elif cmd == "serve":
    serve()
  else:
    print(f"unknown command: {cmd}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
  main()
