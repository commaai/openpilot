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

DOCS_DIR = Path(__file__).resolve().parent
SITE_DIR = DOCS_DIR / "_site"
TEMPLATE_FILE = DOCS_DIR / "template.html"
EXCLUDE_DIRS = {"_site", "__pycache__"}

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
GLOSSARY_SKIP = frozenset("a code h1 h2 h3 h4 h5 h6 kbd pre script style".split())

_ENTITY = re.compile(r"&(?:#x?[0-9a-fA-F]+|[a-zA-Z]+);")
_LIST = re.compile(r"^(\s*)([*+-]|\d+\.)\s+(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_HR = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_ATTR_URL = re.compile(r"""(?P<pre>\b(?:href|src)=(?P<q>["']))(?P<url>.*?)(?P=q)""")
_VOID = frozenset("br img hr meta link input".split())
_URL = re.compile(r"https?://[^\s<>\[\]\"']+")
_ADMONITION = re.compile(r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]$", re.I)


def page_route(path: str) -> str:
  path = path.removesuffix(".md")
  return posixpath.dirname(path) or "." if posixpath.basename(path) == "index" else path


def page_href(current: str, target: str) -> str:
  route = posixpath.relpath(page_route(target), page_route(current))
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


def rewrite_html_urls(fragment: str, page: str) -> str:
  def repl(m: re.Match[str]) -> str:
    r = rewrite_relative_url(m.group("url"), page)
    return m.group(0) if r is None else f'{m.group("pre")}{r}{m.group("q")}'

  return _ATTR_URL.sub(repl, fragment)


def esc(text: str, attr: bool = False) -> str:
  held: list[str] = []

  def hold(m: re.Match[str]) -> str:
    held.append(m.group(0))
    return f"\0{len(held) - 1}\0"

  return re.sub(r"\0(\d+)\0", lambda m: held[int(m.group(1))], html.escape(_ENTITY.sub(hold, text), quote=attr))


def clean_tooltip(description: str) -> str:
  text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", description)
  return re.sub(r"\s+", " ", re.sub(r"[*_~]", "", re.sub(r"`([^`]+)`", r"\1", text))).strip()


def glossary_slug(label: str) -> str:
  return label.replace(" ", "-").replace("_", "-").lower()


GLOSSARY_TERMS = [(glossary_slug(l), re.compile(rf"(?<!\w){re.escape(l)}(?!\w)", re.I), clean_tooltip(d)) for l, d in GLOSSARY_DESCRIPTIONS.items()]
GLOSSARY_DEFINITIONS = "\n".join(f'* <span id="{glossary_slug(l)}"></span>**{l}**: {d}' for l, d in GLOSSARY_DESCRIPTIONS.items())


def inject_glossary(body: str, page: str) -> str:
  if page == GLOSSARY_PAGE:
    return body
  route = "." if page == "index.md" else page.removesuffix(".md")
  base, seen, out, skip, depth = f"{posixpath.relpath(GLOSSARY_ROUTE, route)}/#", set(), [], None, 0
  for part in re.split(r"(<[^>]+>)", body):
    if not part:
      continue
    if part.startswith("<"):
      out.append(part)
      if part.startswith("<!") or not (m := re.match(r"</?\s*([a-zA-Z0-9]+)", part)):
        continue
      tag = m.group(1).lower()
      if tag not in GLOSSARY_SKIP:
        continue
      closing, void = part.startswith("</"), part.endswith("/>") or tag in _VOID
      if closing and skip == tag and depth:
        depth -= 1
        skip = None if not depth else skip
      elif not closing and not void:
        skip, depth = (tag, 1) if skip is None else (skip, depth + (skip == tag))
      continue
    if depth:
      out.append(part)
      continue
    cur, text = 0, part
    while True:
      best = None
      for order, (slug, pat, tip) in enumerate(GLOSSARY_TERMS):
        if slug in seen or (found := pat.search(text, cur)) is None:
          continue
        cand = (found.start(), found.start() - found.end(), order, slug, tip, found.end(), found.group(0))
        if best is None or cand[:3] < best[:3]:
          best = cand
      if best is None:
        out.append(text[cur:])
        break
      start, _, _, slug, tip, end, matched = best
      out.append(text[cur:start])
      out.append(
        f'<a class="glossary-term" data-glossary-term="" href="{base}{slug}">'
        + f'<span class="glossary-term__label">{matched}</span>'
        + f'<span class="glossary-term__tooltip" data-search-exclude="">{esc(tip)}</span></a>'
      )
      seen.add(slug)
      cur = end
  return "".join(out)


def slugify(text: str) -> str:
  text = html.unescape(re.sub(r"<[^>]+>", "", text)).lower()
  return re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)).strip("-")


def _parse_link(text: str, start: int) -> tuple[str, str, int] | None:
  if start >= len(text) or text[start] != "[":
    return None
  depth, i = 0, start
  while i < len(text):
    depth += (text[i] == "[") - (text[i] == "]")
    if text[i] == "]" and depth == 0:
      label = text[start + 1 : i]
      if i + 1 >= len(text) or text[i + 1] != "(":
        return None
      j, dp = i + 2, 1
      while j < len(text) and dp:
        dp += (text[j] == "(") - (text[j] == ")")
        j += 1
      return None if dp else (label, text[i + 2 : j - 1], j)
    i += 1
  return None


def autolink_plain(text: str) -> str:
  parts: list[str] = []
  last = 0
  for m in _URL.finditer(text):
    start = m.start()
    if start > 0 and text[start - 1].isalnum():
      continue
    parts.append(esc(text[last:start]))
    url = m.group(0).rstrip(".,;:!?)]")
    parts.append(f'<a href="{esc(url, True)}">{esc(url)}</a>')
    last = start + len(url)
  parts.append(esc(text[last:]))
  return "".join(parts)


def render_inline(text: str, page: str) -> str:
  out, i, n = [], 0, len(text)
  while i < n:
    if text[i] == "\n" and i >= 2 and text[i - 2 : i] == "  " and out and out[-1].endswith("  "):
      out[-1] = out[-1][:-2]
      out.append("<br>\n")
      i += 1
      continue
    if text[i] == "`" and (end := text.find("`", i + 1)) != -1:
      out.append(f"<code>{esc(text[i + 1 : end])}</code>")
      i = end + 1
      continue
    if text[i] == "!" and i + 1 < n and text[i + 1] == "[" and (p := _parse_link(text, i + 1)):
      label, url, end = p
      src = rewrite_relative_url(url, page) or url
      out.append(f'<img alt="{esc(label, True)}" src="{esc(src, True)}">')
      i = end
      continue
    if text[i] == "[" and (p := _parse_link(text, i)):
      label, url, end = p
      href = rewrite_relative_url(url, page) or url
      out.append(f'<a href="{esc(href, True)}">{render_inline(label, page)}</a>')
      i = end
      continue
    if text[i] == "<":
      if text.startswith("<!--", i):
        end = text.find("-->", i + 4)
        end = n if end < 0 else end + 3
        out.append(rewrite_html_urls(text[i:end], page))
        i = end
        continue
      if m := re.match(r"<[^>]+>", text[i:]):
        out.append(rewrite_html_urls(m.group(0), page))
        i += len(m.group(0))
        continue
    if (text.startswith("**", i) or text.startswith("__", i)) and (end := text.find(text[i : i + 2], i + 2)) != -1:
      out.append(f"<strong>{render_inline(text[i + 2 : end], page)}</strong>")
      i = end + 2
      continue
    if text[i] in "*_" and i + 1 < n and text[i + 1] not in " \t\n" and (end := text.find(text[i], i + 1)) > i + 1:
      out.append(f"<em>{render_inline(text[i + 1 : end], page)}</em>")
      i = end + 1
      continue
    j = i + 1
    while j < n and text[j] not in "`[<!*_":
      j += 1
    out.append(autolink_plain(text[i:j]))
    i = j
  return "".join(out)


def _trow(line: str) -> list[str]:
  return [c.strip() for c in line.strip().removeprefix("|").removesuffix("|").split("|")]


def _is_sep(line: str) -> bool:
  return "|" in line and all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in _trow(line))


def _align(sep: str) -> str:
  s = sep.strip()
  if s.startswith(":") and s.endswith(":"):
    return ' style="text-align: center;"'
  if s.endswith(":"):
    return ' style="text-align: right;"'
  if s.startswith(":"):
    return ' style="text-align: left;"'
  return ""


def _list_info(line: str) -> tuple[int, str, str] | None:
  m = _LIST.match(line)
  return None if not m else (len(m.group(1)) // 4, "ol" if m.group(2)[-1] == "." else "ul", m.group(3))


def _render_blocks(text: str, page: str) -> str:
  lines, out, i, n = text.splitlines(), [], 0, 0
  n = len(lines)
  while i < n:
    line, s = lines[i], lines[i].strip()
    if not s:
      i += 1
      continue

    if s.startswith("```"):
      lang, body = s[3:].strip(), []
      i += 1
      while i < n and not lines[i].strip().startswith("```"):
        body.append(lines[i])
        i += 1
      if i < n:
        i += 1
      code = html.escape("\n".join(body) + ("\n" if body else ""))
      cls = f' class="language-{html.escape(lang)}"' if lang else ""
      out.append(f"<pre><code{cls}>{code}</code></pre>")
      continue

    if m := _HEADING.match(s):
      content, level = m.group(2).rstrip("#").strip(), len(m.group(1))
      sid = slugify(content)
      out.append(f'<h{level} id="{sid}">{render_inline(content, page)}<a class="headerlink" href="#{sid}" title="Permanent link">#</a></h{level}>')
      i += 1
      continue

    if _HR.fullmatch(s):
      out.append("<hr>")
      i += 1
      continue

    if "|" in line and i + 1 < n and _is_sep(lines[i + 1]):
      headers, aligns = _trow(line), [_align(c) for c in _trow(lines[i + 1])]
      i += 2
      rows = []
      while i < n and "|" in lines[i] and lines[i].strip():
        rows.append(_trow(lines[i]))
        i += 1
      parts = (
        ["<table>", "<thead>", "<tr>"]
        + [f"<th{aligns[j] if j < len(aligns) else ''}>{render_inline(h, page)}</th>" for j, h in enumerate(headers)]
        + ["</tr>", "</thead>", "<tbody>"]
      )
      for row in rows:
        parts.append("<tr>")
        for j in range(len(headers)):
          parts.append(f"<td{aligns[j] if j < len(aligns) else ''}>{render_inline(row[j] if j < len(row) else '', page)}</td>")
        parts.append("</tr>")
      out.append("\n".join(parts + ["</tbody>", "</table>"]))
      continue

    if _list_info(line):
      items: list[tuple[int, str, list[str]]] = []
      while i < n:
        if not lines[i].strip():
          if i + 1 < n and _list_info(lines[i + 1]):
            i += 1
            continue
          break
        info = _list_info(lines[i])
        if not info:
          break
        level, kind, body = info
        chunk = [body]
        i += 1
        while i < n and lines[i].strip() and _list_info(lines[i]) is None:
          t = lines[i].strip()
          if t.startswith(("```", ">")) or _HEADING.match(t) or _HR.fullmatch(t):
            break
          chunk.append(lines[i])
          i += 1
        items.append((level, kind, chunk))

      def render_list(items: list[tuple[int, str, list[str]]], start: int, min_level: int) -> tuple[str, int]:
        if start >= len(items) or items[start][0] < min_level:
          return "", start
        kind, chunks, idx = items[start][1], [f"<{items[start][1]}>"], start
        while idx < len(items) and items[idx][0] >= min_level:
          level, ikind, body_lines = items[idx]
          if level > min_level:
            nested, idx = render_list(items, idx, level)
            chunks[-1] = (chunks[-1][:-5] + nested + "</li>") if chunks[-1].endswith("</li>") else chunks[-1] + nested
            continue
          if ikind != kind:
            chunks += [f"</{kind}>", f"<{ikind}>"]
            kind = ikind
          idx += 1
          body = render_inline("\n".join(body_lines), page)
          nested = ""
          if idx < len(items) and items[idx][0] > min_level:
            nested, idx = render_list(items, idx, min_level + 1)
          chunks.append(f"<li>{body}{nested}\n</li>" if nested else f"<li>{body}</li>")
        chunks.append(f"</{kind}>")
        return "\n".join(chunks), idx

      out.append(render_list(items, 0, items[0][0])[0])
      continue

    if s.startswith(">"):
      q = []
      while i < n and lines[i].strip().startswith(">"):
        q.append(re.sub(r"^>\s?", "", lines[i].strip()))
        i += 1
      m = _ADMONITION.match(q[0].strip()) if q else None
      if m:
        kind = m.group(1).lower()
        title = m.group(1).capitalize()
        body = _render_blocks("\n".join(q[1:]), page)
        out.append(f'<div class="admonition {kind}">\n<p class="admonition-title">{title}</p>\n{body}\n</div>')
      else:
        out.append(f"<blockquote>\n<p>{render_inline(chr(10).join(q), page)}</p>\n</blockquote>")
      continue

    if s.startswith("<!--"):
      out.append(rewrite_html_urls(line, page))
      i += 1
      # Preserve a blank line after HTML comments (python-markdown does).
      if i < n and not lines[i].strip():
        out.append("")
        while i < n and not lines[i].strip():
          i += 1
      continue

    buf = [line]
    i += 1
    while i < n and lines[i].strip():
      t = lines[i].strip()
      if t.startswith(("```", ">")) or _HEADING.match(t) or _HR.fullmatch(t):
        break
      if "|" in lines[i] and i + 1 < n and _is_sep(lines[i + 1]):
        break
      buf.append(lines[i])
      i += 1
    out.append(f"<p>{render_inline(chr(10).join(buf), page)}</p>")

  return "\n".join(out)


def render_markdown(text: str, page: str) -> str:
  return inject_glossary(_render_blocks(text, page), page)


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
  out.write_text(
    "\n".join(
      [
        "<!doctype html>",
        f'<meta http-equiv="refresh" content="0; url={html.escape(target)}">',
        f'<link rel="canonical" href="{html.escape(target)}">',
        f"<script>location.replace({json.dumps(target)} + location.search + location.hash)</script>",
      ]
    )
  )


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
    (path.relative_to(DOCS_DIR), path.read_text())
    for path in sorted(DOCS_DIR.rglob("*.md"))
    if path != DOCS_DIR / "README.md" and not any(part in EXCLUDE_DIRS for part in path.relative_to(DOCS_DIR).parts)
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
    page_html = template
    for name, value in {
      "TITLE": html.escape(title),
      "ROOT": root,
      "HOME_HREF": page_href(rel, "index.md"),
      "NAV": render_nav_html(rel),
      "BODY": body,
      "EDIT_URL": html.escape(f"{REPO_URL}blob/master/docs/{edit_path}"),
    }.items():
      page_html = page_html.replace(f"{{{{{name}}}}}", value)
    out = SITE_DIR / ("" if route == "." else route) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html)
    write_html_redirect(rel_path)

  print(f"docs: built {len(pages)} pages into {SITE_DIR}")


def _watched_files() -> list[Path]:
  return [p for p in DOCS_DIR.rglob("*") if p.is_file() and not any(part in EXCLUDE_DIRS for part in p.relative_to(DOCS_DIR).parts)]


def serve() -> None:
  build()
  mtimes = {p: p.stat().st_mtime for p in _watched_files()}
  handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE_DIR))
  httpd = http.server.ThreadingHTTPServer(("", 0), handler)
  print(f"docs: serving on http://localhost:{httpd.server_port}/ (watching for changes)")
  try:
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    while True:
      time.sleep(0.5)
      new_mtimes = {p: p.stat().st_mtime for p in _watched_files()}
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
