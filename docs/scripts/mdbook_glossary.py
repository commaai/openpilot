import json
from pathlib import Path
import re
import sys
import tomllib


GLOSSARY_PLACEHOLDER = "{{GLOSSARY_DEFINITIONS}}"
GLOSSARY_PATH = Path("glossary.toml")
GLOSSARY_PAGE = "concepts/glossary.md"
DATA_SCRIPT_ID = "openpilot-glossary-data"


def load_glossary(root: Path) -> dict:
  with (root / GLOSSARY_PATH).open("rb") as f:
    glossary_data = tomllib.load(f)
  return glossary_data.get("glossary", {})


def generate_anchor_id(name: str) -> str:
  return name.replace(" ", "-").replace("_", "-").lower()


def humanize(name: str) -> str:
  return name.replace("_", " ")


def format_markdown_term(name: str, definition: dict) -> str:
  anchor_id = generate_anchor_id(name)
  display = definition.get("display", humanize(name))
  markdown = f'* <span id="{anchor_id}"><strong>{display}</strong></span>'

  abbreviation = definition.get("abbreviation")
  if abbreviation:
    markdown += f" *({abbreviation})*"

  description = definition.get("description")
  if description:
    markdown += f": {description}"

  return f"{markdown}\n"


def glossary_markdown(vocabulary: dict) -> str:
  markdown = []
  for category, terms in vocabulary.items():
    markdown.append(f"## {humanize(category).title()}\n")
    for name, definition in terms.items():
      markdown.append(format_markdown_term(name, definition))
    markdown.append("\n")
  return "".join(markdown).rstrip()


def strip_markdown(text: str) -> str:
  stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
  stripped = stripped.replace("`", "").replace("**", "").replace("*", "")
  return " ".join(stripped.split())


def glossary_payload(vocabulary: dict) -> list[dict]:
  payload = []

  for _category, terms in vocabulary.items():
    for name, definition in terms.items():
      description = definition.get("description", "").strip()
      if not description:
        continue

      display = str(definition.get("display", humanize(name)))
      abbreviation = definition.get("abbreviation")
      aliases = definition.get("aliases", [])

      search_terms = [display]
      if abbreviation:
        search_terms.append(str(abbreviation))
      search_terms.extend(str(alias) for alias in aliases)

      deduped_terms = []
      seen = set()
      for term in search_terms:
        normalized = term.casefold()
        if normalized in seen:
          continue
        seen.add(normalized)
        deduped_terms.append(term)

      payload.append({
        "anchor": generate_anchor_id(name),
        "description": strip_markdown(description),
        "display": display,
        "terms": deduped_terms,
      })

  return payload


def inject_glossary_payload(content: str, payload: list[dict]) -> str:
  payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
  script = f'<script id="{DATA_SCRIPT_ID}" type="application/json">{payload_json}</script>'
  return f"{content.rstrip()}\n\n{script}\n"


def chapter_source_path(chapter: dict) -> str:
  source_path = chapter.get("source_path") or chapter.get("path") or ""
  return str(source_path)


def walk_book_items(items: list, vocabulary: dict, payload: list[dict]) -> None:
  for item in items:
    if not isinstance(item, dict):
      continue

    if "Chapter" in item:
      chapter = item["Chapter"]
      source_path = chapter_source_path(chapter)
      content = chapter.get("content", "")

      if source_path.endswith(GLOSSARY_PAGE):
        chapter["content"] = content.replace(GLOSSARY_PLACEHOLDER, glossary_markdown(vocabulary))
      else:
        chapter["content"] = inject_glossary_payload(content, payload)

      walk_book_items(chapter.get("sub_items", []), vocabulary, payload)
      continue

    for value in item.values():
      if isinstance(value, list):
        walk_book_items(value, vocabulary, payload)


def supports_renderer() -> int:
  # The glossary preprocessor only mutates markdown content and raw HTML blocks,
  # so it is safe for any renderer mdBook asks about.
  return 0


def main() -> int:
  if len(sys.argv) > 1 and sys.argv[1] == "supports":
    return supports_renderer()

  context, book = json.load(sys.stdin)
  root = Path(context.get("root", "."))

  vocabulary = load_glossary(root)
  payload = glossary_payload(vocabulary)

  sections = book.get("sections") or book.get("items") or []
  walk_book_items(sections, vocabulary, payload)

  json.dump(book, sys.stdout, ensure_ascii=False)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
