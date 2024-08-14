import re
import tomllib

with open("docs/glossary.toml", "rb") as f:
  glossary = tomllib.load(f)


def glossary_markdown(glossary):
  markdown_string = ""

  for header, terms in glossary.items():
    markdown_string += f"## {header}\n"

    for name, definition in terms.items():
      markdown_string += f"* **{name}**"
      if "abbreviation" in definition:
        markdown_string += f" *({definition['abbreviation']})*"
      if "description" in definition:
        markdown_string += f": {definition['description']}\n"

  return markdown_string


def tooltip_html(glossary, html):
  for _, terms in glossary.items():
    for term, definition in terms.items():
      if "description" in definition:
        # Removes markdown link formating, but keeps the link text
        clean_description = re.sub(r"\[(.+)\]\(.+\)", r"\1", definition["description"])

        html = re.sub(
          re.escape(term),
          lambda match, descr=clean_description: f"<span data-tooltip='{descr}'>{match.group(0)}</span>",
          html,
          flags=re.IGNORECASE,
        )

  return html


def on_page_markdown(markdown, **kwargs):
  return markdown.replace("{{GLOSSARY_DEFINITIONS}}", glossary_markdown(glossary))


def on_page_content(html, **kwargs):
  # Don't add tooltips to the glossary page
  if kwargs.get("page").title == "Glossary":
    return html
  else:
    return tooltip_html(glossary, html)
