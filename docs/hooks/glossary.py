import re
import tomllib

with open("docs/glossary.toml", "rb") as f:
  glossary_data = tomllib.load(f)

glossary = glossary_data.get("glossary", {})

def glossary_markdown(glossary):
  markdown_string = ""

  # Iterate over each category and terms
  for category, terms in glossary.items():
    markdown_string += f"## {category.replace('_', ' ').title()}\n"

    for name, definition in terms.items():
      markdown_string += f"* **{name.replace('_', ' ').title()}**"
      if "abbreviation" in definition and definition["abbreviation"]:
        markdown_string += f" *({definition['abbreviation']})*"
      if "description" in definition and definition["description"]:
        markdown_string += f": {definition['description']}\n"

  return markdown_string


def tooltip_html(glossary, html):
  for category, terms in glossary.items():
    for term, definition in terms.items():
      if "description" in definition and definition["description"]:
        # Removes Markdown link formatting, but keeps the link text
        clean_description = re.sub(r"\[(.+)]\(.+\)", r"\1", definition["description"])

        # Embed a tooltip-content element
        html = re.sub(
          re.escape(term),
          lambda match, descr=clean_description: (
            f"<span data-tooltip>{match.group(0)}"
            f"<span class='tooltip-content'>{descr}</span>"
            f"</span>"
          ),
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
