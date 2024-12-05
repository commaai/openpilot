import re
import tomllib

with open("docs/glossary.toml", "rb") as f:
  glossary_data = tomllib.load(f)

glossary = glossary_data.get("glossary", {})

def generate_anchor_id(name):
  return name.replace(" ", "-").replace("_", "-").lower()

def format_markdown_term(name, definition):
  anchor_id = generate_anchor_id(name)
  markdown = f"* [**{name.replace('_', ' ').title()}**](#{anchor_id})"
  if definition.get("abbreviation"):
    markdown += f" *({definition['abbreviation']})*"
  if definition.get("description"):
    markdown += f": {definition['description']}\n"
  return markdown

def glossary_markdown(vocabulary):
  markdown = ""
  for category, terms in vocabulary.items():
    markdown += f"## {category.replace('_', ' ').title()}\n\n"
    for name, definition in terms.items():
      markdown += format_markdown_term(name, definition)
  return markdown

def format_tooltip_html(term_key, definition, html):
  display_term = term_key.replace("_", " ").title()
  clean_description = re.sub(r"\[(.+)]\(.+\)", r"\1", definition["description"])
  glossary_link = (
    f"<a href='/concepts/glossary#{term_key}' class='tooltip-glossary-link' title='View in glossary'>Glossary🔗</a>"
  )
  return re.sub(
    re.escape(display_term),
    lambda match: (
      f"<span data-tooltip>{match.group(0)}"
      f"<span class='tooltip-content'>{clean_description} {glossary_link}</span>"
      f"</span>"
    ),
    html,
    flags=re.IGNORECASE,
  )


def tooltip_html(vocabulary, html):
  for _category, terms in vocabulary.items():
    for term_key, definition in terms.items():
      if definition.get("description"):
        pattern = (
          rf"(?<!\w)"
          rf"{re.escape(term_key.replace('_', ' ').title())}"
          rf"(?![^<]*<\/a>)(?!\([^)]*\))"  # check it's not a link
        )
        html = re.sub(
          pattern,
          lambda match: format_tooltip_html(term_key, definition, match.group(0)),
          html,
          flags=re.IGNORECASE,
        )
  return html

# Page Hooks
def on_page_markdown(markdown, **kwargs):
  return markdown.replace("{{GLOSSARY_DEFINITIONS}}", glossary_markdown(glossary))

def on_page_content(html, **kwargs):
  if kwargs.get("page").title == "Glossary":
    return html
  return tooltip_html(glossary, html)
