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
  if "abbreviation" in definition and definition["abbreviation"]:
    markdown += f" *({definition['abbreviation']})*"
  if "description" in definition and definition["description"]:
    markdown += f": {definition['description']}\n"
  return markdown

def glossary_markdown(glossary):
  markdown = ""
  for category, terms in glossary.items():
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

def tooltip_html(glossary, html):
  for category, terms in glossary.items():
    for term_key, definition in terms.items():
      if "description" in definition and definition["description"]:
        html = format_tooltip_html(term_key, definition, html)
  return html

# Page Hooks
def on_page_markdown(markdown, **kwargs):
  return markdown.replace("{{GLOSSARY_DEFINITIONS}}", glossary_markdown(glossary))

def on_page_content(html, **kwargs):
  if kwargs.get("page").title == "Glossary":
    return html
  return tooltip_html(glossary, html)
