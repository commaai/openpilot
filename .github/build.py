import pathlib

GITHUB_FOLDER = pathlib.Path(__file__).parent

PULL_REQUEST_TEMPLATES = (GITHUB_FOLDER / "PULL_REQUEST_TEMPLATE")

order = ["fingerprint", "car_bugfix", "bugfix", "car_port", "refactor"]

def create_pull_request_template():
  with open(GITHUB_FOLDER / "pull_request_template.md", "w") as f:
    f.write("<!-- Please copy and paste the relevant template -->\n\n")

    for t in order:
      template = PULL_REQUEST_TEMPLATES / f"{t}.md"
      text = template.read_text()

      # Remove metadata for GitHub
      start = text.find("---")
      end = text.find("---", start+1)
      text = text[end + 4:]

      # Remove comments
      text = text.replace("<!-- ", "").replace("-->", "")

      f.write(f"<!--- ***** Template: {template.stem.replace('_', ' ').title()} *****\n")
      f.write(text)
      f.write("\n\n")
      f.write("-->\n\n")

create_pull_request_template()
