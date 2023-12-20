
import sys
import json
import re
from github import Github
import os

BOT_REVIEW_LABEL = "bot-review"

# Read the pr event file, which is passed in as first arg
def read_pr_event():
    with open(sys.argv[1], 'r') as file:
        event_payload = json.load(file)
    return event_payload

# Read template file
def read_template_file():
    template_file_path = ".github/pull_request_template.md"
    with open(template_file_path, 'r') as template_file:
        combined_templates = template_file.read()
    return combined_templates

# Separate out each template
template_separator = re.compile(r"<!--- \*{5} Template: (.*?) \*{5}\n\n(.*?)\n\n-->", re.DOTALL)
def separate_templates(combined_templates):
    matches = template_separator.findall(combined_templates)
    for (name,content) in matches:
        yield name,content

# Find fields in a template or pull request. They look like **field name**
field_finder = re.compile(r"\*{2}(.+?)\*{2}")
def find_field_set(content):
    return set(field_finder.findall(content))

if __name__ == "__main__":
    g = Github(os.environ['GITHUB_TOKEN'])

    pr_event = read_pr_event()
    repo_name = pr_event['repository']['full_name']
    pr_number = pr_event['pull_request']['number']
    pr_body = pr_event["pull_request"]["body"]
    pr = g.get_repo(repo_name).get_pull(pr_number)
    pr.add_to_labels(BOT_REVIEW_LABEL)

    fields_in_pr_body = find_field_set(pr_body)
    combined_templates = read_template_file()
    templates = separate_templates(combined_templates)

    # Calculate which templates match
    possible_template_matches = []
    for template_name,template_content in templates:
        required_fields = find_field_set(template_content)
        if fields_in_pr_body.issuperset(required_fields):
            possible_template_matches.append(template_name)

    # Return results
    if len(possible_template_matches) > 0:
        print("PR matches template(s): ",", ".join(possible_template_matches))
        pr.remove_from_labels(BOT_REVIEW_LABEL)
        sys.exit(0) # Pass
    else:
        print("PR does not match any known templates")
        sys.exit(1) # Fail

