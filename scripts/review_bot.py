
import sys
import json
import re
import time

# Read the file passed in as first arg
def get_pr_body():
    with open(sys.argv[1], 'r') as file:
        event_payload = json.load(file)
    return event_payload["pull_request"]["body"]

# Read template file
def read_template_file():
    template_file_path = ".github/pull_request_template.md"
    with open(template_file_path, 'r') as template_file:
        combined_templates = template_file.read()
    return combined_templates

# Separate out each template
template_separator = re.compile(r"<!--- \*{5}+ Template: (.*?) \*{5}\n\n(.*?)\n\n-->", re.DOTALL)
def separate_templates(combined_templates):
    matches = template_separator.findall(combined_templates)
    for (name,content) in matches:
        yield name,content

# Find fields in a template or pull request. They look like **field name**
field_finder = re.compile(r"\*{2}(.+?)\*{2}")
def find_field_set(content):
    return set(field_finder.findall(content))

if __name__ == "__main__":
    pr_body            = get_pr_body()
    fields_in_pr_body  = find_field_set(pr_body)
    combined_templates = read_template_file()
    templates          = separate_templates(combined_templates)

    # Calculate which templates match
    possible_template_matches = []
    for template_name,template_content in templates:
        required_fields = find_field_set(template_content)
        if fields_in_pr_body.issuperset(required_fields):
            possible_template_matches.append(template_name)

    time.sleep(20)
    # Return results
    if len(possible_template_matches) > 0:
        print("PR matches template(s): ",", ".join(possible_template_matches))
        sys.exit(0) # Pass
    else:
        print("PR does not match any known templates")
        sys.exit(1) # Fail
