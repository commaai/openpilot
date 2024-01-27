#!/usr/bin/env python3
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car.docs_definitions import Column
import os
import re

STAR_ICON_TEMPLATE = '<a href="##"><img valign="top" ' + \
  'src="https://media.githubusercontent.com/media/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
VIDEO_ICON_TEMPLATE = '<a href="{}" target="_blank">' + \
  '<img height="18px" src="https://media.githubusercontent.com/media/commaai/openpilot/master/docs/assets/icon-youtube.svg"></img></a>'
COLUMN_SEPARATOR = "|"
ARROW_SYMBOL = "➡️"

def get_column_headers():
  column_headers = [column.value for column in Column]
  return COLUMN_SEPARATOR.join(column_headers)

COLUMNS = f"{COLUMN_SEPARATOR}{get_column_headers()}{COLUMN_SEPARATOR}"
COLUMN_HEADER = f"|---|---|---|{'|'.join([':---:'] * (len(Column) - 3))}|"

def extract_car_info_from_text(text):
  pattern = r"<!-- ALL CAR INFO HERE -->(.*?)<!-- ALL CAR INFO HERE ENDS -->"
  matches = re.search(pattern, text, re.DOTALL)

  if matches:
    return matches.group(1).strip()
  else:
    raise ValueError("No car info found in CARS.md")

def split_info_by_line(extracted_info):
  return [info_line.strip() for info_line in extracted_info.split('\n') if info_line.strip()]

def convert_info_to_dict(car_info_lines):
  processed_car_info = []
  for line in car_info_lines:
    info_parts = line.split(COLUMN_SEPARATOR)
    car_info = {column.value: value.strip() for column, value in zip(Column, info_parts[1:])}
    car_info['Detail sentence'] = info_parts[-1].strip().replace('<!-- detail sentence:', '').replace(' -->', '')
    processed_car_info.append(car_info)
  return processed_car_info

def process_markdown_file(file_path):
  with open(file_path, 'r') as file:
    markdown_content = file.read()
  car_info = extract_car_info_from_text(markdown_content)
  # Remove headers by [2:]
  car_info_lines = split_info_by_line(car_info)[2:]

  return convert_info_to_dict(car_info_lines)

def extract_model_name_from_model_data(model_data):
  # Hack for body
  if model_data == 'body':
    return model_data

  # Extract only model name, without years and footnotes markup
  match = re.compile(r"^(.*?)(?:\s\d{4}|\s\d{4}-\d{2}|\[)", flags=re.MULTILINE).search(model_data)

  if match:
    return match.group(0)
  else:
    raise ValueError("Can't extract model name from", model_data)
    
def compare_car_info(old_array, new_array):
  changes = {
    "additions": [],
    "deletions": [],
    "modifications": [],
    "detail_sentence_changes": []
  }

  new_info_dict = {car['Model']: car for car in new_array}

  for old_car in old_array:
    model = old_car['Model']
    old_model_name = extract_model_name_from_model_data(model)

    # Search for the same model name in new info
    matched_model_in_new_info = next((key for key in new_info_dict if old_model_name in key), None)

    if not matched_model_in_new_info:
      changes['deletions'].append(old_car)
    else:
      model = matched_model_in_new_info
      modified = False
      current_state = new_info_dict[model].copy()

      # Check for detail sentence changes separately
      if old_car.get('Detail sentence') != new_info_dict[model].get('Detail sentence'):
        changes["detail_sentence_changes"].append(f"- Sentence for {model} changed!\n" +
                                  "  ```diff\n" +
                                  f"  - {old_car['Detail sentence']}\n" +
                                  f"  + {new_info_dict[model]['Detail sentence']}\n" +
                                  "  ```")

      # Check for other modifications
      for key, value in old_car.items():
        if key != 'Detail sentence' and new_info_dict[model].get(key) != value:
          modified = True
          if 'modified_fields' not in current_state:
            current_state['modified_fields'] = {}
          current_state['modified_fields'][key] = value

      if modified:
        changes['modifications'].append(current_state)
      # Remove models from 'new' dict that are also in 'old' to avoid marking them as additions.
      del new_info_dict[model]

  changes['additions'] = list(new_info_dict.values())
  return changes

def compare_car_dicts(old_car, new_car):
  return set(old_car) - set(new_car)

def format_changes_as_markdown(changes):
  markdown_builder = ["### ⚠️ This PR makes changes to the car information ⚠️"]
  change_titles = {
    "modifications": "## 🔀 Column Changes",
    "deletions": "## ❌ Removed",
    "additions": "## ➕ Added",
    "detail_sentence_changes": "## 📖 Detail Sentence Changes"
  }

  for change_type, title in change_titles.items():
    if changes[change_type]:
      markdown_builder.append(title)
      if change_type != "detail_sentence_changes":
        markdown_builder.append(COLUMNS)
        markdown_builder.append(COLUMN_HEADER)
        markdown_builder.extend(format_change_list(changes[change_type], change_type))
      else:
        # Special handling for detail sentence changes
        for detail_change in changes["detail_sentence_changes"]:
          markdown_builder.append(detail_change)

  return "\n".join(markdown_builder)

def format_change_list(change_list, change_type):
  formatted_changes = []
  for change in change_list:
    if change_type == "modifications":
      row = format_modified_row(change)
    else:
      row = COLUMN_SEPARATOR.join([change.get(col.value, '') for col in Column])
    formatted_changes.append(f"{COLUMN_SEPARATOR}{row}{COLUMN_SEPARATOR}")
  return formatted_changes

def format_modified_row(modified_car):
  modified_row = []
  for col in Column:
    value = modified_car.get(col.value, '')
    if col.value in modified_car.get('modified_fields', {}):
      old_value = modified_car['modified_fields'][col.value]
      modified_value = f"{old_value} {ARROW_SYMBOL} {value}"
      modified_row.append(modified_value)
    else:
      modified_row.append(value)
  return COLUMN_SEPARATOR.join(modified_row)

def print_car_info_diff():
  MD_BEFORE = os.path.join(BASEDIR, "docs", "CARS_1.md")
  MD_AFTER = os.path.join(BASEDIR, "docs", "CARS_2.md")

  x = process_markdown_file(MD_BEFORE)
  y = process_markdown_file(MD_AFTER)

  comparison_result = compare_car_info(x, y)
  if any(len(c) for c in comparison_result.values()):
    formatted_changes = format_changes_as_markdown(comparison_result)
    print(formatted_changes)

if __name__ == "__main__":
  print_car_info_diff()
