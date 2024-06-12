#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import xml.etree.ElementTree as ET
from typing import cast

import requests

TRANSLATIONS_DIR = pathlib.Path(__file__).resolve().parent
TRANSLATIONS_LANGUAGES = TRANSLATIONS_DIR / "languages.json"

OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROMPT = "You are a professional translator from English to {language} (ISO 639 language code). " + \
                "The following sentence or word is in the GUI of a software called openpilot, translate it accordingly."


def get_language_files(languages: list[str] = None) -> dict[str, pathlib.Path]:
  files = {}

  with open(TRANSLATIONS_LANGUAGES) as fp:
    language_dict = json.load(fp)

    for filename in language_dict.values():
      path = TRANSLATIONS_DIR / f"{filename}.ts"
      language = path.stem.split("main_")[1]

      if languages is None or language in languages:
        files[language] = path

  return files


def translate_phrase(text: str, language: str) -> str:
  response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    json={
      "model": OPENAI_MODEL,
      "messages": [
        {
          "role": "system",
          "content": OPENAI_PROMPT.format(language=language),
        },
        {
          "role": "user",
          "content": text,
        },
      ],
      "temperature": 0.8,
      "max_tokens": 1024,
      "top_p": 1,
    },
    headers={
      "Authorization": f"Bearer {OPENAI_API_KEY}",
      "Content-Type": "application/json",
    },
  )

  if 400 <= response.status_code < 600:
    raise requests.HTTPError(f'Error {response.status_code}: {response.json()}', response=response)

  data = response.json()

  return cast(str, data["choices"][0]["message"]["content"])


def translate_file(path: pathlib.Path, language: str, all_: bool) -> None:
  tree = ET.parse(path)

  root = tree.getroot()

  for context in root.findall("./context"):
    name = context.find("name")
    if name is None:
      raise ValueError("name not found")

    print(f"Context: {name.text}")

    for message in context.findall("./message"):
      source = message.find("source")
      translation = message.find("translation")

      if source is None or translation is None:
        raise ValueError("source or translation not found")

      if not all_ and translation.attrib.get("type") != "unfinished":
        continue

      llm_translation = translate_phrase(cast(str, source.text), language)

      print(f"Source: {source.text}\n" +
            f"Current translation: {translation.text}\n" +
            f"LLM translation: {llm_translation}")

      translation.text = llm_translation

  with path.open("w", encoding="utf-8") as fp:
    fp.write('<?xml version="1.0" encoding="utf-8"?>\n' +
             '<!DOCTYPE TS>\n' +
             ET.tostring(root, encoding="utf-8").decode())


def main():
  arg_parser = argparse.ArgumentParser("Auto translate")

  group = arg_parser.add_mutually_exclusive_group(required=True)
  group.add_argument("-a", "--all-files", action="store_true", help="Translate all files")
  group.add_argument("-f", "--file", nargs="+", help="Translate the selected files. (Example: -f fr de)")

  arg_parser.add_argument("-t", "--all-translations", action="store_true", default=False, help="Translate all sections. (Default: only unfinished)")

  args = arg_parser.parse_args()

  if OPENAI_API_KEY is None:
    print("OpenAI API key is missing. (Hint: use `export OPENAI_API_KEY=YOUR-KEY` before you run the script).\n" +
          "If you don't have one go to: https://beta.openai.com/account/api-keys.")
    exit(1)

  files = get_language_files(None if args.all_files else args.file)

  if args.file:
    missing_files = set(args.file) - set(files)
    if len(missing_files):
      print(f"No language files found: {missing_files}")
      exit(1)

  print(f"Translation mode: {'all' if args.all_translations else 'only unfinished'}. Files: {list(files)}")

  for lang, path in files.items():
    print(f"Translate {lang} ({path})")
    translate_file(path, lang, args.all_translations)


if __name__ == "__main__":
  main()
