#!/usr/bin/env python3

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from typing import TextIO, cast

import requests

OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROMPT = "You are a professional translator from English to {language} (ISO 639 language code)." \
                "The following sentence or word is in the GUI of a software called openpilot, translate it accordingly."


def print_log(text: str) -> None:
  print(text, file=sys.stderr)


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
      "max_tokens": 128,
      "top_p": 1,
    },
    headers={
      "Authorization": f"Bearer {OPENAI_API_KEY}",
    },
  )

  response.raise_for_status()

  data = response.json()

  return cast(str, data["choices"][0]["message"]["content"])


def translate_file(input_file: TextIO, output_file: TextIO, language: str, all: bool) -> None:
  tree = ET.parse(input_file)

  root = tree.getroot()

  root.attrib["language"] = language

  for context in root.findall("./context"):
    name = context.find("name")
    if name is None:
      raise ValueError("name not found")

    print_log(f"Context: {name.text}")

    for message in context.findall("./message"):
      source = message.find("source")
      translation = message.find("translation")

      if source is None or translation is None:
        raise ValueError("source or translation not found")

      if not all and translation.attrib.get("type") != "unfinished":
        continue

      llm_translation = translate_phrase(cast(str, source.text), language)

      print_log(f"Source: {source.text}\n"
                f"Current translation: {translation.text}\n"
                f"LLM translation: {llm_translation}")

      translation.text = llm_translation

  output_file.write('<?xml version="1.0" encoding="utf-8"?>\n' +
                    '<!DOCTYPE TS>\n' +
                    ET.tostring(root, encoding="utf-8").decode())


def main() -> None:
  if OPENAI_API_KEY is None:
    print("OpenAI api key is missing. (Hint: use `export OPENAI_API_KEY=YOUR-KEY` before you run the script).\n"
          "If you don't have one go to: https://beta.openai.com/account/api-keys.")
    exit(1)

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("input", nargs="?", type=argparse.FileType("r"), help="The input file")
  arg_parser.add_argument("output", nargs="?", type=argparse.FileType("w", encoding="utf-8"), default=sys.stdout, help="The output file")
  arg_parser.add_argument("-a", "--all", action="store_true", default=False, help="Translate all")
  arg_parser.add_argument("-l", "--language", required=True, help="Translate to (Language code)")

  args = arg_parser.parse_args()

  print_log(f"Translates to {args.language} ({'all' if args.all else 'only unfinished'})")

  translate_file(args.input, args.output, args.language, args.all)


if __name__ == "__main__":
  main()
