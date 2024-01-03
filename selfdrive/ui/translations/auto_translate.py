#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import cast

import requests

MODEL = "gpt-4"

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

logger = logging.getLogger()

handler = logging.StreamHandler(sys.stdout)

logger.addHandler(handler)
logger.setLevel(logging.INFO)


def translate_phrase(text: str, to: str) -> str:
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": f"Translate the following text from English to {to} (ISO 639 language code)",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "temperature": 0.8,
            "max_tokens": 64,
            "top_p": 1,
        },
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )

    data = response.json()

    return cast(str, data["choices"][0]["message"]["content"])


def translate_file(
    path: str, language: str, only_unfinished: bool, output_path: str
) -> None:
    tree = ET.parse(path)

    root = tree.getroot()

    root.attrib["language"] = language

    for context in root.findall("./context"):
        name = context.find("name")
        if name is None:
            raise ValueError("name not found")

        logger.info("Context: %s", name.text)

        for message in context.findall("./message"):
            source = message.find("source")
            translation = message.find("translation")

            if source is None or translation is None:
                raise ValueError("source or translation not found")

            if only_unfinished and translation.attrib.get("type") != "unfinished":
                continue

            llm_translation = translate_phrase(cast(str, source.text), language)

            logger.info("Source: %s", source.text)
            logger.info("Current translation: %s", translation.text)
            logger.info("LLM translation: %s", llm_translation)

            translation.text = llm_translation

    # tree.write(output_path, encoding="utf-8", xml_declaration=True) not add <!DOCTYPE TS>

    with open(output_path, "w", encoding="utf-8") as fp:
        doc_type = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n'
        tostring = ET.tostring(root, encoding="utf-8")

        fp.write(doc_type)
        fp.write(tostring.decode())


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-s", "--source", help="The source file")
    arg_parser.add_argument(
        "-u",
        "--only-unfinished",
        action="store_true",
        default=False,
        help="Translate only unfinished",
    )
    arg_parser.add_argument(
        "-l", "--language", required=True, help="Translate to (Language code)"
    )
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output file")

    args = arg_parser.parse_args()

    logger.info(
        "Translates %s (%s) from English to %s.",
        args.source,
        "only unfinished" if args.only_unfinished else "all",
        args.language,
    )

    translate_file(args.source, args.language, args.only_unfinished, args.output)


if __name__ == "__main__":
    main()
