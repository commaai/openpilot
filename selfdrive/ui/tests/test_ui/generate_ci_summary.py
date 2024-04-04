import os
import json
import pathlib
from openpilot.tools.lib.openpilotcontainers import OpenpilotCIContainer

run_id = os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
TEST_DIR = pathlib.Path(__file__).parent
SCREENSHOTS_DIR = TEST_DIR / "report" / "screenshots"

summary = open(os.environ["GITHUB_STEP_SUMMARY"], 'a+')
languages = json.load(open(os.path.normpath(__file__ + "/../../../translations/languages.json"), "r"))

summary.write("# UI screenshots\n\n")

for language_dir in SCREENSHOTS_DIR.iterdir():
  language_name = list(languages.keys())[list(languages.values()).index(language_dir.name)]
  summary.write(f"## {language_name}\n\n")

  for case in language_dir.glob("**/*"):
    name = str(case.relative_to(SCREENSHOTS_DIR)).replace("/", "-").replace(".png", "")
    blob_name = f"{run_id}-{name}"
    link = OpenpilotCIContainer.upload_file(str(case), blob_name, "image/png")
    summary.write(f"### {name}\n\n![{name}]({link})\n\n")

summary.close()