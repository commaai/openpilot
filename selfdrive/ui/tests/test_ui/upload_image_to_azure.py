import argparse
import os
import sys
from openpilot.tools.lib.openpilotcontainers import OpenpilotCIContainer

run_id = os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('short_name')
args = parser.parse_args()

blob_name = f"{run_id}-{args.short_name}"

print(f"uploading file {args.image_path} to azure with blob_name {blob_name}", file=sys.stderr)
print(OpenpilotCIContainer.upload_file(args.image_path, blob_name, "image/png"))
