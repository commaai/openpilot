import argparse
import os
import hashlib

from openpilot.common.basedir import BASEDIR

DEFAULT_MODEL_NAME_PATH = os.path.join(BASEDIR, "common", "model.h")
MODEL_HASH_PATH = os.path.join(BASEDIR, "sunnypilot", "modeld", "tests", "model_hash")
ONNX_PATH = os.path.join(BASEDIR, "selfdrive", "modeld", "models", "supercombo.onnx")


def get_hash(path: str) -> str:
  sha256_hash = hashlib.sha256()
  with open(path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
      sha256_hash.update(byte_block)
  return sha256_hash.hexdigest()


def update_model_hash():
  new_hash = get_hash(ONNX_PATH)
  with open(MODEL_HASH_PATH, "w") as f:
    f.write(new_hash)
  print(f"Generated and updated new hash to {MODEL_HASH_PATH}")


def get_current_default_model_name():
  print("[GET DEFAULT MODEL NAME]")
  with open(DEFAULT_MODEL_NAME_PATH) as f:
    name = f.read().split('"')[1]
  print(f'Current default model name: "{name}"')

  return name


def update_default_model_name(name: str):
  print("[CHANGE DEFAULT MODEL NAME]")
  with open(DEFAULT_MODEL_NAME_PATH, "w") as f:
    f.write(f'#define DEFAULT_MODEL "{name}"\n')
  print(f'New default model name: "{name}"')
  print("[DONE]")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Update default model name and hash")
  parser.add_argument("--new_name", type=str, help="New default model name")
  args = parser.parse_args()

  if not args.new_name:
    print("Warning: No new default model name provided. Use --new_name to specify")
    print("Default model name and hash will not be updated! (aborted)")
    exit(0)

  current_name = get_current_default_model_name()
  new_name = f"{args.new_name} (Default)"
  if current_name == new_name:
    print(f'Proposed default model name: "{new_name}"')
    confirm = input("Proposed default model name is the same as the current default model name. Confirm? (y/n):").upper().strip()
    if confirm != "Y":
      print("Default model name and hash will not be updated! (aborted)")
      exit(0)

  update_default_model_name(new_name)
  update_model_hash()
