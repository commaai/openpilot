import json
import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / 'models'
COMPILED_FLAGS_PATH = MODELS_DIR / 'tg_compiled_flags.json'


def set_tinygrad_backend_from_compiled_flags() -> None:
  if os.path.isfile(COMPILED_FLAGS_PATH):
    with open(COMPILED_FLAGS_PATH) as f:
      flags = json.load(f)
    for k, v in flags.items():
      os.environ.setdefault(k, str(v))
