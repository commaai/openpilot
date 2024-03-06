import json
import os
from openpilot.system.hardware.hw import Paths


class MissingAuthConfigError(Exception):
  pass


def get_token():
  try:
    with open(os.path.join(Paths.config_root(), 'auth.json')) as f:
      auth = json.load(f)
      return auth['access_token']
  except Exception:
    return None


def set_token(token):
  os.makedirs(Paths.config_root(), exist_ok=True)
  with open(os.path.join(Paths.config_root(), 'auth.json'), 'w') as f:
    json.dump({'access_token': token}, f)


def clear_token():
  try:
    os.unlink(os.path.join(Paths.config_root(), 'auth.json'))
  except FileNotFoundError:
    pass
