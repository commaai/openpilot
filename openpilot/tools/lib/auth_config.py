import json
import os
from openpilot.common.hardware.hw import Paths
from openpilot.common.safe import read_json


class MissingAuthConfigError(Exception):
  pass


def get_token():
  auth = read_json(os.path.join(Paths.config_root(), 'auth.json'), {})
  return auth.get('access_token') if isinstance(auth, dict) else None


def set_token(token):
  os.makedirs(Paths.config_root(), exist_ok=True)
  with open(os.path.join(Paths.config_root(), 'auth.json'), 'w') as f:
    json.dump({'access_token': token}, f)


def clear_token():
  try:
    os.unlink(os.path.join(Paths.config_root(), 'auth.json'))
  except FileNotFoundError:
    pass
