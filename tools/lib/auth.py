import json
import os
from common.file_helpers import mkdirs_exists_ok

CONFIG_DIR = os.path.expanduser('~/.comma/')
mkdirs_exists_ok(CONFIG_DIR)

def get_token():
  try:
    with open(os.path.join(CONFIG_DIR, 'auth.json')) as f:
      auth = json.load(f)
      return auth['access_token']
  except:
    raise MissingAuthConfigError('Login with tools/lib/login.py')

class MissingAuthConfigError(Exception):
  pass
