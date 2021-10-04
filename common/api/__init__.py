import jwt
import os
import requests
from datetime import datetime, timedelta
from common.basedir import PERSIST
from selfdrive.version import version

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

class Api():
  def __init__(self, dongle_id):
    self.dongle_id = dongle_id
    with open(f"{PERSIST}/comma/id_rsa") as f:
      self.private_key = f.read()

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, **params):
    return api_get(endpoint, method=method, timeout=timeout, access_token=access_token, **params)

  def get_token(self):
    now = datetime.utcnow()
    payload = {
      'identity': self.dongle_id,
      'nbf': now,
      'iat': now,
      'exp': now + timedelta(hours=1)
    }
    token = jwt.encode(payload, self.private_key, algorithm='RS256')
    if isinstance(token, bytes):
      token = token.decode('utf8')
    return token
    

def api_get(endpoint, method='GET', timeout=None, access_token=None, **params):
  headers = {}
  if access_token is not None:
    headers['Authorization'] = f"JWT {access_token}"

  headers['User-Agent'] = f"openpilot-{version}"

  return requests.request(method, f"{API_HOST}/{endpoint}", timeout=timeout, headers=headers, params=params)
