import jwt
import requests
from datetime import datetime, timedelta

from selfdrive.version import version

class Api(object):
  def __init__(self, dongle_id, private_key):
    self.dongle_id = dongle_id
    self.private_key = private_key

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, **params):
    return api_get(endpoint, method=method, timeout=timeout, access_token=access_token, **params)

  def get_token(self):
    return jwt.encode({'identity': self.dongle_id, 'exp': datetime.utcnow() + timedelta(hours=1)}, self.private_key, algorithm='RS256')

def api_get(endpoint, method='GET', timeout=None, access_token=None, **params):
  backend = "https://api.commadotai.com/"

  headers = {}
  if access_token is not None:
    headers['Authorization'] = "JWT "+access_token

  headers['User-Agent'] = "openpilot-" + version

  return requests.request(method, backend+endpoint, timeout=timeout, headers = headers, params=params)

