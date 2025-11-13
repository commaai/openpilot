import os
import requests
import time
API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

# TODO: this should be merged into common.api

class CommaApi:
  def __init__(self, token=None):
    self.session = requests.Session()
    self.session.headers['User-agent'] = 'OpenpilotTools'
    if token:
      self.session.headers['Authorization'] = 'JWT ' + token

  def request(self, method, endpoint, **kwargs):
    retries = 5
    for i in range(retries):
      with self.session.request(method, API_HOST + '/' + endpoint, **kwargs) as resp:
        if 500 <= resp.status_code < 600 and i < (retries - 1):  # don't sleep on last attempt
          print("API error, retrying in 1s")
          time.sleep(1)
          continue

        resp_json = resp.json()
        if isinstance(resp_json, dict) and resp_json.get('error'):
          if resp.status_code in [401, 403]:
            raise UnauthorizedError('Unauthorized. Authenticate with tools/lib/auth.py')

          e = APIError(str(resp.status_code) + ":" + resp_json.get('description', str(resp_json['error'])))
          e.status_code = resp.status_code
          raise e
        return resp_json
    return None

  def get(self, endpoint, **kwargs):
    return self.request('GET', endpoint, **kwargs)

  def post(self, endpoint, **kwargs):
    return self.request('POST', endpoint, **kwargs)

class APIError(Exception):
  pass

class UnauthorizedError(Exception):
  pass
