import os
import requests

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

class CommaApi():
  def __init__(self, token):
    self.session = requests.Session()
    self.session.headers.update({
      'Authorization': 'JWT ' + token,
      'User-agent': 'OpenpilotTools'
    })

  def get(self, endpoint, **kwargs):
    resp = self.session.get(API_HOST + '/' + endpoint, **kwargs)
    resp_json = resp.json()
    if isinstance(resp_json, dict) and resp_json.get('error'):
      e = APIError(resp_json['error'])
      e.status_code = resp.status_code
      raise e
    return resp_json

class APIError(Exception):
  pass
