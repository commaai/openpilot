import os
import requests

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

class CommaApi():
  def __init__(self, token=None):
    self.session = requests.Session()
    self.session.headers['User-agent'] ='OpenpilotTools'
    if token:
      self.session.headers['Authorization'] = 'JWT ' + token

  def request(self, method, endpoint, **kwargs):
    resp = self.session.request(method, API_HOST + '/' + endpoint, **kwargs)
    resp_json = resp.json()
    if isinstance(resp_json, dict) and resp_json.get('error'):
      e = APIError(resp_json['error'])
      e.status_code = resp.status_code
      raise e
    return resp_json

  def get(self, endpoint, **kwargs):
    return self.request('GET', endpoint, **kwargs)

  def post(self, endpoint, **kwargs):
    return self.request('POST', endpoint, **kwargs)

class APIError(Exception):
  pass
