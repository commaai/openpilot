import os
import requests
API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')
# API_HOST = os.getenv('API_HOST', 'https://api.aks.comma.ai')

# TODO: this should be merged into common.api

class CommaApi:
  def __init__(self, token=None):
    self.session = requests.Session()
    self.session.headers['User-agent'] = 'OpenpilotTools'
    if token:
      self.session.headers['Authorization'] = 'JWT ' + token

  def request(self, method, endpoint, **kwargs):
    url = API_HOST + '/' + endpoint
    print(f'[API REQUEST] {method} {url}')
    with self.session.request(method, url, **kwargs) as resp:
      print(f'[API RESPONSE] {method} {url} -> {resp.status_code}')
      if resp.status_code == 503:
        # Check if it's from nginx (has Server header) or backend pod
        server_header = resp.headers.get('Server', 'unknown')
        print(f'503 source: Server={server_header}, Headers={dict(resp.headers)}')
        e = APIError(f"503: Service Temporarily Unavailable (from {server_header})")
        e.status_code = 503
        raise e

      resp_json = resp.json()
      if isinstance(resp_json, dict) and resp_json.get('error'):
        if resp.status_code in (401, 403):
          raise UnauthorizedError('Unauthorized. Authenticate with tools/lib/auth.py')

        e = APIError(str(resp.status_code) + ":" + resp_json.get('description', str(resp_json['error'])))
        e.status_code = resp.status_code
        raise e
      return resp_json

  def get(self, endpoint, **kwargs):
    return self.request('GET', endpoint, **kwargs)

  def post(self, endpoint, **kwargs):
    return self.request('POST', endpoint, **kwargs)

class APIError(Exception):
  pass

class UnauthorizedError(Exception):
  pass
