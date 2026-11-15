import os
import requests
from requests.adapters import HTTPAdapter, Retry
API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

# TODO: this should be merged into common.api

class CommaApi:
  def __init__(self, token=None):
    self.session = requests.Session()
    self.session.headers['User-agent'] = 'OpenpilotTools'
    if token:
      self.session.headers['Authorization'] = 'JWT ' + token

    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504], raise_on_status=False)
    self.session.mount('https://', HTTPAdapter(max_retries=retries))

  def request(self, method, endpoint, **kwargs):
    with self.session.request(method, API_HOST + '/' + endpoint, **kwargs) as resp:
      status = resp.status_code
      if status == 401:
        raise UnauthorizedError("Comma account sign-in required (HTTP 401). Run 'op auth'.", status)
      if status == 403:
        raise APIError("Access denied (HTTP 403). Check that your comma account has access.", status)

      try:
        resp_json = resp.json()
      except requests.exceptions.JSONDecodeError:
        if resp.ok:
          raise APIError(f"Comma API returned invalid JSON (HTTP {status}).", status) from None
        resp_json = None

      error = resp_json.get('error') if isinstance(resp_json, dict) else None
      if not resp.ok or error:
        detail = (resp_json.get('description') or error) if isinstance(resp_json, dict) else None
        raise APIError(f"Comma API HTTP {status}: {detail or resp.reason}", status)
      return resp_json

  def get(self, endpoint, **kwargs):
    return self.request('GET', endpoint, **kwargs)

  def post(self, endpoint, **kwargs):
    return self.request('POST', endpoint, **kwargs)

class APIError(Exception):
  def __init__(self, message: str, status_code: int | None = None):
    super().__init__(message)
    self.status_code = status_code

class UnauthorizedError(APIError):
  pass
