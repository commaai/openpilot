import requests

from selfdrive.version import version

def api_get(endpoint, method='GET', timeout=None, access_token=None, **params):
  backend = "https://api.commadotai.com/"

  headers = {}
  if access_token is not None:
    headers['Authorization'] = "JWT "+access_token

  headers['User-Agent'] = "openpilot-" + version

  return requests.request(method, backend+endpoint, timeout=timeout, headers = headers, params=params)

