import requests

def api_get(endpoint, method='GET', timeout=None, access_token=None, **params):
  backend = "https://api.commadotai.com/"

  params['_version'] = "OPENPILOTv0.3"

  headers = {}
  if access_token is not None:
    headers['Authorization'] = "JWT "+access_token

  return requests.request(method, backend+endpoint, timeout=timeout, headers = headers, params=params)

