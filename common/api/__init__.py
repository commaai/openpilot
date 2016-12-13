import requests

def api_get(endpoint, method='GET', timeout=None, **params):
  backend = "https://api.commadotai.com/"

  params['_version'] = "OPENPILOTv0.2"

  return requests.request(method, backend+endpoint, timeout=timeout, params=params)
