from openpilot.common.api.comma_connect import CommaConnectApi


class Api:
  def __init__(self, dongle_id):
    self.service = CommaConnectApi(dongle_id)

  def request(self, method, endpoint, **params):
    return self.service.request(method, endpoint, **params)

  def get(self, *args, **kwargs):
    return self.service.get(*args, **kwargs)

  def post(self, *args, **kwargs):
    return self.service.post(*args, **kwargs)

  def get_token(self, expiry_hours=1):
    return self.service.get_token(expiry_hours)


def api_get(endpoint, method='GET', timeout=None, access_token=None, **params):
  return CommaConnectApi(None).api_get(endpoint, method, timeout, access_token, **params)
