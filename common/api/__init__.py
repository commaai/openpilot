from openpilot.common.api.comma_connect import CommaConnectApi
from sunnypilot.sunnylink.api import SunnylinkApi


class Api:
  def __init__(self, dongle_id, use_sunnylink=False):
    if use_sunnylink:
      self.service = SunnylinkApi(dongle_id)
    else:
      self.service = CommaConnectApi(dongle_id)

  def request(self, method, endpoint, **params):
    return self.service.request(method, endpoint, **params)

  def get(self, *args, **kwargs):
    return self.service.get(*args, **kwargs)

  def post(self, *args, **kwargs):
    return self.service.post(*args, **kwargs)

  def get_token(self, expiry_hours=1):
    return self.service.get_token(expiry_hours)


def api_get(endpoint, method='GET', timeout=None, access_token=None, use_sunnylink=False, **params):
  return SunnylinkApi(None).api_get(endpoint, method, timeout, access_token, **params) if use_sunnylink \
    else CommaConnectApi(None).api_get(endpoint, method, timeout, access_token, **params)
