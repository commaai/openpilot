import os

from openpilot.common.api.base import BaseApi

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')


class CommaConnectApi(BaseApi):
  def __init__(self, dongle_id):
    super().__init__(dongle_id, API_HOST)
    self.user_agent = "openpilot-"
