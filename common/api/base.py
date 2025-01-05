import jwt
import requests
import unicodedata
from datetime import datetime, timedelta, UTC
from openpilot.system.hardware.hw import Paths
from openpilot.system.version import get_version


class BaseApi:
  def __init__(self, dongle_id, api_host, user_agent="openpilot-"):
    self.dongle_id = dongle_id
    self.api_host = api_host
    self.user_agent = user_agent
    with open(f'{Paths.persist_root()}/comma/id_rsa') as f:
      self.private_key = f.read()

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, **params):
    return self.api_get(endpoint, method=method, timeout=timeout, access_token=access_token, **params)

  def _get_token(self, expiry_hours=1, **extra_payload):
    now = datetime.now(UTC).replace(tzinfo=None)
    payload = {
      'identity': self.dongle_id,
      'nbf': now,
      'iat': now,
      'exp': now + timedelta(hours=expiry_hours),
      **extra_payload
    }
    token = jwt.encode(payload, self.private_key, algorithm='RS256')
    if isinstance(token, bytes):
      token = token.decode('utf8')
    return token

  def get_token(self, expiry_hours=1):
    return self._get_token(expiry_hours)

  def remove_non_ascii_chars(self, text):
    normalized_text = unicodedata.normalize('NFD', text)
    ascii_encoded_text = normalized_text.encode('ascii', 'ignore')
    return ascii_encoded_text.decode()

  def api_get(self, endpoint, method='GET', timeout=None, access_token=None, **params):
    headers = {}
    if access_token is not None:
      headers['Authorization'] = "JWT " + access_token

    version = self.remove_non_ascii_chars(get_version())
    headers['User-Agent'] = self.user_agent + version

    return requests.request(method, f"{self.api_host}/{endpoint}", timeout=timeout, headers=headers, params=params)
