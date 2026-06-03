import jwt
import requests
from datetime import datetime, timedelta, UTC
from pathlib import Path
from openpilot.system.hardware.hw import Paths
from openpilot.system.version import get_version

from openpilot.common.params import Params

API_HOST = Params().get("APIHost", return_default=True)

FALLBACK_IDENTITY_DIR = Path("/data/persist/comma")


class Api:
  def __init__(self, dongle_id):
    self.dongle_id = dongle_id
    self.jwt_algorithm, self.private_key, _ = get_key_pair()

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, **params):
    return api_get(endpoint, method=method, timeout=timeout, access_token=access_token, **params)

  def get_token(self, payload_extra=None, expiry_hours=1):
    now = datetime.now(UTC).replace(tzinfo=None)
    payload = {
      'identity': self.dongle_id,
      'nbf': now,
      'iat': now,
      'exp': now + timedelta(hours=expiry_hours)
    }
    if payload_extra is not None:
      payload.update(payload_extra)
    token = jwt.encode(payload, self.private_key, algorithm=self.jwt_algorithm)
    if isinstance(token, bytes):
      token = token.decode('utf8')
    return token


def api_get(endpoint, method='GET', timeout=None, access_token=None, session=None, **params):
  headers = {}
  if access_token is not None:
    headers['Authorization'] = "JWT " + access_token

  headers['User-Agent'] = "openpilot-" + get_version()

  # TODO: add session to Api
  req = requests if session is None else session
  return req.request(method, API_HOST + "/" + endpoint, timeout=timeout, headers=headers, params=params)


def get_key_pair() -> tuple[str, str, str] | tuple[None, None, None]:
  for identity_dir in (Path(Paths.persist_root()) / "comma", FALLBACK_IDENTITY_DIR):
    private_key_path = identity_dir / "id_ed25519"
    public_key_path = identity_dir / "id_ed25519.pub"
    if private_key_path.is_file() and public_key_path.is_file():
      return "EdDSA", private_key_path.read_text(), public_key_path.read_text()
  return None, None, None
