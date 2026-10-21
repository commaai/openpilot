"""Hands the page the comma JWT the Python tools use (~/.comma/auth.json), so local use needs no sign-in."""
from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.ycpe.daemon.core import route


@route("GET", "/local/auth")
def local_auth(req):
  return {"token": get_token()}  # never logged or printed
