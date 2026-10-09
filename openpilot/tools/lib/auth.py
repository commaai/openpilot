#!/usr/bin/env python3
"""
Usage::

  usage: auth.py [-h] [{google,apple,github,jwt}] [jwt]

  Login to your comma account

  positional arguments:
    {google,apple,github,jwt}
    jwt

  optional arguments:
    -h, --help            show this help message and exit


Examples::

  ./auth.py  # Log in with google account
  ./auth.py github  # Log in with GitHub Account
  ./auth.py jwt ey......hw  # Log in with a JWT from https://jwt.comma.ai, for use in CI
"""

import argparse
import json
import sys
import subprocess
import pprint
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from openpilot.tools.lib.api import CommaApi, UnauthorizedError
from openpilot.tools.lib.auth_config import set_token, get_token

class ClientRedirectServer(ThreadingHTTPServer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.query_params: dict[str, Any] = {}
    self.result_lock = threading.Lock()

  def get_request(self):
    request, address = super().get_request()
    request.settimeout(1)  # Bound incomplete requests, including browser preconnections.
    return request, address


class ClientRedirectHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    if urlsplit(self.path).path not in ('/auth', '/auth/'):
      self.send_response(204)
      self.end_headers()
      return

    query_parsed = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
    with self.server.result_lock:
      if not self.server.query_params and ('code' in query_parsed or 'error' in query_parsed):
        self.server.query_params = query_parsed

    self.send_response(200)
    self.send_header('Content-type', 'text/plain')
    self.end_headers()
    try:
      self.wfile.write(b'Sign-in received. You can close this tab and return to Cabana or your terminal.')
    except ConnectionError:
      pass  # A closing browser tab must not discard the received callback.

  def log_message(self, format: str, *args: object) -> None:  # noqa: A002  # stdlib override
    pass  # this prevent http server from dumping messages to stdout


def auth_redirect_link(method, port):
  provider_id = {
    'google': 'g',
    'apple': 'a',
    'github': 'h',
  }[method]

  params = {
    'redirect_uri': f"https://api.comma.ai/v2/auth/{provider_id}/redirect/",
    'state': f'service,localhost:{port}',
  }

  if method == 'google':
    params.update({
      'type': 'web_server',
      'client_id': '45471411055-ornt4svd2miog6dnopve7qtmh5mnu6id.apps.googleusercontent.com',
      'response_type': 'code',
      'scope': 'https://www.googleapis.com/auth/userinfo.email',
      'prompt': 'select_account',
    })
    return 'https://accounts.google.com/o/oauth2/auth?' + urlencode(params)
  elif method == 'github':
    params.update({
      'client_id': '28c4ecb54bb7272cb5a4',
      'scope': 'read:user',
    })
    return 'https://github.com/login/oauth/authorize?' + urlencode(params)
  elif method == 'apple':
    params.update({
      'client_id': 'ai.comma.login',
      'response_type': 'code',
      'response_mode': 'form_post',
      'scope': 'name email',
    })
    return 'https://appleid.apple.com/auth/authorize?' + urlencode(params)
  else:
    raise NotImplementedError(f"no redirect implemented for method {method}")


def login(method, timeout=180):
  """Sign in through a browser and save the token, returning a success/error status."""
  try:
    with ClientRedirectServer(('localhost', 0), ClientRedirectHandler) as server:
      url = auth_redirect_link(method, server.server_port)
      print(f'To sign in, use your browser and navigate to {url}', file=sys.stderr)
      browser = subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', url],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      deadline = time.monotonic() + timeout
      while time.monotonic() < deadline:
        server.timeout = min(0.1, max(0, deadline - time.monotonic()))
        server.handle_request()
        params = server.query_params
        if 'error' in params:
          return {"error": "Sign-in was declined. Choose a provider to try again."}
        if 'code' in params:
          provider = {'google': 'g', 'apple': 'a', 'github': 'h'}[method]
          if len(params['code']) != 1 or not params['code'][0].strip() or params.get('provider') != [provider]:
            return {"error": "Invalid sign-in response. Please try again."}
          response = CommaApi().post('v2/auth/', data={'code': params['code'], 'provider': params['provider']}, timeout=30)
          token = response.get('access_token')
          if not isinstance(token, str) or not token.strip():
            return {"error": "Sign-in did not return an access token. Please try again."}
          CommaApi(token).get('v1/me', timeout=30)
          set_token(token)
          return {"success": True}
        if browser.poll() not in (None, 0):
          return {"error": "Could not open your browser. Check your default browser and try again."}
      return {"error": "Sign-in timed out. Choose a provider to try again."}
  except Exception:
    return {"error": "Could not complete sign-in. Check your connection and try again."}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Login to your comma account')
  parser.add_argument('method', default='google', const='google', nargs='?', choices=['google', 'apple', 'github', 'jwt'])
  parser.add_argument('jwt', nargs='?')
  parser.add_argument('--json', action='store_true', help='Return browser sign-in status as JSON')

  args = parser.parse_args()
  if args.json:
    if args.method == 'jwt':
      parser.error('--json requires a browser sign-in provider')
    print(json.dumps(login(args.method)))
    sys.exit(0)

  if args.method == 'jwt':
    if args.jwt is None:
      print("method JWT selected, but no JWT was provided")
      exit(1)

    set_token(args.jwt)
  else:
    result = login(args.method)
    if "error" in result:
      print(result["error"], file=sys.stderr)
      sys.exit(1)

  try:
    me = CommaApi(token=get_token()).get('/v1/me')
    print("Authenticated!")
    pprint.pprint(me)
  except UnauthorizedError:
    print("Got invalid JWT")
    exit(1)
