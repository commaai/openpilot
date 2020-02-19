#!/usr/bin/env python3

import json
import os
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, parse_qs
from common.file_helpers import mkdirs_exists_ok
from tools.lib.api import CommaApi, APIError

CONFIG_DIR = os.path.expanduser('~/.comma')
mkdirs_exists_ok(CONFIG_DIR)

def get_token():
  try:
    with open(os.path.join(CONFIG_DIR, 'auth.json')) as f:
      auth = json.load(f)
      return auth['access_token']
  except:
    raise MissingAuthConfigError('Login with tools/lib/auth.py')

def set_token(token):
  with open(os.path.join(CONFIG_DIR, 'auth.json'), 'w') as f:
    json.dump({'access_token': token}, f)

class MissingAuthConfigError(Exception):
  pass

class ClientRedirectServer(HTTPServer):
  query_params = {}

class ClientRedirectHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    if not self.path.startswith('/auth_redirect'):
      self.send_response(204)
      return

    query = self.path.split('?', 1)[-1]
    query = parse_qs(query, keep_blank_values=True)
    self.server.query_params = query

    self.send_response(200)
    self.send_header('Content-type', 'text/plain')
    self.end_headers()
    self.wfile.write(b'Return to the CLI to continue')

  def log_message(self, format, *args):
    pass  # this prevent http server from dumping messages to stdout

def auth_redirect_link(port):
  redirect_uri = f'http://localhost:{port}/auth_redirect'
  params = {
    'type': 'web_server',
    'client_id': '45471411055-ornt4svd2miog6dnopve7qtmh5mnu6id.apps.googleusercontent.com',
    'redirect_uri': redirect_uri,
    'response_type': 'code',
    'scope': 'https://www.googleapis.com/auth/userinfo.email',
    'prompt': 'select_account',
  }

  return (redirect_uri, 'https://accounts.google.com/o/oauth2/auth?' + urlencode(params))

if __name__ == '__main__':
  port = 9090
  redirect_uri, oauth_uri = auth_redirect_link(port)

  web_server = ClientRedirectServer(('localhost', port), ClientRedirectHandler)
  webbrowser.open(oauth_uri, new=2)

  while True:
    web_server.handle_request()
    if 'code' in web_server.query_params:
      code = web_server.query_params['code']
      break
    elif 'error' in web_server.query_params:
      print('Authentication Error: "%s". Description: "%s" ' % (
        web_server.query_params['error'],
        web_server.query_params.get('error_description')))
      break

  try:
    auth_resp = CommaApi().post('v2/auth/', data={'code': code, 'redirect_uri': redirect_uri})
    set_token(auth_resp['access_token'])
    print('Authenticated')
  except APIError as e:
    print(f'Login error: {e}')
