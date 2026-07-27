#!/usr/bin/env python3

from openpilot.common import http
from openpilot.common.params import Params
import sys


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f"{sys.argv[0]} <github username>")
    exit(1)

  username = sys.argv[1]
  keys = http.get(f"https://github.com/{username}.keys", timeout=10)

  if keys.status_code == 200:
    params = Params()
    params.put_bool("SshEnabled", True, block=True)
    params.put("GithubSshKeys", keys.text, block=True)
    params.put("GithubUsername", username, block=True)
    print("Set up ssh keys successfully")
  else:
    print("Error getting public keys from github")
