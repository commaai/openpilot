#!/usr/bin/env python3

import requests
from common.params import Params
import sys


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f"{sys.argv[0]} <github username>")
    exit(1)

  username = sys.argv[1]
  keys = requests.get(f"https://github.com/{username}.keys")

  if keys.status_code == 200:
    Params().put("GithubSshKeys", keys.text)
    Params().put("GithubUsername", username)
    print("Setup ssh keys successfully")
  else:
    print("Error getting public keys from github")
