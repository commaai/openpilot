#!/usr/bin/env python3
import requests
import json
from .automated.helpers import _connect_wifi  # pylint: disable=import-error
from panda import Panda
from nose.tools import assert_equal

if __name__ == "__main__":
  print("Fetching latest firmware from github.com/commaai/panda-artifacts")
  r = requests.get("https://raw.githubusercontent.com/commaai/panda-artifacts/master/latest.json")
  latest_version = json.loads(r.text)['version']

  for p in Panda.list():
    dongle_id, pw = Panda(p).get_serial()
    print(dongle_id, pw)
    assert(dongle_id.isalnum())
    _connect_wifi(dongle_id, pw)

    r = requests.get("http://192.168.0.10/")
    print(r.text)
    wifi_dongle_id = r.text.split("ssid: panda-")[1].split("<br/>")[0]
    st_version = r.text.split("st version:")[1].strip().split("<br/>")[0]
    esp_version = r.text.split("esp version:")[1].strip().split("<br/>")[0]

    assert_equal(str(dongle_id), wifi_dongle_id)
    assert_equal(latest_version, st_version)
    assert_equal(latest_version, esp_version)
