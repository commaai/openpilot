#!/usr/bin/env python3
import hashlib
import http.server
import json
import os
import unittest
import random
import requests
import shutil
import socketserver
import tempfile
import multiprocessing
from pathlib import Path

from selfdrive.hardware.eon.neos import RECOVERY_DEV, NEOSUPDATE_DIR, get_fn, download_file, \
                                        verify_update_ready, download_neos_update

EON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(EON_DIR, "neos.json")

PORT = 8000

def server_thread(port):
  socketserver.TCPServer.allow_reuse_address = True
  httpd = socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler)
  httpd.serve_forever()


class TestNeosUpdater(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # generate a fake manifest
    cls.manifest = {}
    for i in ('ota', 'recovery'):
      with tempfile.NamedTemporaryFile(delete=False, dir=os.getcwd()) as f:
        dat = os.urandom(random.randint(1000, 100000))
        f.write(dat)
        cls.manifest[f"{i}_url"] = f"http://localhost:{PORT}/" + os.path.relpath(f.name)
        cls.manifest[F"{i}_hash"] = hashlib.sha256(dat).hexdigest()
        if i == "recovery":
          cls.manifest["recovery_len"] = len(dat)

    with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
      f.write(json.dumps(cls.manifest))
      cls.fake_manifest = f.name

  @classmethod
  def tearDownClass(cls):
    os.unlink(cls.fake_manifest)
    os.unlink(os.path.basename(cls.manifest['ota_url']))
    os.unlink(os.path.basename(cls.manifest['recovery_url']))

  def setUp(self):
    # server for update files
    self.server = multiprocessing.Process(target=server_thread, args=(PORT, ))
    self.server.start()

    # clean up
    if os.path.exists(NEOSUPDATE_DIR):
      shutil.rmtree(NEOSUPDATE_DIR)

  def tearDown(self):
    self.server.kill()
    self.server.join(1)

  def _corrupt_recovery(self):
    with open(RECOVERY_DEV, "wb") as f:
      f.write(b'\x00'*100)

  def test_manifest(self):
    with open(MANIFEST) as f:
      m = json.load(f)

    assert m['ota_hash'] in m['ota_url']
    assert m['recovery_hash'] in m['recovery_url']
    assert m['recovery_len'] > 0

    for url in (m['ota_url'], m['recovery_url']):
      r = requests.head(m['recovery_url'])
      r.raise_for_status()
      self.assertEqual(r.headers['Content-Type'], "application/octet-stream")
      if url == m['recovery_url']:
        self.assertEqual(int(r.headers['Content-Length']), m['recovery_len'])

  def test_download_hash_check(self):
    os.makedirs(NEOSUPDATE_DIR, exist_ok=True)
    Path(get_fn(self.manifest['ota_url'])).touch()
    with self.assertRaisesRegex(Exception, "failed hash check"):
      download_file(self.manifest['ota_url'], get_fn(self.manifest['ota_url']),
                    self.manifest['ota_hash']+'a', "system")

    # should've unlinked after the failed hash check, should succeed now
    download_file(self.manifest['ota_url'], get_fn(self.manifest['ota_url']),
                  self.manifest['ota_hash'], "system")

  # TODO: needs an http server that supports Content-Range
  #def test_download_resume(self):
  #  os.makedirs(NEOSUPDATE_DIR, exist_ok=True)
  #  with open(os.path.basename(self.manifest['ota_url']), "rb") as src, \
  #       open(get_fn(self.manifest['ota_url']), "wb") as dest:
  #    l = dest.write(src.read(8192))
  #    assert l == 8192
  #  download_file(self.manifest['ota_url'], get_fn(self.manifest['ota_url']),
  #                self.manifest['ota_hash'], "system")

  def test_download_no_internet(self):
    self.server.kill()
    os.makedirs(NEOSUPDATE_DIR, exist_ok=True)
    # fail, no internet
    with self.assertRaises(requests.exceptions.ConnectionError):
      download_file(self.manifest['ota_url'], get_fn(self.manifest['ota_url']),
                    self.manifest['ota_hash'], "system")

    # already cached, ensure we don't hit the server
    shutil.copyfile(os.path.basename(self.manifest['ota_url']), get_fn(self.manifest['ota_url']))
    download_file(self.manifest['ota_url'], get_fn(self.manifest['ota_url']),
                  self.manifest['ota_hash'], "system")


  def test_download_update(self):
    download_neos_update(self.fake_manifest)
    self.assertTrue(verify_update_ready(self.fake_manifest))

  def test_verify_update(self):
    # good state
    download_neos_update(self.fake_manifest)
    self.assertTrue(verify_update_ready(self.fake_manifest))

    # corrupt recovery
    self._corrupt_recovery()
    self.assertFalse(verify_update_ready(self.fake_manifest))

    # back to good state
    download_neos_update(self.fake_manifest)
    self.assertTrue(verify_update_ready(self.fake_manifest))

    # corrupt ota
    self._corrupt_recovery()
    with open(os.path.join(NEOSUPDATE_DIR, os.path.basename(self.manifest['ota_url'])), "ab") as f:
      f.write(b'\x00')
    self.assertFalse(verify_update_ready(self.fake_manifest))

if __name__ == "__main__":
  unittest.main()
