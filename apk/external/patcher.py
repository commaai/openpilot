#!/usr/bin/env python2.7

import os
import sys
import glob
import shutil
import urllib2
import hashlib
import subprocess


EXTERNAL_PATH = os.path.dirname(os.path.abspath(__file__))

if os.path.exists("/init.qcom.rc"):
  # android
  APKPATCH = os.path.join(EXTERNAL_PATH, 'tools/apkpatch_android')
  SIGNAPK = os.path.join(EXTERNAL_PATH, 'tools/signapk_android')
else:
  APKPATCH = os.path.join(EXTERNAL_PATH, 'tools/apkpatch')
  SIGNAPK = os.path.join(EXTERNAL_PATH, 'tools/signapk')

APKS = {
  'com.waze': {
    'src': 'https://apkcache.s3.amazonaws.com/com.waze_1021278.apk',
    'src_sha256': 'f00957e93e2389f9e30502ac54994b98ac769314b0963c263d4e8baa625ab0c2',
    'patch': 'com.waze.apkpatch',
    'out_sha256': '9ec8b0ea3c78c666342865b1bfb66e368a3f5c911df2ad12835206ec8b19f444'
  },
  'com.spotify.music': {
    'src': 'https://apkcache.s3.amazonaws.com/com.spotify.music_24382006.apk',
    'src_sha256': '0610fea68ee7ba5f8e4e0732ad429d729dd6cbb8bc21222c4c99db6cb09fbff4',
    'patch': 'com.spotify.music.apkpatch',
    'out_sha256': '5a3d6f478c7e40403a98ccc8906d7e0ae12b06543b41f5df52149dd09c647c11'
  },
}

def sha256_path(path):
  with open(path, 'rb') as f:
    return hashlib.sha256(f.read()).hexdigest()

def remove(path):
  try:
    os.remove(path)
  except OSError:
    pass

def process(download, patch):
  # clean up any junk apks
  for out_apk in glob.glob(os.path.join(EXTERNAL_PATH, 'out/*.apk')):
    app = os.path.basename(out_apk)[:-4]
    if app not in APKS:
      print "remove junk", out_apk
      remove(out_apk)

  complete = True
  for k,v in APKS.iteritems():
    apk_path = os.path.join(EXTERNAL_PATH, 'out', k+'.apk')
    print "checking", apk_path
    if os.path.exists(apk_path) and sha256_path(apk_path) == v['out_sha256']:
      # nothing to do
      continue

    complete = False

    remove(apk_path)

    src_path = os.path.join(EXTERNAL_PATH, 'src', v['src_sha256'])
    if not os.path.exists(src_path) or sha256_path(src_path) != v['src_sha256']:
      if not download:
        continue

      print "downloading", v['src'], "to", src_path
      # download it
      resp = urllib2.urlopen(v['src'])
      data = resp.read()
      with open(src_path, 'wb') as src_f:
        src_f.write(data)

      if sha256_path(src_path) != v['src_sha256']:
        print "download was corrupted..."
        continue

    if not patch:
      continue

    # ignoring lots of TOCTTOU here...

    apk_temp = "/tmp/"+k+".patched"
    remove(apk_temp)
    apk_temp2 = "/tmp/"+k+".signed"
    remove(apk_temp2)

    try:
      print "patching", v['patch']
      subprocess.check_call([APKPATCH, 'apply', src_path, apk_temp, os.path.join(EXTERNAL_PATH, v['patch'])])
      print "signing", apk_temp
      subprocess.check_call([SIGNAPK,
        os.path.join(EXTERNAL_PATH, 'tools/certificate.pem'), os.path.join(EXTERNAL_PATH, 'tools/key.pk8'),
        apk_temp, apk_temp2])

      out_sha256 = sha256_path(apk_temp2) if os.path.exists(apk_temp2) else None

      if out_sha256 == v['out_sha256']:
        print "done", apk_path
        shutil.move(apk_temp2, apk_path)
      else:
        print "patch was corrupted", apk_temp2, out_sha256
    finally:
      remove(apk_temp)
      remove(apk_temp2)

  return complete

if __name__ == "__main__":
  ret = True
  if len(sys.argv) == 2 and sys.argv[1] == "download":
    ret = process(True, False)
  elif len(sys.argv) == 2 and sys.argv[1] == "patch":
    ret = process(False, True)
  else:
    ret = process(True, True)
  sys.exit(0 if ret else 1)
