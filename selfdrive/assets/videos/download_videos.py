#!/usr/bin/env python3
import sys
import urllib.request
import requests
import os
import shutil
from common.basedir import BASEDIR
import traceback

AZ_BASEDIR = "https://commadataci.blob.core.windows.net/cesdemo"
MANIFEST = f"{AZ_BASEDIR}/manifest.txt"
VIDEOS_PATH = f"{BASEDIR}/selfdrive/assets/videos/out"


print("downloading videos")
os.makedirs(VIDEOS_PATH, exist_ok=True)

manifest = requests.get(MANIFEST)
if manifest.status_code != 200:
  sys.exit(1)

manifest = manifest.content.decode('utf-8').splitlines()
manifest = [f.strip() for f in manifest]
manifest = [f.replace('out/', '') for f in manifest if
            len(f) and not f.startswith('#')]

for f in manifest:
  print(f'Downloading {f}')
  file_path = os.path.join(VIDEOS_PATH, f)
  try:
    urllib.request.urlretrieve(os.path.join(AZ_BASEDIR, "out", f), file_path)
  except Exception:
    print(traceback.format_exc())

for f in os.listdir(VIDEOS_PATH):
  if f not in manifest:
    print(f'Removing {f}')
    os.remove(os.path.join(VIDEOS_PATH, f), )
