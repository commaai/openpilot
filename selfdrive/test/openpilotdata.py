import json
import os
import pathlib
import requests


BASEDIR = pathlib.Path(__file__).parent
SEGMENTS_DIR = BASEDIR / "segments"

SEGMENTS_DATA_FILE = BASEDIR / "segments.json"

def load_segments_data():
  with open(SEGMENTS_DATA_FILE) as f:
    return json.load(f)

def save_segments_data(data):
  with open(SEGMENTS_DATA_FILE, "w") as f:
    json.dump(data, f, indent=2)


OPENPILOT_DATA_REPO = os.environ.get("OPENPILOT_DATA_REPO", "jnewb1/openpilot-data")
OPENPILOT_DATA_BRANCH = os.environ.get("OPENPILOT_DATA_BRANCH", "master")

LFS_INSTANCE = os.environ.get("GITLAB_LFS_INSTANCE", "https://gitlab.com/jnewberry0502/openpilot-data")

# Helpers related to interfacing with the openpilot-data repository, which contains a collection of public segments for users to perform validation on.

def parse_lfs_pointer(text):
  header, lfs_version = text.splitlines()[0].split(" ")
  assert header == "version"
  assert lfs_version == "https://git-lfs.github.com/spec/v1"

  header, oid_raw = text.splitlines()[1].split(" ")
  assert header == "oid"
  header, oid = oid_raw.split(":")
  assert header == "sha256"

  header, size = text.splitlines()[2].split(" ")
  assert header == "size"

  return oid, size

def get_lfs_file_url(oid, size):
  data = {
    "operation": "download",
    "transfers": [ "basic" ],
    "objects": [
      {
        "oid": oid,
        "size": int(size)
      }
    ],
    "hash_algo": "sha256"
  }

  headers = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json"
  }

  response = requests.post(f"{LFS_INSTANCE}.git/info/lfs/objects/batch", json=data, headers=headers)

  assert response.ok

  obj = response.json()["objects"][0]

  assert "error" not in obj, obj

  return obj["actions"]["download"]["href"]


def get_github_url(path):
  return f"https://raw.githubusercontent.com/{OPENPILOT_DATA_REPO}/{OPENPILOT_DATA_BRANCH}/{path}"

def get_github_route_url(route, seg):
  return get_github_url(f"segments/{route}--{seg}.bz2")

def get_url(route, seg):
  response = requests.head(get_github_route_url(route, seg))

  if "text/plain" in response.headers.get("content-type"):
    # This is an LFS pointer, so download the raw data from lfs

    response = requests.get(get_github_route_url(route, seg))

    oid, size = parse_lfs_pointer(response.text)

    url = get_lfs_file_url(oid, size)
  else:
    # File has not been uploaded to LFS yet
    # (either we are on a fork where the data hasn't been pushed to LFS yet, or the CI job to push hasn't finished)
    url = get_github_route_url(route, seg)

  return url
