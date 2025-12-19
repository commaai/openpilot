import os
import requests


# Forks with additional car support can fork the commaCarSegments repo on huggingface or host the LFS files themselves
COMMA_CAR_SEGMENTS_REPO = os.environ.get("COMMA_CAR_SEGMENTS_REPO", "https://huggingface.co/datasets/commaai/commaCarSegments")
COMMA_CAR_SEGMENTS_BRANCH = os.environ.get("COMMA_CAR_SEGMENTS_BRANCH", "main")
COMMA_CAR_SEGMENTS_LFS_INSTANCE = os.environ.get("COMMA_CAR_SEGMENTS_LFS_INSTANCE", COMMA_CAR_SEGMENTS_REPO)

def get_comma_car_segments_database():
  from opendbc.car.fingerprints import MIGRATION

  database = requests.get(get_repo_raw_url("database.json")).json()

  ret = {}
  for platform in database:
    # TODO: remove this when commaCarSegments is updated to remove selector
    ret[MIGRATION.get(platform, platform)] = [s.rstrip('/s') for s in database[platform]]

  return ret


# Helpers related to interfacing with the commaCarSegments repository, which contains a collection of public segments for users to perform validation on.

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

  response = requests.post(f"{COMMA_CAR_SEGMENTS_LFS_INSTANCE}.git/info/lfs/objects/batch", json=data, headers=headers)

  assert response.ok

  obj = response.json()["objects"][0]

  assert "error" not in obj, obj

  return obj["actions"]["download"]["href"]

def get_repo_raw_url(path):
  if "huggingface" in COMMA_CAR_SEGMENTS_REPO:
    return f"{COMMA_CAR_SEGMENTS_REPO}/raw/{COMMA_CAR_SEGMENTS_BRANCH}/{path}"

def get_repo_url(path):
  # Automatically switch to LFS if we are requesting a file that is stored in LFS

  response = requests.head(get_repo_raw_url(path))

  if "text/plain" in response.headers.get("content-type"):
    # This is an LFS pointer, so download the raw data from lfs
    response = requests.get(get_repo_raw_url(path))
    assert response.status_code == 200
    oid, size = parse_lfs_pointer(response.text)

    return get_lfs_file_url(oid, size)
  else:
    # File has not been uploaded to LFS yet
    # (either we are on a fork where the data hasn't been pushed to LFS yet, or the CI job to push hasn't finished)
    return get_repo_raw_url(path)


def get_url(route, segment, file="rlog.zst"):
  return get_repo_url(f"segments/{route.replace('|', '/')}/{segment}/{file}")
