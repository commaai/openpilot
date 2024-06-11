#!/usr/bin/env python3
import argparse
import collections
import multiprocessing
import os

import requests
from tqdm import tqdm

import openpilot.system.hardware.tici.casync as casync


def get_chunk_download_size(chunk):
  sha = chunk.sha.hex()
  path = os.path.join(remote_url, sha[:4], sha + ".cacnk")
  if os.path.isfile(path):
    return os.path.getsize(path)
  else:
    r = requests.head(path, timeout=10)
    r.raise_for_status()
    return int(r.headers['content-length'])


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Compute overlap between two casync manifests')
  parser.add_argument('frm')
  parser.add_argument('to')
  args = parser.parse_args()

  frm = casync.parse_caibx(args.frm)
  to = casync.parse_caibx(args.to)
  remote_url = args.to.replace('.caibx', '')

  most_common = collections.Counter(t.sha for t in to).most_common(1)[0][0]

  frm_dict = casync.build_chunk_dict(frm)

  # Get content-length for each chunk
  with multiprocessing.Pool() as pool:
    szs = list(tqdm(pool.imap(get_chunk_download_size, to), total=len(to)))
  chunk_sizes = {t.sha: sz for (t, sz) in zip(to, szs, strict=True)}

  sources: dict[str, list[int]] = {
    'seed': [],
    'remote_uncompressed': [],
    'remote_compressed': [],
  }

  for chunk in to:
    # Assume most common chunk is the zero chunk
    if chunk.sha == most_common:
      continue

    if chunk.sha in frm_dict:
      sources['seed'].append(chunk.length)
    else:
      sources['remote_uncompressed'].append(chunk.length)
      sources['remote_compressed'].append(chunk_sizes[chunk.sha])

  print()
  print("Update statistics (excluding zeros)")
  print()
  print("Download only with no seed:")
  print(f"  Remote (uncompressed)\t\t{sum(sources['seed'] + sources['remote_uncompressed']) / 1000 / 1000:.2f} MB\tn = {len(to)}")
  print(f"  Remote (compressed download)\t{sum(chunk_sizes.values()) / 1000 / 1000:.2f} MB\tn = {len(to)}")
  print()
  print("Upgrade with seed partition:")
  print(f"  Seed   (uncompressed)\t\t{sum(sources['seed']) / 1000 / 1000:.2f} MB\t\t\t\tn = {len(sources['seed'])}")
  sz, n = sum(sources['remote_uncompressed']), len(sources['remote_uncompressed'])
  print(f"  Remote (uncompressed)\t\t{sz / 1000 / 1000:.2f} MB\t(avg {sz / 1000 / 1000 / n:4f} MB)\tn = {n}")
  sz, n = sum(sources['remote_compressed']), len(sources['remote_compressed'])
  print(f"  Remote (compressed download)\t{sz / 1000 / 1000:.2f} MB\t(avg {sz / 1000 / 1000 / n:4f} MB)\tn = {n}")
