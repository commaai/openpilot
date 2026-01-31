import json, multiprocessing, functools
from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.helpers import tqdm, getenv

raid_root = Path(getenv("RAID_ROOT", "/raid"))

def fetch_file(item):
  path, info = item
  h, size = info["hash"], info["size"]

  path = raid_root / Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  try:
    pt = Tensor(bytes.fromhex(h), device="CPU").fs_load(size).to(f"disk:{path.as_posix()}").realize()
  except Exception as e:
    print(f"error fetching {path}, {h}, {size}: {e}")
    raise

  pt.uop.buffer.deallocate()

def fetch_mapping(h, l):
  mapping_tensor = Tensor(bytes.fromhex(h)).fs_load(l).realize()
  mapping = mapping_tensor.data().tobytes().decode()
  mapping = json.loads(mapping)
  mapped_files = mapping.items()
  return list(mapped_files)

if __name__ == "__main__":
  h, l = getenv("HASH", "d734f5e3be9f1e9d863bfaa4fc6c1ef2"), getenv("LENGTH", 175866113)

  with multiprocessing.Pool(processes=1) as pool:
    mapped_files = pool.apply(functools.partial(fetch_mapping, h, l))

  print(f"fetched mapping for {len(mapped_files)} files")

  with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    for _ in tqdm(pool.imap_unordered(fetch_file, mapped_files), total=len(mapped_files)):
      pass
