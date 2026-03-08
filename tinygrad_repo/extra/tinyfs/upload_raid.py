from pathlib import Path
import multiprocessing, json

from tinygrad.tensor import Tensor
from tinygrad.helpers import tqdm

raid_root = Path("/raid")

def upload_file(path: Path):
  pt = Tensor(path).realize()
  h = pt.fs_store().realize()
  pt.uop.realized.deallocate()
  return h.data().hex(), path, pt.nbytes()

if __name__ == "__main__":
  raid_files = sorted([p for p in raid_root.rglob("*") if p.is_file()])
  print(f"found {len(raid_files)} files in /raid")

  mapping = {}
  with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    for h, p, s in tqdm(pool.imap_unordered(upload_file, raid_files), total=len(raid_files)):
      mapping[p.relative_to(raid_root).as_posix()] = {"hash": h, "size": s}

  # sort the mapping by key
  mapping = dict(sorted(mapping.items()))

  mapping = json.dumps(mapping).encode()
  mapping_tensor = Tensor(mapping, device="CPU")
  h = mapping_tensor.fs_store().realize()

  print(f"final hash: {h.data().hex()}, size: {len(mapping)}")
