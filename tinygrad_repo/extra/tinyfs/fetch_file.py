from tinygrad.tensor import Tensor
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--hash", type=str, required=True, help="file hash to fetch")
  parser.add_argument("--len", type=int, required=True, help="file length to fetch")
  parser.add_argument("--dest", type=str, required=True, help="destination path to save the file")
  args = parser.parse_args()

  Tensor(bytes.fromhex(args.hash), device="CPU").fs_load(args.len).to(f"disk:{args.dest}").realize()
