#!/usr/bin/env python3
import os
import glob
import onnx

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
MASTER_PATH = os.getenv("MASTER_PATH", BASEDIR)
MODEL_PATH = "/selfdrive/modeld/models/"

def get_checkpoint(f):
  model = onnx.load(f)
  metadata = {prop.key: prop.value for prop in model.metadata_props}
  return metadata['model_checkpoint'].split('/')[0]

if __name__ == "__main__":
  print("| | master | PR branch |")
  print("|-| -----  | --------- |")

  for f in glob.glob(BASEDIR + MODEL_PATH + "/*.onnx"):
    # TODO: add checkpoint to DM
    if "dmonitoring" in f:
      continue

    fn = os.path.basename(f)
    master = get_checkpoint(MASTER_PATH + MODEL_PATH + fn)
    pr = get_checkpoint(BASEDIR + MODEL_PATH + fn)
    print(
      "|", fn, "|",
      f"[{master}](https://reporter.comma.life/experiment/{master})", "|",
      f"[{pr}](https://reporter.comma.life/experiment/{pr})", "|"
    )
