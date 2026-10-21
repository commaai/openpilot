#!/usr/bin/env python3
import json
import os
import pickle
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path

from openpilot.tools.ycpe.daemon import replay, replay_runner
from openpilot.tools.ycpe.daemon.replay import Replay, environment, missing

# stands in for replay_runner.py: the same events, then whatever argv's "fail" asks for
FAKE_RUNNER = textwrap.dedent("""
  import json, os, sys, time
  args = json.loads(sys.argv[1])
  json.dump(args, open(os.path.join(args["out"], "args.json"), "w"))
  ev = lambda e, **d: print(json.dumps({"event": e, **d}), flush=True)
  for i, (n, path) in enumerate(args["segs"]):
    ev("load", seg=n, i=i, count=len(args["segs"]))
  ev("replay", procs=args["procs"], messages=100)
  for i, (n, _) in enumerate(args["segs"]):
    warm = args["warm"] and i == 0
    ev("progress", done=50 * i + 25, total=100, seg=n, warm=warm, elapsed=0.1)
    if os.environ.get("FAKE_HANG"):
      time.sleep(60)
    if warm:
      continue
    os.makedirs(os.path.join(args["out"], str(n)))
    open(os.path.join(args["out"], str(n), "rlog.zst"), "wb").write(b"x")
    ev("segment", seg=n, messages=1)
  if os.environ.get("FAKE_FAIL"):
    print("Traceback (most recent call last):", file=sys.stderr)
    print("ValueError: Fingerprinting failed", file=sys.stderr)
    sys.exit(1)
  ev("diffing")
  json.dump({"signals": [], "diverged": 0, "compared": 0, "services": {}}, open(os.path.join(args["out"], "diff.json"), "w"))
  ev("done", seconds=0.1, diverged=0)
""")


def checkout(root: Path, nested=True, built=()) -> Path:
  pkg = root / "openpilot" if nested else root
  (pkg / "selfdrive").mkdir(parents=True)
  for f in built:
    (root / f).parent.mkdir(parents=True, exist_ok=True)
    (root / f).touch()
  return root


BUILT = ["msgq_repo/msgq/ipc_pyx.so", "msgq_repo/msgq/visionipc/visionipc_pyx.cpython-312-x86_64-linux-gnu.so", "openpilot/common/libparams_c.so"]


class TestCheckout(unittest.TestCase):
  def setUp(self):
    self.dir = Path(tempfile.mkdtemp())

  def test_missing_names_the_files(self):
    bare = checkout(self.dir / "bare")
    gaps = missing(bare, ["controlsd", "plannerd"])
    self.assertEqual(gaps["all"], ["msgq (msgq_repo/msgq/ipc_pyx.so)", "visionipc (msgq_repo/msgq/visionipc/visionipc_pyx.so)",
                                   "params (openpilot/common/libparams_c.so)"])
    self.assertNotIn("controlsd", gaps)
    self.assertIn("longitudinal_mpc_lib", gaps["plannerd"][0])
    built = checkout(self.dir / "built", built=BUILT)
    self.assertEqual(missing(built, ["controlsd", "radard"]), {})
    self.assertEqual(missing(built, ["modeld"])["modeld"], ["tinygrad (tinygrad_repo/tinygrad/__init__.py)",
                     "driving model (openpilot/selfdrive/modeld/models/driving_tinygrad.pkl)",
                     "driving model's warp (openpilot/selfdrive/modeld/models/driving_warp_<size>_tinygrad.pkl)"])

  def test_models_need_their_device(self):
    models = "openpilot/selfdrive/modeld/models/"
    built = checkout(self.dir / "built", built=[*BUILT, "tinygrad_repo/tinygrad/__init__.py", models + "dm_warp_1344x760_tinygrad.pkl"])
    (built / models / "dmonitoring_model_tinygrad.pkl").write_bytes(pickle.dumps({"device": "CUDA", "renderer": ["NV", "CUDAX"]}))
    self.assertEqual(replay.model_devices(str(built / models / "dmonitoring_model_tinygrad.pkl"), 0), ("CUDA", "NV"))
    devices = replay.DEVICES
    try:
      replay.DEVICES = {**devices, "CUDA": ("an NVIDIA GPU with libcuda", lambda: False), "NV": ("an NVIDIA GPU", lambda: True)}
      self.assertEqual(missing(built, ["dmonitoringmodeld"]),
                       {"dmonitoringmodeld": ["an NVIDIA GPU with libcuda (dmonitoring_model_tinygrad.pkl is compiled for CUDA)"]})
      replay.DEVICES = {**devices, "CUDA": ("", lambda: True), "NV": ("", lambda: True)}
      self.assertEqual(missing(built, ["dmonitoringmodeld"]), {})
    finally:
      replay.DEVICES = devices

  def test_flat_checkout_gets_a_shim(self):
    flat = checkout(self.dir / "flat", nested=False, built=["common/params_pyx.so", *BUILT[:2]])
    (flat / "msgq_repo").mkdir(exist_ok=True)
    self.assertEqual(missing(flat, ["controlsd"]), {})
    _, env = environment(flat, self.dir / "shims")
    paths = env["PYTHONPATH"].split(os.pathsep)
    self.assertEqual(os.path.realpath(Path(paths[0]) / "openpilot"), str(flat.resolve()))
    self.assertEqual(paths[1:], [str(flat / "msgq_repo")])
    nested = checkout(self.dir / "nested")
    python, env = environment(nested, self.dir / "shims")
    self.assertEqual(env["PYTHONPATH"], str(nested))
    self.assertEqual(python, sys.executable)


class TestReplay(unittest.TestCase):
  def setUp(self):
    self.dir = Path(tempfile.mkdtemp())
    self.repo = checkout(self.dir / "repo", built=BUILT)
    runner = self.dir / "runner.py"
    runner.write_text(FAKE_RUNNER)
    self._runner, replay.RUNNER = replay.RUNNER, runner
    self.reports = []

  def tearDown(self):
    replay.RUNNER = self._runner
    for k in ("FAKE_FAIL", "FAKE_HANG"):
      os.environ.pop(k, None)

  def start(self, procs=("controlsd",)):
    return Replay(self.dir / "out", "a" * 16 + "|r", [3, 4], list(procs), self.repo)

  def report(self, done, total, text, **meta):
    self.reports.append((done, total, text, meta))

  def test_runs_and_writes_a_manifest(self):
    r = self.start()
    m = r.run(lambda route, seg, file: f"/logs/{seg}", self.report)
    self.assertEqual((m["state"], m["segs"], m["warm"], m["written"]), ("done", [3, 4], 2, [3, 4]))
    self.assertEqual(json.loads((r.dir / "manifest.json").read_text())["state"], "done")
    self.assertFalse((r.dir / "2").exists())
    texts = [t for _, _, t, _ in self.reports if t]
    self.assertIn("downloading segment 2 to warm up on", texts)
    self.assertIn("downloading segment 3", texts)
    self.assertTrue(any(t.startswith("warming up on segment 2 · ") for t in texts))
    self.assertIn("replaying controlsd", texts)
    self.assertTrue(any(t.startswith("segment 3 · 0/2 written") for t in texts))
    self.assertEqual(self.reports[-1][3].get("phase"), "diff")

  def test_vision_processes_get_the_videos(self):
    built = [*BUILT, "tinygrad_repo/tinygrad/__init__.py", "openpilot/selfdrive/modeld/models/dmonitoring_model_tinygrad.pkl",
             "openpilot/selfdrive/modeld/models/dm_warp_1928x1208_tinygrad.pkl"]
    r = Replay(self.dir / "out", "a" * 16 + "|r", [3], ["dmonitoringmodeld"], checkout(self.dir / "vision", built=built))
    m = r.run(lambda route, seg, file: f"/logs/{seg}/{file}", self.report)
    self.assertEqual(m["written"], [3])
    args = json.loads((r.dir / "args.json").read_text())
    self.assertEqual(args["videos"], {"2": {"cabinCameraState": "/logs/2/dcamera.hevc"}, "3": {"cabinCameraState": "/logs/3/dcamera.hevc"}})
    def fetch(route, seg, file):
      if file == "dcamera.hevc" and seg == 3:
        raise FileNotFoundError(file)
      return f"/logs/{seg}/{file}"
    with self.assertRaises(RuntimeError) as e:
      Replay(self.dir / "out", "a" * 16 + "|r", [3], ["dmonitoringmodeld"], r.repo).run(fetch, self.report)
    self.assertEqual(str(e.exception), "segment 3 has no dcamera.hevc uploaded, and dmonitoringmodeld needs its frames")

  def test_stops_at_the_end_of_the_route(self):
    def fetch(route, seg, file):
      if seg > 3:
        raise FileNotFoundError(seg)
      return f"/logs/{seg}"
    m = self.start().run(fetch, self.report)
    self.assertEqual((m["segs"], m["warm"], m["written"]), ([3], 2, [3]))

  def test_cold_start_without_a_segment_before(self):
    def fetch(route, seg, file):
      if seg == 2:
        raise FileNotFoundError(seg)
      return f"/logs/{seg}"
    m = self.start().run(fetch, self.report)
    self.assertEqual((m["segs"], m["warm"], m["written"]), ([3, 4], None, [3, 4]))
    m = Replay(self.dir / "out", "a" * 16 + "|r", [0], ["controlsd"], self.repo).run(lambda route, seg, file: f"/logs/{seg}", self.report)
    self.assertEqual((m["warm"], m["written"]), (None, [0]))

  def test_failure_carries_the_error_tail(self):
    os.environ["FAKE_FAIL"] = "1"
    r = self.start()
    with self.assertRaises(RuntimeError) as e:
      r.run(lambda route, seg, file: f"/logs/{seg}", self.report)
    self.assertIn("ValueError: Fingerprinting failed", str(e.exception))
    m = json.loads((r.dir / "manifest.json").read_text())
    self.assertEqual((m["state"], m["written"]), ("error", [3, 4]))

  def test_missing_parts_fail_before_running(self):
    r = Replay(self.dir / "out", "a" * 16 + "|r", [0], ["plannerd"], checkout(self.dir / "bare"))
    with self.assertRaises(RuntimeError) as e:
      r.run(lambda route, seg, file: self.fail("fetched"), self.report)
    self.assertIn("msgq_repo/msgq/ipc_pyx.so", str(e.exception))
    self.assertIn("acados_ocp_solver_pyx.so", str(e.exception))

  def test_cancel_kills_the_runner(self):
    os.environ["FAKE_HANG"] = "1"
    stop = threading.Event()
    r = self.start()
    threading.Timer(1.0, stop.set).start()
    with self.assertRaises(InterruptedError):
      r.run(lambda route, seg, file: f"/logs/{seg}", self.report, stop.is_set)
    self.assertEqual(json.loads((r.dir / "manifest.json").read_text())["state"], "cancelled")


class TestDiff(unittest.TestCase):
  def test_flatten(self):
    out = {}
    replay_runner.flatten({"a": 1, "b": {"c": 2.5, "d": True, "eDEPRECATED": 1, "deprecated": {"x": 1}}, "l": [1, 2], "big": list(range(17)),
                           "s": "text", "logMonoTime": 5, "carStateMonoTime": 5, "frameId": 7, "frameIdExtra": 7}, "/svc", out)
    self.assertEqual(out, {"/svc/a": 1.0, "/svc/b/c": 2.5, "/svc/b/d": 1.0, "/svc/l/0": 1.0, "/svc/l/1": 2.0})


if __name__ == "__main__":
  unittest.main()
