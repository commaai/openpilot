import os, sys, time, multiprocessing

N = int(os.environ.get("NPROC", str(os.cpu_count())))
DEVICE = os.environ.get("DEV", "AMD")

# this tests the total number of processes that can be running tinygrad at a time
def proc(i, device, stop_evt):
  from tinygrad import Tensor

  try:
    a = Tensor.ones(2, device=device).contiguous()
    b = Tensor.ones(2, device=device).contiguous()
    c = (a + b).realize()
    assert c.tolist() == [2, 2]
  except Exception as e:
    # fail if it fails
    print(f"[child {i:2d}] tinygrad op failed: {e}", file=sys.stderr)
    # non-zero exit code propagated back to parent
    sys.exit(1)

  # TODO: wait here for global exit if success. fail if it fails
  # -> We wait on a global Event shared from the parent.
  print(f"[child {i:2d}] success")
  stop_evt.wait()
  # Normal successful exit
  sys.exit(0)

if __name__ == "__main__":
  print(f"testing {N} concurrent tinygrad processes")

  # global exit event, shared by all children
  stop_evt = multiprocessing.Event()
  procs = []

  # launch n proc of proc 1 per 200 ms
  for i in range(N):
    p = multiprocessing.Process(target=proc, args=(i, DEVICE, stop_evt), name=f"tinygrad-proc-{i}")
    p.start()
    procs.append(p)
    time.sleep(0.1)  # 100 ms between launches

  # signal global exit
  time.sleep(0.5)
  stop_evt.set()

  # join all children
  for p in procs: p.join()

  # check for failures
  failed = [p for p in procs if p.exitcode != 0]
  if failed:
    print(f"{len(failed)} / {len(procs)} processes failed "
          f"with exit codes: {[p.exitcode for p in failed]}", file=sys.stderr)
    sys.exit(1)

  print(f"All {len(procs)} tinygrad processes ran successfully")
  sys.exit(0)
