import pickle, datetime, os, tempfile, subprocess, zipfile, importlib.util
from extra.hcqfuzz.spec import TestSpec
from tinygrad.helpers import getenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "tests")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def collect_tests():
  run_files = getenv("RUN_FILES", "").split(",")
  skip_tests = getenv("SKIP_FILES", "").split(",")

  tests = []
  for filename in os.listdir(TEST_DIR):
    if filename.endswith(".py") and not filename.startswith("__"):
      if run_files and filename[:-3] not in run_files: continue
      if skip_tests and filename[:-3] in skip_tests: continue

      filepath = os.path.join(TEST_DIR, filename)
      module_name = f"tests.{filename[:-3]}"
      module = importlib.import_module(module_name)
      for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, TestSpec) and attr is not TestSpec:
          tests.append(attr())
  return tests

def on_start_run(dev, test):
  os.makedirs(REPORTS_DIR, exist_ok=True)
  pickle.dump((dev, test), open(f"{REPORTS_DIR}/last_launch.pkl", "wb"))

def create_report(dev, test, result, stdout, stderr):
  os.makedirs(REPORTS_DIR, exist_ok=True)

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  report_name = f"{timestamp}_{test.name()}_report"
  report_path = os.path.join(REPORTS_DIR, report_name)

  os.makedirs(report_path, exist_ok=False)

  pickle_path = os.path.join(report_path, "repro.pkl")
  with open(pickle_path, "wb") as f: pickle.dump((dev, test), f)

  stdout_path = os.path.join(report_path, "stdout.txt")
  with open(stdout_path, "w") as f: f.write(stdout)

  stderr_path = os.path.join(report_path, "stderr.txt")
  with open(stderr_path, "w") as f: f.write(stderr)

  dmesg_path = os.path.join(report_path, "dmesg.txt")
  dmesg_output = subprocess.check_output(["sudo", "dmesg", "--ctime", "--color=never"], text=True)
  with open(dmesg_path, "w") as f: f.write(dmesg_output)

  summary_path = os.path.join(report_path, "summary.txt")
  with open(summary_path, "w") as f:
    f.write(f"Test: {test.name()}\n")
    f.write(f"Dev params: {vars(dev)}\n")
    f.write(f"Test params: {vars(test)}\n")
    f.write(f"Exit Code: {result}\n")

  print(f"Crash report saved to {report_path}")

_log_file = None
def init_log():
  global _log_file
  os.makedirs(REPORTS_DIR, exist_ok=True)

  ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  name = f"log_{ts}.log"
  _log_file = open(f"{REPORTS_DIR}/{name}", "a", buffering=1)

def log(msg="", end="\n", flush=False):
  global _log_file
  _log_file.write(msg.replace("\r", "\n") + end)
  if flush: _log_file.flush()
  print(msg + " " * 60, end=end, flush=flush)
