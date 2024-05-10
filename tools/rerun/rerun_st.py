from openpilot.tools.lib.logreader import LogReader
from cereal.services import services
from functools import partial
import subprocess
import sys
import os

NUM_CPUS = 4
# DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"
DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19/1:2"

WHEEL_URL = "https://build.rerun.io/commit/660463d/wheels"
RERUN_VERSION = "rerun-cli 0.16.0-alpha.2 [rustc 1.76.0 (07dca489a 2024-02-04), LLVM 17.0.6] x86_64-unknown-linux-gnu main 660463d, built 2024-04-28T12:33:59Z"


try:
    import rerun as rr
    import rerun.blueprint as rrb
    # Simple version check
    result = subprocess.run(["rerun", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        if result.stdout.strip() != RERUN_VERSION.strip():
          print("YOU NEED TO UNINSTALL RERUN AND THAN START THIS SCRIPT")
          print("You have "+ result.stdout)
          print("You need "+ RERUN_VERSION)
          print("\nHINT: pip uninstall rerun-sdk")
          exit(0)
except ImportError:
    print("Rerun SDK is not installed. Trying to install it...")
    # subprocess.run([sys.executable, "-m", "pip", "install", "rerun-sdk"])
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--pre", "-f", WHEEL_URL, "--upgrade", "rerun-sdk"],
        check=True
    )
    print("Rerun installed, restarting script")
    os.execv(sys.executable, [sys.executable] + sys.argv)


topics = sorted(services.keys())
excluded = ['sentinel']
is_first_run = True


# Log dict message to rerun
def log_msg(msg, parent_key=''):
    stack = [(msg, parent_key)]
    while stack:
        current_msg, current_parent_key = stack.pop()
        if isinstance(current_msg, dict):
            for key, value in current_msg.items():
                new_key = f"{current_parent_key}/{key}"
                if isinstance(value, (int, float)):
                    rr.log(str(new_key), rr.Scalar(value))
                elif isinstance(value, dict):
                    stack.append((value, new_key))
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, (int, float)):
                            rr.log(f"{new_key}/{index}", rr.Scalar(item))
        else:
            pass  # Not a plottable value


# Create a blueprint for selected topics
def createBlueprint():
  timeSeriesViews = []
  for topic in topics:
      timeSeriesViews.append(rrb.TimeSeriesView(name=topic, origin=f"/{topic}/", visible=False))
      rr.log(topic, rr.SeriesLine(name=topic), timeless=True)
      blueprint = rrb.Blueprint(rrb.Grid(rrb.Vertical(*timeSeriesViews,rrb.SelectionPanel(expanded=False),rrb.TimePanel(expanded=False)),
                                         rrb.Spatial2DView(name="thumbnail", origin="/thumbnail")))
  return blueprint


# Log thumbnail, could be used for potential logging of any image
def log_thumbnail(thumbnailMsg):
  bytesImgData = thumbnailMsg.get('thumbnail')
  rr.log("/thumbnail", rr.ImageEncoded(contents=bytesImgData))


def process(blueprint, lr):
  ret = []
  rr.init("rerun_test", spawn=True, default_blueprint=blueprint)
  for msg in lr:
    ret.append(msg)
    rr.set_time_nanos("TIMELINE", msg.logMonoTime)
    if msg.which() != "thumbnail":
      log_msg(msg.to_dict()[msg.which()], msg.which())
    else:
      log_thumbnail(msg.to_dict()[msg.which()])
  return ret


if __name__ == '__main__':
  if len(sys.argv) == 1:
    route_name = DEMO_ROUTE
    print("No route was provided, using demo route")
  else:
    route_name = sys.argv[1]
  # Get logs for a route
  blueprint = createBlueprint()
  print("Getting route log paths")
  lr = LogReader(route_name)
  lr.run_across_segments(NUM_CPUS, partial(process, blueprint))
