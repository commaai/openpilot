import multiprocessing
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route
from cereal.services import services
import subprocess
import sys
import os
import capnp


NUM_CPUS = 1
DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"
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


# Import newest rerun version, replace when rerun version is greater than 0.15.1
# try:
#     import rerun as rr
#     import rerun.blueprint as rrb
# except ImportError:
#     print("Rerun SDK is not installed. Trying to install it...")
#     subprocess.run([sys.executable, "-m", "pip", "install", "rerun-sdk"])
#     print("Rerun installed, restarting script")
#     os.execv(sys.executable, [sys.executable] + sys.argv)


topics = sorted(services.keys())
excluded = ['sentinel']
is_first_run = True


# Log data-(key, value) to rerun
# Here a property could be excluded from graphing
def log_data(key, value):
    rr.log(str(key), rr.Scalar(value))


# Log capnp message to rerun
def log_msg(msg, parent_key=''):
    stack = [(msg, parent_key)]
    while stack:
        current_msg, current_parent_key = stack.pop()
        if hasattr(current_msg, 'schema') and hasattr(current_msg.schema, 'fields'):
            msgFields = current_msg.schema.fields
            for msgField in msgFields:
                try:
                    value = getattr(current_msg, msgField)
                except capnp.KjException:
                    continue
                new_key = current_parent_key + "/" + msgField
                if isinstance(value, (int, float)):
                    log_data(new_key, value)
                elif isinstance(value, capnp.lib.capnp._DynamicStructReader):
                    stack.append((value, new_key))
                elif isinstance(value, capnp.lib.capnp._DynamicListReader):
                    for index, item in enumerate(value):
                        if isinstance(item, (int, float)):
                            log_data(new_key + '/' + str(index), item)
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
  bytesImgData = thumbnailMsg.thumbnail
  rr.log("/thumbnail", rr.ImageEncoded(contents=bytesImgData))


# Download segment data and log it
def process_log(log_path):
  rlog = LogReader(log_path)
  for msg in rlog:
    global is_first_run
    if is_first_run:
        blueprint = createBlueprint()
        rr.init("rerun_test", spawn=True, default_blueprint=blueprint)
        is_first_run = False
    rr.set_time_nanos("TIMELINE", msg.logMonoTime)
    if msg.which() != "thumbnail":
        log_msg(getattr(msg, msg.which()), msg.which())
    else:
       log_thumbnail(getattr(msg, msg.which()))


# Create blueprint and initiate rerun and data logging
def addGraphs(log_paths):
  print(f"Downloading logs [{len(log_paths)}]")
  with multiprocessing.Pool(NUM_CPUS) as pool:
    for log_path in log_paths:
        pool.apply_async(process_log, (log_path,))
    pool.close()
    pool.join()
  print("Messages sent to rerun!")


if __name__ == '__main__':
  if len(sys.argv) == 1:
    route_name = DEMO_ROUTE
    print("No route was provided, using demo route")
  else:
    route_name = sys.argv[1]
  # Get logs for a route
  print("Getting route log paths")
  route = Route(route_name)
  logPaths = route.log_paths()
  addGraphs(logPaths)
