# TODO: install pip3 install rerun-sdk
import sys
import json
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route
import rerun as rr

def convertBytesToString(data):
  if isinstance(data, bytes):
    return data.decode('latin-1')
  elif isinstance(data, list):
    return [convertBytesToString(item) for item in data]
  elif isinstance(data, dict):
    return {key: convertBytesToString(value) for key, value in data.items()}
  else:
    return data
  
def addGraph(logPaths):
  counter = 1
  for logPath in logPaths:
    print(counter)
    counter+=1
    rlog = LogReader(logPath)
    for msg in rlog:
      if msg.which() == "accelerometer":
        rr.set_time_sequence(msg.which(), msg.logMonoTime)
        rr.log("acceleration", rr.Scalar(msg.accelerometer.acceleration.v[0]))
        # print(json.dumps(convertBytesToString(msg.to_dict())))
        # exit(0)
   

if __name__ == '__main__':
  if len(sys.argv) == 1:
    route_name = "a2a0ccea32023010|2023-07-27--13-01-19"
    print("No route was provided, using demo route")
  else:
    route_name = sys.argv[1]
  # Get logs for a route
  print("Getting route log paths")
  rr.init("plot_example", spawn=True)
  route = Route(route_name)
  logPaths = route.log_paths()
  addGraph(logPaths)