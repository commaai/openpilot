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


def flatJson(data, logMonoTime, parent_key=''):
    if isinstance(data, dict):
        for k, v in data.items():
            if parent_key:
                key = f"{parent_key}.{k}"
            else:
                key = k
            if isinstance(v, (dict, list)):
                flatJson(v, logMonoTime, key)
            else:
                if isinstance(v, (int, float)):
                  rr.set_time_sequence(k, logMonoTime)
                  rr.log(k, rr.Scalar(v))
                else:
                  print("Not plottable value " + str(v))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, (int, float)):
              rr.set_time_sequence(str(i), logMonoTime)
              rr.log(f"{parent_key}[{i}]", rr.Scalar(v))
            else:
               print("Not plottable value " + str(v))
    else:
        if isinstance(v, (int, float)):
          rr.log(parent_key, rr.Scalar(data))
        else:
          print("Not plottable value " + str(data))

# TODO: Add "Space view" for every msg.which() property
def addGraph(logPaths):
  counter = 1
  for logPath in logPaths:
    print(counter)
    counter+=1
    rlog = LogReader(logPath)
    for msg in rlog:
      if msg.which() == "deviceState": # Select one property to plot, one space view available
        jsonMsg = json.loads(json.dumps(convertBytesToString(msg.to_dict())))
        flatJson(jsonMsg, jsonMsg.get("logMonoTime"), parent_key='')
        continue


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
