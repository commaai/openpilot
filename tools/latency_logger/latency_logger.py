import argparse
import json
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


from tools.lib.route import Route
from tools.lib.logreader import LogReader
    
DEMO_ROUTE = "9f583b1d93915c31|2022-04-01--17-51-29"

SERVICES = ['camerad', 'modeld', 'plannerd', 'controlsd', 'boardd']
MONOTIME_KEYS = ['modelMonoTime', 'lateralPlanMonoTime']
MSGQ_TO_SERVICE = {
  'roadCameraState': 'camerad',
  'wideRoadCameraState': 'camerad',
  'modelV2': 'modeld',
  'lateralPlan': 'plannerd',
  'longitudinalPlan': 'plannerd',
  'sendcan': 'controlsd',
  'controlsState': 'controlsd'
}
SERVICE_TO_DURATIONS = {
  'camerad': ['processingTime'],
  'modeld': ['modelExecutionTime', 'gpuExecutionTime'],
  'plannerd': ["solverExecutionTime"],
  'controlsd': [],
  'boardd': []
}

def get_logreader(route):
  lr = LogReader(route.log_paths()[0], sort_by_time = True)
  for i, msg in enumerate(lr):
    if msg.which() == "logMessage":
      jmsg = json.loads(msg.logMessage)
      if "ctx" in jmsg and 'daemon' in jmsg['ctx']:
        if jmsg['ctx']['daemon'] == 'camerad' and 'timestamp' in jmsg['msg']:
          return list(lr)[i:]

def read_logs(logreader):
  timestamps = defaultdict(lambda: defaultdict(list))
  internal_durations = defaultdict(lambda: defaultdict(list))
  mono_to_frame = {}
  frame_mismatches = []
  latest_sendcan_monotime = 0
  for msg in logreader:
    msg_name = msg.which()
    if msg_name == 'sendcan':
      latest_sendcan_monotime = msg.logMonoTime
      continue
    elif msg_name in set(MSGQ_TO_SERVICE):
      mono_time = msg.logMonoTime
      service = MSGQ_TO_SERVICE[msg_name]
      msg = getattr(msg, msg_name)

      frame_id = -1
      if hasattr(msg, "frameId"):
        frame_id = msg.frameId
      else:
        for key in MONOTIME_KEYS:
          if hasattr(msg, key) and getattr(msg, key) in mono_to_frame:
            frame_id = mono_to_frame[getattr(msg, key)]
      mono_to_frame[mono_time] = frame_id

      timestamps[frame_id][service].append((msg_name+" published", mono_time))
      for duration in SERVICE_TO_DURATIONS[service]:
        internal_durations[frame_id][service].append((msg_name+"."+duration, getattr(msg,duration)))

      if msg_name == 'controlsState':
        timestamps[frame_id][service].append(("sendcan published", latest_sendcan_monotime))
      elif service == SERVICES[0]:
        timestamps[frame_id][service].append((msg_name+" start", msg.timestampSof))
      elif msg_name == 'modelV2':
        if msg.frameIdExtra != frame_id:
          frame_mismatches.append(frame_id)

  assert sum([len(timestamps[-1][service]) for service in timestamps[-1].keys()]) < 20, "Too many frameId fetch fails"
  assert len(frame_mismatches) < 20, "Too many frame mismatches"
  del timestamps[-1]
  for frame_id in frame_mismatches:
    del timestamps[frame_id]
    del internal_durations[frame_id]
  return (timestamps, internal_durations)

def get_interval(frame_id, service, timestamps):
  if service in timestamps[frame_id]:
    service_max = max(timestamps[frame_id][service], key=lambda x: x[1])[1]
    if service == SERVICES[0]:
      service_min = min(timestamps[frame_id][service], key=lambda x: x[1])[1]
      return (service_min, service_max)
    prev_service = SERVICES[SERVICES.index(service)-1]
    prev_service_max = max(timestamps[frame_id][prev_service], key=lambda x: x[1])[1]
    return (prev_service_max, service_max)
  else:
    return (-1,-1)

def find_frame_id(time, service, timestamps):
  for frame_id in reversed(timestamps):
    interval = get_interval(frame_id, service, timestamps)
    if interval[0] <= time <= interval[1]:
      return frame_id
  return -1

## ASSUMES THAT AT LEAST ONE CLOUDLOG IS MADE IN CONTROLSD
def insert_cloudlogs(logreader, timestamps):
  failed_inserts = 0
  latest_controls_frameid = 0
  for msg in logreader:
    if msg.which() == "logMessage":
      jmsg = json.loads(msg.logMessage)
      if "timestamp" in jmsg['msg']:
        time = int(jmsg['msg']['timestamp']['time'])
        service = jmsg['ctx']['daemon']
        event = jmsg['msg']['timestamp']['event']

        frame_id = latest_controls_frameid if service == "boardd" else find_frame_id(time, service, timestamps)
        if frame_id > -1:
          timestamps[frame_id][service].append((event, time))
          if service == 'controlsd':
            latest_controls_frameid = frame_id
        else:
          failed_inserts += 1
  assert failed_inserts < 250, "Too many failed cloudlog inserts"

def print_timestamps(timestamps, internal_durations, relative_self):
  t0 = min(timestamps[min(timestamps.keys())][SERVICES[0]], key=lambda x: x[1])[1]
  for frame_id in timestamps.keys():
    print('='*80)
    print("Frame ID:", frame_id)
    if relative_self:
      t0 = min(timestamps[frame_id][SERVICES[0]], key=lambda x: x[1])[1]
    for service in SERVICES:
      print("  "+service)  
      events = timestamps[frame_id][service]
      for event, time in sorted(events, key = lambda x: x[1]):
        print("    "+'%-53s%-53s' %(event, str((time-t0)/1e6)))  
      for event, time in internal_durations[frame_id][service]:
        print("    "+'%-53s%-53s' %(event, str(time))) 

def graph_timestamps(timestamps, relative_self):
  t0 = min(timestamps[min(timestamps.keys())][SERVICES[0]], key=lambda x: x[1])[1]
  y = 0
  gnt = plt.subplots()[1]
  gnt.set_xlim(0, 150 if relative_self else 750)
  gnt.set_ylim(0, len(timestamps) if relative_self else 10)
  for frame_id, services in timestamps.items():
    if relative_self:
      t0 = min(timestamps[frame_id][SERVICES[0]], key=lambda x: x[1])[1]
    event_bars = []
    service_bars = []
    for service, events in services.items():
      if len(events)==0:
        continue
      start, end = get_interval(frame_id, service,timestamps)
      service_bars.append(((start-t0)/1e6,(end-start)/1e6))
      for event in events:
        event_bars.append(((event[1]-t0)/1e6, 0.1))
    gnt.broken_barh(service_bars, (y, 0.9), facecolors=(["blue", 'green', 'red', 'yellow', 'purple']))
    gnt.broken_barh(event_bars, (y, 0.9), facecolors=("black"))
    y+=1
  plt.show(block=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "A helper to run timestamp print on openpilot routes",
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--relative_self", action = "store_true", help = "Print and plot starting a 0 each time")
  parser.add_argument("--demo", action = "store_true", help = "Use the demo route instead of providing one")
  parser.add_argument("--plot", action="store_true", help="If a plot should be generated")
  parser.add_argument("route_name", nargs = '?', help = "The route to print")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  route = Route(DEMO_ROUTE if args.demo else args.route_name.strip())
  logreader = get_logreader(route)
  timestamps, internal_durations = read_logs(logreader)
  insert_cloudlogs(logreader, timestamps)
  print_timestamps(timestamps, internal_durations, args.relative_self)
  if args.plot:
    graph_timestamps(timestamps, args.relative_self)
