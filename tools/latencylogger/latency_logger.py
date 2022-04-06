import argparse
import json
import mpld3
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

from tools.lib.logreader import logreader_from_route_or_segment
    
DEMO_ROUTE = "9f583b1d93915c31|2022-04-01--17-51-29"

SERVICES = ['camerad', 'modeld', 'plannerd', 'controlsd', 'boardd']
# Retrive controlsd frameId from lateralPlan, mismatch with longitudinalPlan will be ignored
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
}

def read_logs(lr):
  data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  mono_to_frame = {}
  frame_mismatches = []
  latest_sendcan_monotime = 0
  for msg in lr:
    if msg.which() == 'sendcan':
      latest_sendcan_monotime = msg.logMonoTime
      continue
    
    if msg.which() in MSGQ_TO_SERVICE:
      service = MSGQ_TO_SERVICE[msg.which()]
      msg_obj = getattr(msg, msg.which())

      frame_id = -1
      if hasattr(msg_obj, "frameId"):
        frame_id = msg_obj.frameId
      else:
        continue_outer = False
        for key in MONOTIME_KEYS:
          if hasattr(msg_obj, key):
            if getattr(msg_obj, key) == 0:
              # Filter out controlsd messages which arrive before the camera loop 
              continue_outer = True
            elif getattr(msg_obj, key) in mono_to_frame:
              frame_id = mono_to_frame[getattr(msg_obj, key)]
        if continue_outer:
          continue
      mono_to_frame[msg.logMonoTime] = frame_id

      data['timestamp'][frame_id][service].append((msg.which()+" published", msg.logMonoTime))
      if service in SERVICE_TO_DURATIONS:
        for duration in SERVICE_TO_DURATIONS[service]:
          data['duration'][frame_id][service].append((msg.which()+"."+duration, getattr(msg_obj, duration)))

      if service == SERVICES[0]:
        data['timestamp'][frame_id][service].append((msg.which()+" start of frame", msg_obj.timestampSof))
      elif msg.which() == 'controlsState':
        # Sendcan is published before controlsState, but the frameId is retrived in CS
        data['timestamp'][frame_id][service].append(("sendcan published", latest_sendcan_monotime))
      elif msg.which() == 'modelV2':
        if msg_obj.frameIdExtra != frame_id:
          frame_mismatches.append(frame_id)

  # Failed frameId fetches are stored in -1
  assert sum([len(data['timestamp'][-1][service]) for service in data['timestamp'][-1].keys()]) < 20, "Too many frameId fetch fails"
  del data['timestamp'][-1]
  assert len(frame_mismatches) < 20, "Too many frame mismatches"
  return (data, frame_mismatches)

def get_min_time(frame_id, service, timestamps, use_max=False):
  return min(timestamps[frame_id][service], key=lambda x: x[1])[1] if not use_max else max(timestamps[frame_id][service], key=lambda x: x[1])[1]

def get_interval(frame_id, service, timestamps):
  try:
    service_max = get_min_time(frame_id, service, timestamps, use_max=True) 
    if service == SERVICES[0]:
      service_min = get_min_time(frame_id, service, timestamps) 
      return (service_min, service_max)
    prev_service = SERVICES[SERVICES.index(service)-1]
    prev_service_max = get_min_time(frame_id, prev_service, timestamps, use_max=True) 
    return (prev_service_max, service_max)
  except ValueError:
    return (-1,-1)

def find_frame_id(time, service, timestamps):
  for frame_id in reversed(timestamps):
    start, end = get_interval(frame_id, service, timestamps)
    if start <= time <= end:
      return frame_id
  return -1

## ASSUMES THAT AT LEAST ONE CLOUDLOG IS MADE IN CONTROLSD
def insert_cloudlogs(lr, timestamps):
  t0 = get_min_time(min(timestamps.keys()), SERVICES[0], timestamps) 
  failed_inserts = 0
  latest_controls_frameid = 0
  for msg in lr:
    if msg.which() == "logMessage":
      jmsg = json.loads(msg.logMessage)
      if "timestamp" in jmsg['msg']:
        time = int(jmsg['msg']['timestamp']['time'])
        service = jmsg['ctx']['daemon']
        event = jmsg['msg']['timestamp']['event']
        if time < t0:
          # Filter out controlsd messages which arrive before the camera loop 
          continue

        frame_id = latest_controls_frameid if service == "boardd" else find_frame_id(time, service, timestamps)
        if frame_id > -1:
          timestamps[frame_id][service].append((event, time))
          if service == 'controlsd':
            latest_controls_frameid = frame_id
        else:
          failed_inserts += 1
  assert failed_inserts < len(timestamps), "Too many failed cloudlog inserts"

def print_timestamps(timestamps, durations ,relative):
  t0 = get_min_time(min(timestamps.keys()), SERVICES[0], timestamps) 
  for frame_id in timestamps.keys():
    print('='*80)
    print("Frame ID:", frame_id)
    if relative:
      t0 = get_min_time(frame_id, SERVICES[0], timestamps) 
    for service in SERVICES:
      print("  "+service)  
      events = timestamps[frame_id][service]
      for event, time in sorted(events, key = lambda x: x[1]):
        print("    "+'%-53s%-53s' %(event, str((time-t0)/1e6)))  
      for event, time in durations[frame_id][service]:
        print("    "+'%-53s%-53s' %(event, str(time*1000))) 

def graph_timestamps(timestamps, relative):
  t0 = get_min_time(min(timestamps.keys()), SERVICES[0], timestamps) 
  fig, ax = plt.subplots()
  ax.set_xlim(0, 150 if relative else 750)
  ax.set_ylim(0, 15)

  points = {"x": [], "y": [], "labels": []}
  for frame_id, services in timestamps.items():
    if relative:
      t0 = get_min_time(frame_id, SERVICES[0], timestamps) 
    service_bars = []
    for service, events in services.items():
      start, end = get_interval(frame_id, service,timestamps)
      service_bars.append(((start-t0)/1e6,(end-start)/1e6))
      for event in events:
        points["x"].append((event[1]-t0)/1e6)
        points["y"].append(frame_id+0.45)
        points["labels"].append(event[0])
    ax.broken_barh(service_bars, (frame_id, 0.9), facecolors=(["blue", 'green', 'red', 'yellow', 'purple']))

  scatter = ax.scatter(points['x'], points['y'], marker="d", edgecolor='black')
  tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=points["labels"])
  ax.legend()
  mpld3.plugins.connect(fig, tooltip)
  #mpld3.save_html(fig, 'test.html')
  mpld3.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A tool for analyzing openpilot's end-to-end latency",
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--relative", action="store_true", help="Make timestamps relative to the start of each frame")
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("--plot", action="store_true", help="If a plot should be generated")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route to print")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  r = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
  lr = logreader_from_route_or_segment(r, sort_by_time=True)
  data, frame_mismatches = read_logs(lr)
  lr.reset()
  insert_cloudlogs(lr, data['timestamp'])
  print_timestamps(data['timestamp'], data['duration'], args.relative)
  if args.plot:
    graph_timestamps(data['timestamp'], args.relative)
