import argparse
import json
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mpld3
import sys
from collections import defaultdict

from tools.lib.logreader import logreader_from_route_or_segment
    
DEMO_ROUTE = "9f583b1d93915c31|2022-04-06--11-34-03"

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

      next_service = SERVICES[SERVICES.index(service)+1]
      if not data['start'][frame_id][next_service]:
        data['start'][frame_id][next_service] = msg.logMonoTime
      data['end'][frame_id][service] = msg.logMonoTime

      if service in SERVICE_TO_DURATIONS:
        for duration in SERVICE_TO_DURATIONS[service]:
          data['duration'][frame_id][service].append((msg.which()+"."+duration, getattr(msg_obj, duration)))

      if service == SERVICES[0]:
        data['timestamp'][frame_id][service].append((msg.which()+" start of frame", msg_obj.timestampSof))
        if not data['start'][frame_id][service]:
          data['start'][frame_id][service] = msg_obj.timestampSof
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


def find_frame_id(time, service, start_times, end_times):
  for frame_id in reversed(start_times):
    if start_times[frame_id][service] and end_times[frame_id][service]:
      if start_times[frame_id][service] <= time <= end_times[frame_id][service]:
        yield frame_id

## ASSUMES THAT AT LEAST ONE CLOUDLOG IS MADE IN CONTROLSD
def insert_cloudlogs(lr, timestamps, start_times, end_times):
  t0 = start_times[min(start_times.keys())][SERVICES[0]] 
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

        if service == "boardd":
          timestamps[latest_controls_frameid][service].append((event, time))
          end_times[latest_controls_frameid][service] = time
        else:
          frame_id_gen = find_frame_id(time, service, start_times, end_times)
          frame_id = next(frame_id_gen, False)
          if frame_id:
            if service == 'controlsd':
              latest_controls_frameid = frame_id
            timestamps[frame_id][service].append((event, time))
          else:
            failed_inserts += 1
  assert latest_controls_frameid > 0, "No timestamp in controlsd"
  assert failed_inserts < len(timestamps), "Too many failed cloudlog inserts"

def print_timestamps(timestamps, durations, start_times, relative):
  t0 = start_times[min(start_times.keys())][SERVICES[0]]
  for frame_id in timestamps.keys():
    print('='*80)
    print("Frame ID:", frame_id)
    if relative:
      t0 = start_times[frame_id][SERVICES[0]]
    for service in SERVICES:
      print("  "+service)  
      events = timestamps[frame_id][service]
      for event, time in sorted(events, key = lambda x: x[1]):
        print("    "+'%-53s%-53s' %(event, str((time-t0)/1e6)))  
      for event, time in durations[frame_id][service]:
        print("    "+'%-53s%-53s' %(event, str(time*1000))) 

def graph_timestamps(timestamps, start_times, end_times, relative):
  t0 = start_times[min(start_times.keys())][SERVICES[0]] 
  fig, ax = plt.subplots()
  ax.set_xlim(0, 150 if relative else 750)
  ax.set_ylim(0, 15)
  ax.set_xlabel('milliseconds')
  ax.set_ylabel('Frame ID')
  colors = ['blue', 'green', 'red', 'yellow', 'purple']
  assert len(colors) == len(SERVICES), 'Each service needs a color'

  points = {"x": [], "y": [], "labels": []}
  for frame_id, services in timestamps.items():
    if relative:
      t0 = start_times[frame_id][SERVICES[0]]
    service_bars = []
    for service, events in services.items():
      if start_times[frame_id][service] and end_times[frame_id][service]:
        start = start_times[frame_id][service]
        end = end_times[frame_id][service]
        service_bars.append(((start-t0)/1e6,(end-start)/1e6))
        for event in events:
          points['x'].append((event[1]-t0)/1e6)
          points['y'].append(frame_id)
          points['labels'].append(event[0])
    ax.broken_barh(service_bars, (frame_id-0.45, 0.9), facecolors=(colors), alpha=0.5)

  scatter = ax.scatter(points['x'], points['y'], marker='d', edgecolor='black')
  tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=points['labels'])

  mpld3.plugins.connect(fig, tooltip)
  plt.legend(handles=[mpatches.Patch(color=colors[i], label=SERVICES[i]) for i in range(len(SERVICES))])
  return fig

def get_timestamps(lr):
  data, frame_mismatches = read_logs(lr)
  lr.reset()
  insert_cloudlogs(lr, data['timestamp'], data['start'], data['end'])
  return data, frame_mismatches

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
  data, _ = get_timestamps(lr)
  print_timestamps(data['timestamp'], data['duration'], data['start'], args.relative)
  if args.plot:
    mpld3.show(graph_timestamps(data['timestamp'], data['start'], data['end'], args.relative))
