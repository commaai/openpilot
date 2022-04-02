import argparse
import json
import sys
from collections import defaultdict

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
      msg = msg.to_dict()[msg_name]

      frame_id = -1
      if 'frameId' in msg:
        frame_id = msg['frameId']
      else:
        for key in MONOTIME_KEYS:
          if key in msg and msg[key] in mono_to_frame:
            frame_id = mono_to_frame[msg[key]]
      mono_to_frame[mono_time] = frame_id

      timestamps[frame_id][service].append((msg_name+" published", mono_time))
      for duration in SERVICE_TO_DURATIONS[service]:
        internal_durations[frame_id][service].append((msg_name+"."+duration, msg[duration]))

      if msg_name == 'controlsState':
        timestamps[frame_id][service].append(("sendcan published", latest_sendcan_monotime))
      elif service == SERVICES[0]:
        timestamps[frame_id][service].append((msg_name+" start", msg['timestampSof']))
      elif msg_name == 'modelV2':
        if msg['frameIdExtra'] != frame_id:
          frame_mismatches.append(frame_id)

  #TODO: count -1, many of these are because camera starts after, 
  del timestamps[-1]
  #TODO: why is this needed?
  del timestamps[1]
  for frame_id in frame_mismatches:
    del timestamps[frame_id]
    del internal_durations[frame_id]
  return (timestamps, internal_durations)

def get_interval(frame_id, service, timestamps):
  service_min = min(timestamps[frame_id][service], key=lambda x: x[1])[1]
  service_max = max(timestamps[frame_id][service], key=lambda x: x[1])[1]
  if service == SERVICES[0]:
    return (service_min, service_max)
  prev_service = SERVICES[SERVICES.index(service)-1]
  prev_service_max = max(timestamps[frame_id][prev_service], key=lambda x: x[1])[1]
  return (prev_service_max, service_max)

def find_frame_id(time, service, timestamps):
  for frame_id in reversed(timestamps):
    interval = get_interval(frame_id, service, timestamps)
    if interval[0] <= time <= interval[1]:
      return frame_id
  return -1

## ASSUMES THAT AT LEAST ONE CLOUDLOG IS MADE IN CONTROLSD
def insert_cloudlogs(logreader, timestamps):
  t0 = min(timestamps[min(timestamps.keys())][SERVICES[0]], key=lambda x: x[1])[1]
  failed_inserts = 0
  latest_controls_frameid = -1
  for msg in logreader:
    if msg.which() == "logMessage":
      jmsg = json.loads(msg.logMessage)
      if "timestamp" in jmsg['msg']:
        time = int(jmsg['msg']['timestamp']['time'])
        if time < t0:
          continue
        service = jmsg['ctx']['daemon']
        event = jmsg['msg']['timestamp']['event']

        frame_id = latest_controls_frameid if service == "boardd" else find_frame_id(time, service, timestamps)
        if service == 'controlsd':
          latest_controls_frameid = frame_id

        if frame_id > -1:
          timestamps[frame_id][service].append((event, time))
        else:
          failed_inserts += 1
  # TODO print last?
  #return failed_inserts

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "A helper to run timestamp print on openpilot routes",
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--relative_self", action = "store_true", help = "Print and plot starting a 0 each time")
  parser.add_argument("--demo", action = "store_true", help = "Use the demo route instead of providing one")
  parser.add_argument("route_name", nargs = '?', help = "The route to print")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  r = Route(DEMO_ROUTE if args.demo else args.route_name.strip())
  lr = LogReader(r.log_paths()[0], sort_by_time = True)
  timestamps, internal_durations = read_logs(lr)
  insert_cloudlogs(lr, timestamps)
  print_timestamps(timestamps, internal_durations, args.relative_self)
  #TODO
  #print("Num frames skipped due to failed translations:",failed_transl)
  #print("Num frames skipped due to frameId missmatch:",frame_mismatches)
  #print("Num frames skipped due to empty data:", empty_data)
  #print("Num inserts failed:", failed_inserts)
