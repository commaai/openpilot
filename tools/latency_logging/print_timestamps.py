import argparse
import json
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

from tools.lib.route import Route
from tools.lib.logreader import LogReader
    
DEMO_ROUTE = "9f583b1d93915c31|2022-03-30--11-11-53"

SERVICES = ['camerad', 'modeld', 'plannerd', 'controlsd', 'boardd']

MSGQ_TO_SERVICE = {
        'roadCameraState':'camerad',
        'wideRoadCameraState':'camerad',
        'modelV2':'modeld',
        'lateralPlan':'plannerd',
        'longitudinalPlan':'plannerd',
        'sendcan':'controlsd'
        }

SERVICE_TO_DURATIONS = {
        'camerad':['timestampSof', 'processingTime'],
        'modeld':['modelExecutionTime', 'gpuExecutionTime'],
        'plannerd':["solverExecutionTime"],
        'controlsd':[],
        'boardd':[]
        }

def get_relevant_logs(logreader):
  logs = [[], [], []]
  for msg in logreader:
    if msg.which() == "logMessage" and "timestamp" in msg.logMessage:
      msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}").replace("\\", '')
      jmsg = json.loads(msg)
      if "timestampExtra" in jmsg['msg']:
        if jmsg['msg']['timestampExtra']['event'] == "translation":
          logs[0].append(jmsg['msg']['timestampExtra'])
        else:
          logs[1].append(jmsg['msg']['timestampExtra'])
      elif "timestamp" in jmsg['msg']:
        logs[2].append(jmsg)
  return logs

def get_translation_lut(translation_logs):
  translationdict = {}
  for jmsg in translation_logs:
    logMonoTime = float(jmsg['info']['from'])
    frame_id = int(jmsg['info']['to'])
    translationdict[logMonoTime] = frame_id
  return translationdict

def get_service_intervals(timestampExtra_logs, translationdict):
  service_intervals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  frame_mismatches = []
  internal_durations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  failed_transl = 0

  def translate(logMono):
    return -1 if logMono not in translationdict else translationdict[logMono]

  for jmsg in timestampExtra_logs:
    if "smInfo" in jmsg['info']:
      # retrive info from logs
      pub_time = float(jmsg['info']['logMonoTime'])
      msg_name = jmsg['info']['msg_name']
      service = MSGQ_TO_SERVICE[msg_name]
      smInfo = jmsg['info']['smInfo']

      # sendcan does not have frame id, translation from its logMonoTime required
      frame_id = translate(pub_time) if msg_name == 'sendcan' else int(smInfo['frameId'])
      if frame_id == -1:
        failed_transl += 1
        continue

      # check for frame missmatches in model 
      if msg_name == 'modelV2':
        if smInfo['frameIdExtra'] != frame_id:
          frame_mismatches.append(frame_id)

      # retrive the internal durations of the messages 
      for duration in SERVICE_TO_DURATIONS[service]:
        # use timestampSof as the start time for camera
        if duration == 'timestampSof':
          service_intervals[frame_id][service]["Start"].append(float(smInfo[duration]))
        else:
          internal_durations[frame_id][service][duration].append(float(smInfo[duration]*1e3))

      service_intervals[frame_id][service]["End"].append(pub_time)

    # sendcan to panda is sent over usb and not logged automatically
    elif "Pipeline end" == jmsg['event']:
      frame_id = translate(float(jmsg["info"]["logMonoTime"]))
      if frame_id == -1:
        failed_transl += 1
        continue
      service_intervals[frame_id][SERVICES[-1]]["End"].append(float(jmsg["time"]))
  return (service_intervals, frame_mismatches, internal_durations, failed_transl)

def get_empty_data(service_intervals):
  for frame_id in service_intervals.keys():
    for service in SERVICES:
      if service not in service_intervals[frame_id]:
        yield frame_id

def exclude_bad_data(frame_ids, service_intervals):
  for frame_id in frame_ids:
    del service_intervals[frame_id]

def find_frame_id(time, service):
  prev_service = SERVICES[SERVICES.index(service)-1]
  for frame_id, intervals in service_intervals.items():
    if service == 'camerad':
      if min(intervals['camerad']["Start"]) <= time <= min(intervals[service]["End"]):
        return frame_id
    else:
      if min(intervals[prev_service]["End"]) <= time <= max(intervals[service]["End"]):
        return frame_id
  return -1

def fill_intervals(timestamp_logs, service_intervals):
  t0 = min(service_intervals[min(service_intervals.keys())]['camerad']["Start"])
  failed_inserts = 0
  for jmsg in timestamp_logs:
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']
    time = float(jmsg['msg']['timestamp']['time'])
    if time < t0:
        continue
    frame_id = find_frame_id(time, service) 
    if frame_id != -1:
      service_intervals[frame_id][service][event].append(time)
    else:
      failed_inserts +=1
  return failed_inserts

def print_timestamps(service_intervals, relative_self):
  t0 = min([min([min(times) for times in events.values()]) for events in service_intervals[min(service_intervals.keys())].values()])
  for frame_id, services in service_intervals.items():
    print('='*80)
    print("Frame ID:", frame_id)

    print("Timestamps:")
    if relative_self:
      t0 = min([min([min(times) for times in events.values()]) for events in services.values()])
    for service in SERVICES:
      events = service_intervals[frame_id][service]
      print("  "+service)  
      for event, times in sorted(events.items(), key=lambda x: x[1][-1]):
        times = [(time-t0)/1e6 for time in times]
        print("    "+'%-50s%-50s' %(event, str(times)))  

    print("Internal durations:")
    for service, events in dict(internal_durations)[frame_id].items():
      print("  "+service)  
      for event, times in dict(events).items():
        print("    "+'%-50s%-50s' %(event, str(times)))  

def graph_timestamps(service_intervals, relative_self):
  t0 = min([min([min(times) for times in events.values()]) for events in service_intervals[min(service_intervals.keys())].values()])
  gnt = plt.subplots()[1]
  gnt.set_xlim(0, 150 if relative_self else 750)
  gnt.set_ylim(0, len(service_intervals))
  y = 0
  for frame_id, services in service_intervals.items():
    if relative_self:
      t0 = min([min([min(times) for times in events.values()]) for events in services.values()])
    event_bars = []
    service_bars = []
    start = min(service_intervals[frame_id]['camerad']["Start"])
    for service, events in services.items():
      for event, times in sorted(events.items(), key=lambda x: x[1][-1]):
        if event == "End":
            end = max(times) if service != "camerad" else min(times)
            service_bars.append(((start-t0)/1e6, (end-start)/1e6))
            start = end
        for time in times:
          t = (time-t0)/1e6
          event_bars.append((t, 0.1))
    gnt.broken_barh(event_bars, (y, 0.9), facecolors=("black"))
    gnt.broken_barh(service_bars, (y, 0.9), facecolors=(["blue", 'green', 'red', 'yellow', 'purple']))
    y+=1
  plt.show(block=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper to run timestamp print on openpilot routes",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--plot", action="store_true", help="If a plot should be generated")
  parser.add_argument("--relative_self", action="store_true", help="Print and plot starting a 0 each time")
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("route_name", nargs='?', help="The route to print")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  r = Route(DEMO_ROUTE if args.demo else args.route_name.strip())
  lr = LogReader(r.log_paths()[0])
  translation_logs, timestampExtra_logs, timestamp_logs = get_relevant_logs(lr)
  translationdict = get_translation_lut(translation_logs)
  service_intervals, frame_mismatches, internal_durations, failed_transl= get_service_intervals(timestampExtra_logs, translationdict)
  empty_data = list(set(get_empty_data(service_intervals)))
  exclude_bad_data(set(empty_data+frame_mismatches), service_intervals)
  failed_inserts = fill_intervals(timestamp_logs, service_intervals)
  print_timestamps(service_intervals, args.relative_self)
  if args.plot:
    graph_timestamps(service_intervals, args.relative_self)

  print("Num frames skipped due to failed translations:",failed_transl)
  print("Num frames skipped due to frameId missmatch:",len(frame_mismatches))
  print("Num frames skipped due to empty data:", len(empty_data))
  print("Num inserts failed:", failed_inserts)

