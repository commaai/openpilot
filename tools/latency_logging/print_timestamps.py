import argparse
import json
import sys
from collections import defaultdict
from tabulate import tabulate

from tools.lib.route import Route
from tools.lib.logreader import LogReader
    
DEMO_ROUTE = "9f583b1d93915c31|2022-03-28--17-25-12"

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
      rcv_time = float(jmsg['info']['rcvTime']*1e9)
      msg_name = jmsg['info']['msg_name']
      service = MSGQ_TO_SERVICE[msg_name]
      smInfo = jmsg['info']['smInfo']

      # sendcan does not have frame id, translation from its logMonoTime required
      frame_id = translate(pub_time) if msg_name == 'sendcan' else float(smInfo['frameId'])
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
      next_service = SERVICES[SERVICES.index(service)+1]
      service_intervals[frame_id][next_service]["Start"].append(rcv_time)

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

def fill_intervals(timestamp_logs, service_intervals):
  failed_inserts = 0
  def find_frame_id(time, service):
    for frame_id, blocks in service_intervals.items():
      try:
        if service == 'plannerd':
          # plannerd is done when both messages has been sent, other services have their start/end at the first message
          if min(blocks[service]["Start"]) <= float(time) <= max(blocks[service]["End"]):
            return frame_id
        else:
          if min(blocks[service]["Start"]) <= float(time) <= min(blocks[service]["End"]):
            return frame_id
      except:
        pass
    return 0
  for jmsg in timestamp_logs:
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']
    time = float(jmsg['msg']['timestamp']['time'])
    frame_id = find_frame_id(time, service) 
    if frame_id:
      service_intervals[frame_id][service][event].append(time)
    else:
      failed_inserts +=1
  return failed_inserts

def print_timestamps(service_intervals):
  for frame_id, services in service_intervals.items():
    t0 = min([min([min(times) for times in events.values()]) for events in services.values()])
    print(frame_id)
    d = defaultdict( lambda: ("","",[]))
    for service, events in services.items():
      for event, times in events.items():
        key = (min(times)-t0)/1e6
        times = [(float(time)-t0)/1e6 for time in times]
        d[key] = (service, event, times)
    s = sorted(d.items())
    print(tabulate([[item[1][0], item[1][1], item[1][2]] for item in s], headers=["service", "event", "time (ms)"]))
    print("Internal internal_durations:")
    for service, events in dict(internal_durations)[frame_id].items():
      print(service)  
      for event, times in dict(events).items():
        print("    ", event, times)  
    print()

def graph_timestamps(service_intervals):
  fig, gnt = plt.subplots()
  gnt.set_xlim(0, 150)
  gnt.set_ylim(0, len(service_intervals))
  y = 0
  for frame_id, services in service_intervals.items():
    t0 = min([min([min(times) for times in events.values()]) for events in services.values()])
    event_bars = []
    for service, events in services.items():
      for event, times in events.items():
        for time in times:
          t = (time-t0)/1e6
          event_bars.append((t, 0.1))
    gnt.broken_barh(event_bars, (y, 0.9), facecolors=("black"))
    y+=1
  plt.show(block=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper to run timestamp print on openpilot routes",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
  print_timestamps(service_intervals)

  print("Num frames skipped due to failed translations:",failed_transl)
  print("Num frames skipped due to frameId missmatch:",len(frame_mismatches))
  print("Num frames skipped due to empty data:", len(empty_data))
  print("Num inserts failed:", failed_inserts)

