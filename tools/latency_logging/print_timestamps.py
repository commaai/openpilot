from tools.lib.route import Route
from tools.lib.logreader import LogReader
from tabulate import tabulate
import json
from collections import defaultdict

r = Route("9f583b1d93915c31|2022-03-28--17-25-12") 
lr = LogReader(r.log_paths()[0])

services = ['camerad', 'modeld', 'plannerd', 'controlsd', 'boardd']

msgq_to_service = {}
msgq_to_service['roadCameraState'] = "camerad"
msgq_to_service['wideRoadCameraState'] = "camerad"
msgq_to_service['modelV2'] = "modeld"
msgq_to_service['lateralPlan'] = "plannerd"
msgq_to_service['longitudinalPlan'] = "plannerd"
msgq_to_service['sendcan'] = "controlsd"

service_to_durations = defaultdict(list)
service_to_durations['camerad'] = ['timestampSof', 'processingTime']
service_to_durations['modeld'] = ['modelExecutionTime', 'gpuExecutionTime']
service_to_durations['plannerd'] = ["solverExecutionTime"]

# these have to be iterated after the translation event have been made, but prevent reiteration of all logs
timestampExtra_logs = [] 
timestamp_logs = []

# Build translation dict while collecting the other logs
translationdict = {}
for msg in lr:
  if msg.which() == "logMessage" and "timestamp" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}").replace("\\", '')
    jmsg = json.loads(msg)
    if "timestampExtra" in jmsg['msg']:
      if jmsg['msg']['timestampExtra']['event'] == "translation":
        jmsg = jmsg['msg']['timestampExtra']
        logMonoTime = jmsg['info']['from']
        frame_id = jmsg['info']['to']
        translationdict[float(logMonoTime)] = int(frame_id)
      else:
        timestampExtra_logs.append(jmsg['msg']['timestampExtra'])
    elif "timestamp" in jmsg['msg']:
      timestamp_logs.append(jmsg)

failed_transl = 0
def translate(logMono):
  logMono = float(logMono)
  if logMono in translationdict:
    return translationdict[logMono]
  else:
    global failed_transl
    failed_transl+=1
    return -1

# Build service blocks
service_blocks = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
skip_frames = []
durations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for jmsg in timestampExtra_logs:
  if "smInfo" in jmsg['info']:
    pubTime = jmsg['info']['logMonoTime']
    rcvTime =jmsg['info']['rcvTime']*1e9
    msg_name = jmsg['info']['msg_name']
    service = msgq_to_service[msg_name]
    smInfo = jmsg['info']['smInfo']
    frame_id = translate(pubTime) if msg_name == 'sendcan' else smInfo['frameId']
    if msg_name == 'modelV2':
      if smInfo['frameIdExtra'] != frame_id:
        skip_frames.append(frame_id)
    for duration in service_to_durations[service]:
      if duration == 'timestampSof':
        service_blocks[int(frame_id)][service]["Start"].append(float(smInfo[duration]))
      else:
        durations[int(frame_id)][service][duration].append(smInfo[duration]*1e3)
    service_blocks[frame_id][service]["End"].append(float(pubTime))
    next_service = services[services.index(service)+1]
    service_blocks[int(frame_id)][next_service]["Start"].append( float(rcvTime))
  elif "Pipeline end" == jmsg['event']:
    frame_id = translate(jmsg["info"]["logMonoTime"])
    service_blocks[int(frame_id)][services[-1]]["End"].append(float(jmsg["time"]))

# Exclude bad data
if failed_transl > 0:
  del service_blocks[-1]
for frame_id in skip_frames:
  del service_blocks[list(service_blocks.keys()).index(frame_id)]
empty_data = set()
for frame_id in service_blocks.keys():
  for service in services:
    if service not in service_blocks[frame_id]:
      empty_data.add(frame_id)
for frame_id in empty_data:
  del service_blocks[frame_id]

def find_frame_id(time, service):
  for frame_id, blocks in service_blocks.items():
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

# Insert logs in blocks
for jmsg in timestamp_logs:
  service = jmsg['ctx']['daemon']
  event = jmsg['msg']['timestamp']['event']
  time = jmsg['msg']['timestamp']['time']
  frame_id = find_frame_id(time, service) 
  service_blocks[int(frame_id)][service][event].append(float(time))
if 0 in service_blocks:
  del service_blocks[0]

# Print
for frame_id, services in service_blocks.items():
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
  print("Internal durations:")
  for service, events in dict(durations)[frame_id].items():
    print(service)  
    for event, times in dict(events).items():
      print("    ", event, times)  
  print()

print("Num skipped due to failed translations:",failed_transl)
print("Num skipped due to frameId missmatch:",len(skip_frames))
print("Num skipped due to empty data:", len(empty_data))
'''
fig, gnt = plt.subplots()
maxx = max([max([max(events.values()) for events in services.values()]) for services in timestamps.values()])/1e6
gnt.set_xlim(0, 150)
maxy = len(timestamps)
gnt.set_ylim(0, maxy)

avg_times = defaultdict(list)

count = 0
for frame_id, services in timestamps.items():
  t0 = min([min(events.values())for events in services.values()])
  service_bars = []
  event_bars = []
  print(frame_id)
  for service, events in services.items():
    start = min(events.values())
    end = max(events.values())
    #service_bars.append(((start-t0)/1e6, (end-start)/1e6))
    for event, time in events.items():
      t = (time-t0)/1e6
      event_bars.append((t, 0.1))
      avg_times[service+"."+event].append(t)
      print("    ", service+"."+event, t)
  #gnt.broken_barh(service_bars, (count, 0.9), facecolors=("blue"))
  gnt.broken_barh(event_bars, (count, 0.9), facecolors=("black"))
  count+=1
print(tabulate([[event, sum(times)/len(times), max(times), len(times)] for event, times in avg_times.items()], headers=["event", "avg", "max", "len"]))
plt.show(block=True)
'''
