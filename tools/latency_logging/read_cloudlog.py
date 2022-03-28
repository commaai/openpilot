from selfdrive.swaglog import cloudlog
from tools.lib.route import Route
from tools.lib.logreader import LogReader

import json
from collections import defaultdict


timestamps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
translationdict = {}

r = Route("9f583b1d93915c31|2022-03-28--10-21-49") 
lr = LogReader(r.log_paths()[0])

services = ['camerad', 'modeld', 'plannerd', 'controlsd', 'boardd']

msgq_to_service = {}
msgq_to_service['roadCameraState'] = "camerad"
msgq_to_service['wideRoadCameraState'] = "camerad"
msgq_to_service['modelV2'] = "modeld"
msgq_to_service['lateralPlan'] = "plannerd"
msgq_to_service['longitudinalPlan'] = "plannerd"
msgq_to_service['sendcan'] = "controlsd"
msgq_to_service['controlsState'] = "controlsd"

service_to_durations = defaultdict(list)
service_to_durations['camerad'] = ['timestampEof', 'timestampSof', 'processingTime']
service_to_durations['modeld'] = ['timestampEof', 'modelExeccutionTime', 'gpuExecutionTime']
service_to_durations['plannerd'] = ["solverExcecutionTime"]

service_blocks = defaultdict(dict)

# Build translation dict
for msg in lr:
  if msg.which() == "logMessage" and "translation" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    jmsg = json.loads(msg)['msg']['timestampExtra']
    logMonoTime = jmsg['info']['from']
    frame_id = jmsg['info']['to']
    translationdict[logMonoTime] = frame_id

skip_frames = []

# Build service blocks
for msg in lr:
    if msg.which() == "logMessage" and ("smInfo" in msg.logMessage or "timestampExtra" in msg.logMessage):
        msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
        jmsg = json.loads(msg)['msg']['timestampExtra']
        #make more robust than strings
        if "smInfo" in msg:
            pubTime = jmsg['info']['logMonoTime']
            rcvTime =jmsg['info']['rcvTime']*1e9
            msg_name = jmsg['info']['msg_name']
            service = msgq_to_service[msg_name]
            smInfo = jmsg['info']['smInfo']
            if msg_name== 'sendcan':
                frame_id = translationdict[pubTime]
                service_blocks[frame_id][service]["End"] = pubTime
                next_service = services[services.index(service)+1]
                service_blocks[frame_id][next_service]["Start"] = rcvTime
            else:
                durations = {}
                for duration in service_to_durations[service]:
                    durations[duration] = smInfo[duration] 
                frame_id = 0
                try:
                    frame_id = smInfo['frameId'] 
                except:
                    print(msg_name)
                if smInfo['name'] == 'modelV2':
                    if smInfo['framIdExtra'] != frame_id:
                        skip_frames.append(frame_id)
                service_blocks[frame_id][service]["End"] = pubTime
                service_blocks[frame_id][service]["Durations"] = durations
                next_service = services[services.index(service)+1]
                service_blocks[frame_id][next_service]["Start"] = rcvTime
        elif "Pipeline start" in msg:
            frame_id = jmsg["info"]["frameId"]
            time = jmsg["time"]
            service_blocks[frame_id][services[0]]["Start"] = time
        elif "Pipeline end" in msg:
            frame_id = translationdict[jmsg["info"]["logMonoTime"]]
            time = jmsg["time"]
            service_blocks[frame_id][services[-1]]["End"] = time 

def find_frame_id(time, service):
    for frame_id, blocks in service_blocks.items():
        if blocks[service]["Start"] <= time <= blocks[service]["End"]:
            return frame_id
    return 0

# Insert logs in blocks
for msg in lr:
  if msg.which() == "logMessage" and "timestamp" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    jmsg = json.loads(msg)
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']
    time = jmsg['msg']['timestamp']['time']
    frame_id = find_frame_id(time, service) 
    timestamps[frame_id][service][event].append(time)

print("Num failed translations:",i)

del timestamps[0]
for frame_id in skip_frames:
    del timestamps[timestamps.keys().index(frame_id)]

with open('timestamps.json', 'w') as outfile:
    json.dump(timestamps, outfile)
