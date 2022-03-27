from selfdrive.swaglog import cloudlog
from tools.lib.route import Route
from tools.lib.logreader import LogReader

import json
from collections import defaultdict


timestamps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
translationdict = {}

r = Route("9f583b1d93915c31|2022-03-26--17-25-22") #major changes 
lr = LogReader(r.log_paths()[0])

services = ['camera', 'model', 'planner', 'controls', 'board']

msgq_to_service = {}
msgq_to_service['roadCameraState'] = "camera"
msgq_to_service['wideRoadCameraState'] = "camera"
msgq_to_service['modelV2'] = "model"
msgq_to_service['lateralPlan'] = "planner"
msgq_to_service['longitudinalPlan'] = "planner"
msgq_to_service['sendcan'] = "controls"
msgq_to_service['controlState'] = "controls"

service_to_durations = {}
service_to_durations['camera'] = ['timestampEof', 'timestampSof', 'processingTime']
service_to_durations['model'] = ['timestampEof', 'modelExeccutionTime', 'gpuExecutionTime']
service_to_durations['planner'] = ["solverExcecutionTime"]

service_blocks = defaultdict(dict)

# Build service blocks
for msg in lr:
    if msg.which() == "logMessage" and ("smInfo" in msg or "timestampExtra" in msg):
        msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
        msg = json.loads(msg)
        #make more robust than strings
        if "smInfo" in msg:
            pubTime = msg['timestampExtra']['info']["msgq"]['logMonoTime']
            rcvTime = msg['timestampExtra']['info']["msgq"]['rcvTime']
            service = msgq_to_service[msg['timestampsExtra']['info']['msg_name']]
            smInfo = msg['timestampsExtra']['info']['smInfo']
            durations = {}
            for duration in service_to_durations[service]:
                durations[duration] = smInfo[duration] 
            frame_id = smInfo['frameId'] 
            service_blocks[frame_id][service]["End"] = pubTime
            service_blocks[frame_id][service]["Durations"] = durations
            next_service = services[services.index(service)+1]
            service_blocks[frame_id][next_service]["Start"] = rcvTime
        elif "Pipeline start" in msg:
            frame_id = msg['timestampExtra']["info"]["frameId"]
            time = msg['timestampExtra']["time"]
            service_blocks[frame_id][services[0]]["Start"] = time
        elif "Pipeline end" in msg:
            #TODO: needs translation
            frame_id = msg['timestampExtra']["info"]["frameId"]
            time = msg['timestampExtra']["time"]
            service_blocks[frame_id][services[-1]]["End"] = time 

# Build translation dict
for msg in lr:
  if msg.which() == "logMessage" and "translation" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    jmsg = json.loads(msg)
    logMonoTime = msg['timestampExtra']['info']['logMonoTime']
    time = msg['timestampExtra']["time"]
    #TODO get frame_id from publishing time
    translationdict[logMonoTime] = frame_id

# TODO: Insert logs in blocks
i = 0
for msg in lr:
  if msg.which() == "logMessage" and "timestamp" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}")
    jmsg = json.loads(msg)
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']
    time = jmsg['msg']['timestamp']['time']
    frame_id = jmsg['msg']['timestamp']['frameId']
    if jmsg['msg']['timestamp']['translate']:
        try:
            frame_id = translationdict[frame_id]
        except:
            i+=1
    timestamps[frame_id][service][event].append(time)

print("Num failed translations:",i)

del timestamps[0]

with open('timestamps.json', 'w') as outfile:
    json.dump(timestamps, outfile)
