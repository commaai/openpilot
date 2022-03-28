from selfdrive.swaglog import cloudlog
from tools.lib.route import Route
from tools.lib.logreader import LogReader

import json
from collections import defaultdict



r = Route("9f583b1d93915c31|2022-03-28--13-04-58") 
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
service_to_durations['camerad'] = ['timestampEof', 'timestampSof', 'processingTime']
service_to_durations['modeld'] = ['timestampEof', 'modelExecutionTime', 'gpuExecutionTime']
service_to_durations['plannerd'] = ["solverExecutionTime"]

translationdict = {}

# Build translation dict
for msg in lr:
  if msg.which() == "logMessage" and "translation" in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}").replace("\\", '')
    jmsg = json.loads(msg)['msg']['timestampExtra']
    logMonoTime = jmsg['info']['from']
    frame_id = jmsg['info']['to']
    translationdict[int(logMonoTime)] = int(frame_id)

skip_frames = []
service_blocks = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

failed_transl = 0
def translate(logMono):
    logMono = int(logMono)
    if logMono in translationdict:
        return translationdict[logMono]
    else:
        global failed_transl
        failed_transl += 1
        return 0

# Build service blocks
for msg in lr:
    if msg.which() == "logMessage" and ("smInfo" in msg.logMessage or "timestampExtra" in msg.logMessage):
        msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}").replace("\\", '')
        jmsg = json.loads(msg)['msg']['timestampExtra']
        #make more robust than strings
        if "smInfo" in msg:
            pubTime = jmsg['info']['logMonoTime']
            rcvTime =jmsg['info']['rcvTime']*1e9
            msg_name = jmsg['info']['msg_name']
            service = msgq_to_service[msg_name]
            smInfo = jmsg['info']['smInfo']
            frame_id = 0
            if msg_name== 'sendcan':
                frame_id = translate(pubTime)
            else:
                frame_id = smInfo['frameId'] 
                if msg_name == 'modelV2':
                    if smInfo['frameIdExtra'] != frame_id:
                        skip_frames.append(frame_id)
                for duration in service_to_durations[service]:
                    if duration == 'timestampSof':
                        service_blocks[frame_id][service]["Start"].append(int(smInfo[duration]))
                    else:
                        pass
                        #service_blocks[frame_id][service][duration].append(int(smInfo[duration]))
            service_blocks[frame_id][service]["End"].append(int(pubTime))
            next_service = services[services.index(service)+1]
            service_blocks[frame_id][next_service]["Start"].append( int(rcvTime))
        elif "Pipeline end" in msg:
            frame_id = translate(jmsg["info"]["logMonoTime"])
            time = jmsg["time"]
            service_blocks[frame_id][services[-1]]["End"].append(int(time))

print("Num failed translations:",failed_transl)

def find_frame_id(time, service):
    for frame_id, blocks in service_blocks.items():
        try:
            if min(blocks[service]["Start"]) <= int(time) <= max(blocks[service]["End"]):
                return frame_id
        except:
            continue
    return 0

# Insert logs in blocks
for msg in lr:
  if msg.which() == "logMessage" and "timestamp" in msg.logMessage and "timestampExtra" not in msg.logMessage:
    msg = msg.logMessage.replace("'", '"').replace('"{', "{").replace('}"', "}").replace("\\", '')
    jmsg = json.loads(msg)
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']
    time = jmsg['msg']['timestamp']['time']
    frame_id = find_frame_id(time, service) 
    service_blocks[frame_id][service][event].append(int(time))


for frame_id in skip_frames:
    del service_blocks[list(service_blocks.keys()).index(frame_id)]
print("Skipped due to frameid missmatch:",len(skip_frames))

with open('timestamps.json', 'w') as outfile:
    json.dump(service_blocks, outfile)
