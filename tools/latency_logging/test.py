import cereal.messaging as messaging
from tools.lib.route import Route
from tools.lib.logreader import LogReader

DEMO_ROUTE = "9f583b1d93915c31|2022-03-30--11-11-53"
r = Route(DEMO_ROUTE)
lr = LogReader(r.log_paths()[0])

MSGQ_TO_SERVICE = {
        'roadCameraState':'camerad',
        'wideRoadCameraState':'camerad',
        'modelV2':'modeld',
        'lateralPlan':'plannerd',
        'longitudinalPlan':'plannerd',
        }
l = set(MSGQ_TO_SERVICE.keys())
i = 0
for msg in lr:
    if msg.which() == 'modelV2':
        print(msg.to_dict()["which"])
        i+=1
        if i > 3:break

