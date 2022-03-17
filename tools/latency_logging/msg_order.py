import cereal.messaging as messaging
from collections import defaultdict
from typing import Dict

topics = ['roadCameraState', 'modelV2',  'lateralPlan', 'longitudinalPlan']
count = defaultdict(int)

sm = messaging.SubMaster(topics)
updated = []
started = False
while 1:
    sm.update()
    if sm.updated[topics[0]]:
        #print((sm['roadCameraState'].timestampEof-sm['roadCameraState'].timestampSof)/1e6)
        if not started:
            started = True
            updated = ['0']
            continue
        count["".join(updated)] += 1
        for key, val in count.items():
            print(", ".join([topics[int(c)] for c in key]), val)
        updated = []
        print("-----------")
    updated.extend([str(topics.index(topic)) for topic, val in sm.updated.items() if val])

