import cereal.messaging as messaging
import matplotlib.pyplot as plt
from collections import defaultdict
from tabulate import tabulate
import matplotlib.patches as mpatches

topics = ['roadCameraState', 'modelV2',  'lateralPlan', 'longitudinalPlan']
frames = defaultdict(lambda: defaultdict(int))

sm = messaging.SubMaster(topics)
for _ in range(20):
    sm.update()
    for topic, is_updated in sm.updated.items():
        if is_updated:
            frames[sm[topic].frameId][topic] = sm.logMonoTime[topic]
            if topic == topics[0]:
                frames[sm[topic].frameId]["start"] = sm[topic].timestampSof
                

for frame, times in frames.items():
    print(frame)
    t0 = times["start"]
    for topic, time in times.items():
        if topic != "start":
            print("     ", topic, (time-t0)/1e6)

