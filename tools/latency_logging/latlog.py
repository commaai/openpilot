import cereal.messaging as messaging
import matplotlib.pyplot as plt
from collections import defaultdict
from tabulate import tabulate
import matplotlib.patches as mpatches

topics = ['roadCameraState', 'wideRoadCameraState', 'driverCameraState', 'modelV2',  'lateralPlan', 'longitudinalPlan', 'carControl']
frames = defaultdict(lambda: defaultdict(int))

sm = messaging.SubMaster(topics)
while len(frames) < 50:
    sm.update()
    for topic, is_updated in sm.updated.items():
        if is_updated:
            if topic == topics[0]:
                frames[sm[topic].frameId]["road_cam_start"] = sm[topic].timestampSof
                frames[sm[topic].frameId]["road_cam_end"] = sm[topic].timestampEof
            if topic == topics[1]:
                frames[sm[topic].frameId]["wide_cam_start"] = sm[topic].timestampSof
                frames[sm[topic].frameId]["wide_cam_end"] = sm[topic].timestampEof
            if topic == topics[2]:
                frames[sm[topic].frameId]["driver_cam_start"] = sm[topic].timestampSof
                frames[sm[topic].frameId]["driver_cam_end"] = sm[topic].timestampEof
            frames[sm[topic].frameId][topic] = sm.logMonoTime[topic]
                
avg_times = defaultdict(list)

fig, gnt = plt.subplots()
gnt.set_xlim(0, 150)
gnt.set_ylim(0, len(frames))
colors = {"road_cam_start":'red', "wide_cam_start":'blue', "driver_cam_start":'green', "road_cam_end":'cyan', "wide_cam_end":'magenta', "driver_cam_end":'yellow', 'roadCameraState':'olive', 'wideRoadCameraState':'steelblue', 'driverCameraState':'deeppink', 'modelV2':'darkseagreen', 'lateralPlan':'coral', 'longitudinalPlan':'brown', 'carControl':'gray'}

count = 0
for frame, times in frames.items():
    t0 = min(times.values())
    events = {}
    for topic, time in times.items():
        #print("     ", topic, (time-t0)/1e6)
        events[((time-t0)/1e6, 0.3)] = colors[topic]
        avg_times[topic].append((time-t0)/1e6)
    gnt.broken_barh(events.keys(), (count, 0.9), facecolors=(events.values()))
    count += 1

print(tabulate([[topic, sum(avg_times[topic])/len(avg_times[topic]), max(avg_times[topic]), len(avg_times[topic])] for topic in colors.keys()], headers=["event", "avg", "max", "len"]))

plt.legend(handles=[mpatches.Patch(color=list(colors.values())[i], label=list(colors.keys())[i]) for i in range(len(colors))])
plt.show(block=True)
