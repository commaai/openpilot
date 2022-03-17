import cereal.messaging as messaging
import matplotlib.pyplot as plt
from collections import defaultdict
from tabulate import tabulate
import matplotlib.patches as mpatches

#TODO: meassure camera time also
# start times cereal
# one already have delay time
# fix so both planning are shown
# commit
# nicer visual
# planner publishes more than just sendcan

topics = ['roadCameraState', 'modelV2',  'lateralPlan', 'longitudinalPlan']
colors = ['red', 'blue', 'green', 'orange']

sm = messaging.SubMaster(topics)
updated = []
started = False
t0=0
size = 100
count = 0
avg_times = defaultdict(list)

fig, gnt = plt.subplots()
gnt.set_xlim(0, 75)
gnt.set_ylim(0, size)

while count < size:
    sm.update()
    times = sm.logMonoTime

    if sm.updated[topics[0]]:
        if not started:
            started = True
            updated = []
            continue

        events = {}
        for topic in updated:
            time = times[topic]/1e6 if topic != topics[0] else t0
            time -= t0
            events[(time, 0.3)] = "tab:"+colors[topics.index(topic)]
            avg_times[topic].append(time)
        gnt.broken_barh(events.keys(), (count, 0.9), facecolors=(events.values()))
        count += 1
        t0 = times[topics[0]]/1e6
        updated=[]
    updated.extend([topic for topic, val in sm.updated.items() if val])

print(tabulate([[topic, sum(t)/len(t), max(t), len(t)] for topic,t in avg_times.items()], headers=["Topic", "avg", "max", "len"]))



plt.legend(handles=[mpatches.Patch(color=colors[i], label=topics[i]) for i in range(len(topics))])
plt.show(block=True)

