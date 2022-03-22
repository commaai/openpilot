import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from tabulate import tabulate


json_file = open('timestamps.json')
timestamps = json.load(json_file)


for frame_id, services in timestamps.items():
    t0 = min([min(events.values())for events in services.values()])
    print(frame_id)
    d = defaultdict( lambda: ("",""))
    for service, events in services.items():
        for event, time in events.items():
            time = (time-t0)/1e6
            d[time] = (service, event)
    s = sorted(d.items())
    print(tabulate([[item[1][0], item[1][1], item[0]] for item in s], headers=["service", "event", "time"]))
    print()

exit()
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
