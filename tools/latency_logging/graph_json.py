import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from tabulate import tabulate


json_file = open('timestamps.json')
timestamps = json.load(json_file)


fig, gnt = plt.subplots()
maxx = max([max([max(events.values()) for events in services.values()]) for services in timestamps.values()])/1e6
gnt.set_xlim(0, maxx)
maxy = len(timestamps)
gnt.set_ylim(0, maxy)

avg_times = defaultdict(list)

count = 0
for frame_id, services in timestamps.items():
    t0 = min([min(events.values())for events in services.values()])
    service_bars = []
    event_bars = []
    for service, events in services.items():
        start = min(events.values())
        end = max(events.values())
        service_bars.append(((start-t0)/1e6, (end-start)/1e6))
        for event, time in events.items():
            event_bars.append(((time-t0)/1e6, maxx/300))
            avg_times[service+"."+event].append((time-t0)/1e6)
    gnt.broken_barh(service_bars, (count, 0.9), facecolors=("blue"))
    gnt.broken_barh(event_bars, (count, 0.9), facecolors=("black"))
    count+=1
print(tabulate([[event, sum(times)/len(times), max(times), len(times)] for event, times in avg_times.items()], headers=["event", "avg", "max", "len"]))
plt.show(block=True)
