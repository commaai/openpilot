#!/usr/bin/env python3
import argparse
import json
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mpld3
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

from openpilot.tools.lib.logreader import LogReader

DEMO_ROUTE = "9f583b1d93915c31|2022-05-18--10-49-51--0"

# SERVICES = ['camerad', 'modeld', 'plannerd', 'controlsd', 'card', 'boardd']
MONOTIME_KEYS = ['modelMonoTime', 'lateralPlanMonoTime', 'logMonoTime']
MSGQ_TO_SERVICE = {
  'roadCameraState': 'camerad',
  'wideRoadCameraState': 'camerad',
  'modelV2': 'modeld',
  'longitudinalPlan': 'plannerd',
  'carState': 'card',
  'sendcan': 'card',
  'controlsState': 'controlsd'
}
SERVICE_TO_DURATIONS = {
  'camerad': ['processingTime'],
  'modeld': ['modelExecutionTime', 'gpuExecutionTime'],
  'plannerd': ['solverExecutionTime'],
}


def read_logs(lr):
  data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  mono_to_frame = {}
  frame_mismatches = []
  frame_id_fails = 0
  latest_sendcan_monotime = 0
  for msg in lr:
    if msg.which() == 'sendcan':
      latest_sendcan_monotime = msg.logMonoTime
      continue

    if msg.which() in MSGQ_TO_SERVICE:
      service = MSGQ_TO_SERVICE[msg.which()]
      msg_obj = getattr(msg, msg.which())

      frame_id = -1
      if hasattr(msg_obj, "frameId"):
        frame_id = msg_obj.frameId
      else:
        continue_outer = False
        for key in MONOTIME_KEYS:
          # if msg.which() == 'controlsState':
          #   print('hi')
          if hasattr(msg_obj, key):
            if msg.which() == 'carState':
              raise Exception
            if getattr(msg_obj, key) == 0:
              # Filter out controlsd messages which arrive before the camera loop
              continue_outer = True
            elif getattr(msg_obj, key) in mono_to_frame:
              frame_id = mono_to_frame[getattr(msg_obj, key)]
        if continue_outer:
          continue
      if frame_id == -1:
        frame_id_fails += 1
        continue
      mono_to_frame[msg.logMonoTime] = frame_id
      data['timestamp'][frame_id][service].append((msg.which() + " published", msg.logMonoTime))

      next_service = SERVICES[SERVICES.index(service) + 1]
      if not data['start'][frame_id][next_service]:
        data['start'][frame_id][next_service] = msg.logMonoTime
      data['end'][frame_id][service] = msg.logMonoTime

      if service in SERVICE_TO_DURATIONS:
        for duration in SERVICE_TO_DURATIONS[service]:
          data['duration'][frame_id][service].append((msg.which() + "." + duration, getattr(msg_obj, duration)))

      if service == SERVICES[0]:
        data['timestamp'][frame_id][service].append((msg.which() + " start of frame", msg_obj.timestampSof))
        if not data['start'][frame_id][service]:
          data['start'][frame_id][service] = msg_obj.timestampSof
      elif msg.which() == 'carState':
        # Sendcan is published before carState, but the frameId is retrieved in CS
        data['timestamp'][frame_id][service].append(("sendcan published", latest_sendcan_monotime))
      elif msg.which() == 'modelV2':
        if msg_obj.frameIdExtra != frame_id:
          frame_mismatches.append(frame_id)

  if frame_id_fails > 20:
    print("Warning, many frameId fetch fails", frame_id_fails)
  if len(frame_mismatches) > 20:
    print("Warning, many frame mismatches", len(frame_mismatches))
  return (data, frame_mismatches)


# This is not needed in 3.10 as a "key" parameter is added to bisect
class KeyifyList:
  def __init__(self, inner, key):
    self.inner = inner
    self.key = key

  def __len__(self):
    return len(self.inner)

  def __getitem__(self, k):
    return self.key(self.inner[k])


def find_frame_id(time, service, start_times, end_times):
  left = bisect_left(KeyifyList(list(start_times.items()),
                                lambda x: x[1][service] if x[1][service] else -1), time) - 1
  right = bisect_right(KeyifyList(list(end_times.items()),
                                  lambda x: x[1][service] if x[1][service] else float("inf")), time)
  return left, right


def find_t0(start_times, frame_id=-1):
  frame_id = frame_id if frame_id > -1 else min(start_times.keys())
  m = max(start_times.keys())
  while frame_id <= m:
    for service in SERVICES:
      if start_times[frame_id][service]:
        return start_times[frame_id][service]
    frame_id += 1
  raise Exception('No start time has been set')


def insert_cloudlogs(lr, timestamps, start_times, end_times):
  # at least one cloudlog must be made in controlsd

  t0 = find_t0(start_times)
  failed_inserts = 0
  latest_controls_frameid = 0
  for msg in lr:
    if msg.which() == "logMessage":
      jmsg = json.loads(msg.logMessage)
      if "timestamp" in jmsg['msg']:
        time = int(jmsg['msg']['timestamp']['time'])
        service = jmsg['ctx']['daemon']
        event = jmsg['msg']['timestamp']['event']
        if time < t0:
          # Filter out controlsd messages which arrive before the camera loop
          continue

        if "frame_id" in jmsg['msg']['timestamp']:
          timestamps[int(jmsg['msg']['timestamp']['frame_id'])][service].append((event, time))
          continue

        if service == "boardd":
          timestamps[latest_controls_frameid][service].append((event, time))
          end_times[latest_controls_frameid][service] = time
        else:
          frame_id = find_frame_id(time, service, start_times, end_times)
          if frame_id:
            if frame_id[0] != frame_id[1]:
              event += " (warning: ambiguity)"
            frame_id = frame_id[0]
            if service == 'controlsd':
              latest_controls_frameid = frame_id
            timestamps[frame_id][service].append((event, time))
          else:
            failed_inserts += 1

  if latest_controls_frameid == 0:
    print("Warning: failed to bind boardd logs to a frame ID. Add a timestamp cloudlog in controlsd.")
  elif failed_inserts > len(timestamps):
    print(f"Warning: failed to bind {failed_inserts} cloudlog timestamps to a frame ID")


def print_timestamps(timestamps, durations, start_times, relative):
  t0 = find_t0(start_times)
  for frame_id in timestamps.keys():
    if frame_id > 50:
      break
    print('=' * 80)
    print("Frame ID:", frame_id)
    if relative:
      t0 = find_t0(start_times, frame_id)

    for service in SERVICES:
      print("  " + service)
      events = timestamps[frame_id][service]
      for event, time in sorted(events, key=lambda x: x[1]):
        print("    " + '%-53s%-53s' % (event, str((time - t0) / 1e6)))
      for event, time in durations[frame_id][service]:
        print("    " + '%-53s%-53s' % (event, str(time * 1000)))


def graph_timestamps(timestamps, start_times, end_times, relative, offset_services=False, title=""):
  # mpld3 doesn't convert properly to D3 font sizes
  plt.rcParams.update({'font.size': 18})

  t0 = find_t0(start_times)
  fig, ax = plt.subplots()
  ax.set_xlim(0, 130 if relative else 750)
  ax.set_ylim(0, 17)
  ax.set_xlabel('Time (milliseconds)')
  colors = ['blue', 'green', 'red', 'yellow', 'orange', 'purple']
  offsets = [[0, -5 * j] for j in range(len(SERVICES))] if offset_services else None
  height = 0.3 if offset_services else 0.9
  assert len(colors) == len(SERVICES), 'Each service needs a color'

  points = {"x": [], "y": [], "labels": []}
  for i, (frame_id, services) in enumerate(timestamps.items()):
    if relative:
      t0 = find_t0(start_times, frame_id)
    service_bars = []
    for service, events in services.items():
      if start_times[frame_id][service] and end_times[frame_id][service]:
        start = start_times[frame_id][service]
        end = end_times[frame_id][service]
        service_bars.append(((start - t0) / 1e6, (end - start) / 1e6))
        for event in events:
          points['x'].append((event[1] - t0) / 1e6)
          points['y'].append(i)
          points['labels'].append(event[0])
    ax.broken_barh(service_bars, (i - height / 2, height), facecolors=(colors), alpha=0.5, offsets=offsets)

  scatter = ax.scatter(points['x'], points['y'], marker='d', edgecolor='black')
  tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=points['labels'])
  mpld3.plugins.connect(fig, tooltip)

  plt.title(title)
  # Set size relative window size is not trivial: https://github.com/mpld3/mpld3/issues/65
  fig.set_size_inches(18, 9)
  plt.legend(handles=[mpatches.Patch(color=colors[i], label=SERVICES[i]) for i in range(len(SERVICES))])
  return fig


def get_timestamps(lr):
  lr = list(lr)
  data, frame_mismatches = read_logs(lr)
  insert_cloudlogs(lr, data['timestamp'], data['start'], data['end'])
  return data, frame_mismatches


def plot(lr):
  PLOT_SERVICES = ['card', 'controlsd']

  seen = set()
  aligned = False

  start_time = None
  # dict of services to events per inferred frame
  times = {s: [[]] for s in PLOT_SERVICES}

  timestamps = [json.loads(msg.logMessage) for msg in lr if msg.which() == 'logMessage' and 'timestamp' in msg.logMessage]
  print(timestamps)
  timestamps = sorted(timestamps, key=lambda m: float(m['msg']['timestamp']['time']))

  # closely matches timestamp time
  start_time = next(msg.logMonoTime for msg in lr)

  for jmsg in timestamps:
    if len(times[PLOT_SERVICES[0]]) > 400:
      continue

    # print()
    # print(msg.logMonoTime)
    time = int(jmsg['msg']['timestamp']['time'])
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']

    # Align the best we can; all seen and this is the first service
    aligned = aligned or (all(s in seen for s in PLOT_SERVICES) and service == PLOT_SERVICES[0])
    if not aligned:
      seen.add(service)
      continue
    # if not all(s in seen for s in PLOT_SERVICES):
    #   continue

    if service in PLOT_SERVICES:

      # new frame when we've seen this event before
      new_frame = event in {e[1] for e in times[service][-1]}
      if new_frame:
        times[service].append([])

      # print(msg.logMonoTime, jmsg)
      print('new_frame', new_frame)
      times[service][-1].append(((time - start_time) * 1e-6, event))

  # for msg in lr:
  #   if len(times[PLOT_SERVICES[0]]) > 100:
  #     continue
  #   if start_time is None:
  #     # closely matches timestamp time
  #     start_time = msg.logMonoTime
  #
  #   if msg.which() == 'logMessage':
  #     # print()
  #     # print(msg.logMonoTime)
  #     jmsg = json.loads(msg.logMessage)
  #     if "timestamp" in jmsg['msg']:
  #       time = int(jmsg['msg']['timestamp']['time'])
  #       service = jmsg['ctx']['daemon']
  #       event = jmsg['msg']['timestamp']['event']
  #
  #       # Align the best we can; all seen and this is the first service
  #       aligned = aligned or (all(s in seen for s in PLOT_SERVICES) and service == PLOT_SERVICES[0])
  #       if not aligned:
  #         seen.add(service)
  #         continue
  #       # if not all(s in seen for s in PLOT_SERVICES):
  #       #   continue
  #
  #       if service in PLOT_SERVICES:
  #
  #         # new frame when we've seen this event before
  #         new_frame = event in {e[1] for e in times[service][-1]}
  #         if new_frame:
  #           times[service].append([])
  #
  #         print(msg.logMonoTime, jmsg)
  #         print('new_frame', new_frame)
  #         times[service][-1].append(((time - start_time) * 1e-6, event))

  points = {"x": [], "y": [], "labels": []}
  colors = ['blue', 'green']
  offset_services = True
  height = 0.9 if offset_services else 0.9
  offsets = [[0, -10 * j] for j in range(len(PLOT_SERVICES))] if offset_services else None

  fig, ax = plt.subplots()

  for idx, service_times in enumerate(zip(*times.values())):
    print()
    print('idx', idx)
    service_bars = []
    for j, (service, frame_times) in enumerate(zip(times.keys(), service_times)):
      if idx + 1 == len(times[service]):
        break
      print(service, frame_times)
      start = frame_times[0][0]
      # use the first event time from next frame
      end = times[service][idx + 1][0][0]  # frame_times[-1][0]
      print('start, end', start, end)
      service_bars.append((start, end - start))
      for event in frame_times:
        points['x'].append(event[0])
        points['y'].append(idx - j * 1)
        points['labels'].append(event[1])
    print(service_bars)

    # offset = offset_services
    # offset each service
    for j, sb in enumerate(service_bars):
      ax.broken_barh([sb], (idx - height / 2 - j * 1, height), facecolors=[colors[j]], alpha=0.5)#, offsets=offsets)
    # ax.broken_barh(service_bars, [(idx - height / 2 - j * 5, height - j * 5) for j in range(len(service_bars))], facecolors=(colors), alpha=0.5)#, offsets=offsets)



  scatter = ax.scatter(points['x'], points['y'], marker='d', edgecolor='black')
  # for lbl, x, y in zip(points['labels'], points['x'], points['y']):
  #   ax.annotate(lbl, (x, y))

  # tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=points['labels'])
  # mpld3.plugins.connect(fig, tooltip)


  plt.legend(handles=[mpatches.Patch(color=colors[i], label=PLOT_SERVICES[i]) for i in range(len(PLOT_SERVICES))])


  # plt.scatter([t[0] for t in times], [t[1] for t in times], marker='d', edgecolor='black')
  ax.set_xlabel('milliseconds')

  plt.show()
  # plt.pause(1000)
  return times, points


if __name__ == "__main__":
  # parser = argparse.ArgumentParser(description="A tool for analyzing openpilot's end-to-end latency",
  #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # # parser.add_argument("--relative", action="store_true", help="Make timestamps relative to the start of each frame")
  # parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  # # parser.add_argument("--plot", action="store_true", help="If a plot should be generated")
  # # parser.add_argument("--offset", action="store_true", help="Vertically offset service to better visualize overlap")
  # parser.add_argument("route_or_segment_name", nargs='?', help="The route to print")
  #
  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit()
  # args = parser.parse_args()
  #
  # r = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
  # lr = LogReader('08e4c2a99df165b1/00000016--c3a4ca99ec/0', sort_by_time=True)
  lr = LogReader('08e4c2a99df165b1/00000017--e2d24ab118/0', sort_by_time=True)  # polls on carControl
  lr = LogReader('08e4c2a99df165b1/00000018--cf65e47c24/0', sort_by_time=True)  # polls on carControl, sends it earlier

  times, points = plot(lr)


  # data, _ = get_timestamps(lr)
  # print_timestamps(data['timestamp'], data['duration'], data['start'], args.relative)
  # if args.plot:
  #   mpld3.show(graph_timestamps(data['timestamp'], data['start'], data['end'], args.relative, offset_services=args.offset, title=r))
