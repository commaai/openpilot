#!/usr/bin/env python3
import argparse
import json
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import sys
from collections import defaultdict

from openpilot.tools.lib.logreader import LogReader

DEMO_ROUTE = "9f583b1d93915c31|2022-05-18--10-49-51--0"

COLORS = ['blue', 'green', 'red', 'yellow', 'orange', 'purple']
PLOT_SERVICES = ['card', 'controlsd']  # , 'boardd']


def plot(lr):
  seen = set()
  aligned = False

  start_time = None
  # dict of services to events per inferred frame
  times = {s: [[]] for s in PLOT_SERVICES}

  first_event = None
  # temp_times = {s: [] for s in PLOT_SERVICES}  # holds only current frame of services

  timestamps = [json.loads(msg.logMessage) for msg in lr if msg.which() == 'logMessage' and 'timestamp' in msg.logMessage]
  # print(timestamps)
  timestamps = sorted(timestamps, key=lambda m: float(m['msg']['timestamp']['time']))

  # closely matches timestamp time
  start_time = next(msg.logMonoTime for msg in lr)

  for jmsg in timestamps:
    if len(times[PLOT_SERVICES[0]]) > 1400:
      continue

    # print()
    # print(msg.logMonoTime)
    time = int(jmsg['msg']['timestamp']['time'])
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']
    # print(jmsg)
    # print(seen)

    if service in PLOT_SERVICES and first_event is None:
      first_event = event

    # Align the best we can; wait for all to be seen and this is the first event
    # TODO: detect first logMessage correctly by keeping track of events before aligned
    aligned = aligned or (all(s in seen for s in PLOT_SERVICES) and event == first_event)
    if not aligned:
      seen.add(service)
      continue

    if service in PLOT_SERVICES:

      # new frame when we've seen this event before
      new_frame = event in {e[1] for e in times[service][-1]}
      if new_frame:
        times[service].append([])

      # print(msg.logMonoTime, jmsg)
      print('new_frame', new_frame)
      times[service][-1].append(((time - start_time) * 1e-6, event))

  points = {"x": [], "y": [], "labels": []}
  height = 0.9
  offsets = [[0, -10 * j] for j in range(len(PLOT_SERVICES))]

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
      ax.broken_barh([sb], (idx - height / 2 - j * 1, height), facecolors=[COLORS[j]], alpha=0.5)  # , offsets=offsets)
    # ax.broken_barh(service_bars, (idx - height / 2, height), facecolors=COLORS, alpha=0.5, offsets=offsets)

  scatter = ax.scatter(points['x'], points['y'], marker='d', edgecolor='black')
  txt = ax.text(0, 0, '', ha='center', fontsize=8, color='red')
  ax.set_xlabel('milliseconds')

  plt.legend(handles=[mpatches.Patch(color=COLORS[i], label=PLOT_SERVICES[i]) for i in range(len(PLOT_SERVICES))])

  def hover(event):
    txt.set_text("")
    status, pts = scatter.contains(event)
    txt.set_visible(status)
    if status:
      pt_idx = pts['ind'][0]
      txt.set_text(f"{points['labels'][pt_idx]} ({points['x'][pt_idx]:0.2f} ms)")
      txt.set_position((event.xdata, event.ydata + 1))
    event.canvas.draw()

  fig.canvas.mpl_connect("motion_notify_event", hover)

  plt.show()
  # plt.pause(1000)
  return times, points


def plot_dist(lr, poll):

  lr = list(lr)

  # carState_service_times = []
  # prev_time = None
  # for msg in lr:
  #   if msg.which() == 'carState':
  #     if prev_time is not None:
  #       carState_service_times.append((msg.logMonoTime - prev_time) * 1e-6)
  #     prev_time = msg.logMonoTime

  # logMonoTime is from logmessaged, not when the timestamp was created. some messages are out of order
  timestamps = [json.loads(msg.logMessage) for msg in tqdm(lr) if msg.which() == 'logMessage' and '"timestamp"' in msg.logMessage]
  timestamps = sorted(timestamps, key=lambda m: float(m['msg']['timestamp']['time']))
  timestamps = [m for m in tqdm(timestamps) if m['ctx']['daemon'] in PLOT_SERVICES]

  initialized = False
  ready = False

  start_card_loop = None
  received_can = None
  state_updated = None
  sent_carState = None
  state_published = None
  sent_carControl = None

  card_e2e_loop_times = []
  card_carInterface_update_times = []
  carState_recv_times = []
  carControl_recv_times = []
  carState_to_carControl_times = []
  card_controls_times = []
  card_loop_times = []

  for jmsg in tqdm(timestamps):
    time = int(jmsg['msg']['timestamp']['time'])
    service = jmsg['ctx']['daemon']
    event = jmsg['msg']['timestamp']['event']

    if event == 'Initialized' and service == 'card':
      initialized = True

    if initialized and event == 'Start card':
      ready = True

    if not ready:
      continue

    if event == 'Start card' and service == 'card':
      if start_card_loop is not None:
        card_e2e_loop_times.append((time - start_card_loop) * 1e-6)
      start_card_loop = time

    elif event == 'Received can' and service == 'card':
      # measuring from this time does not include wait time for can packet, so this measures true card loop time taken
      received_can = time

    elif event == 'State updated' and service == 'card':
      state_updated = time
      card_carInterface_update_times.append((time - received_can) * 1e-6)

    elif event == 'Sent carState' and service == 'card':
      sent_carState = time

    elif event == 'Got carState' and service == 'controlsd':
      # TODO why none
      if sent_carState is not None:
        carState_recv_times.append((time - sent_carState) * 1e-6)

    elif event == 'Logs published' and service == 'controlsd':
      sent_carControl = time

    elif event == 'State published' and service == 'card':
      state_published = time
      if poll:  # only makes sense when polling
        # from carState sent to carControl received
        carControl_recv_times.append((time - sent_carControl) * 1e-6)
        carState_to_carControl_times.append((time - sent_carState) * 1e-6)

    elif event == 'Controls updated' and service == 'card':
      card_controls_times.append((time - state_published) * 1e-6)
      card_loop_times.append((time - received_can) * 1e-6)  # this is time NOT spent waiting for can


  fig, ax = plt.subplots(3)

  fig.suptitle('Polling/waiting on carControl from controlsd' if poll else 'Not polling on carControl')

  ax[0].set_title('cereal communication times')
  ax[0].set_xlim(0, 16)
  sns.histplot(carState_recv_times, kde=True, ax=ax[0],
               label=f'carState->controlsd recv time: \n  minmax: {min(carState_recv_times):0.2f}, {max(carState_recv_times):>5.2f}, ' +
                     f'med: {np.median(carState_recv_times):0.2f}, mean: {np.mean(carState_recv_times):0.2f}, ' +
                     f'95th: {np.percentile(carState_recv_times, 95):0.2f}')
  if poll:
    sns.histplot(carControl_recv_times, kde=True, ax=ax[0],
                 label=f'carControl->card recv time (polling on carControl): \n  minmax: {min(carControl_recv_times):0.2f}, {max(carControl_recv_times):>5.2f}, ' +
                       f'med: {np.median(carControl_recv_times):0.2f}, mean: {np.mean(carControl_recv_times):0.2f}, ' +
                       f'95th: {np.percentile(carControl_recv_times, 95):0.2f}')
  ax[0].legend()

  ax[1].set_title('card loop times')
  ax[1].set_xlim(0, 16)
  if poll:
    sns.histplot(carState_to_carControl_times, kde=True, ax=ax[1],
                 label=f'waiting on carControl (polling on carControl): \n  minmax: {min(carState_to_carControl_times):0.2f}, {max(carState_to_carControl_times):>5.2f}, ' +
                       f'med: {np.median(carState_to_carControl_times):0.2f}, mean: {np.mean(carState_to_carControl_times):0.2f}, ' +
                       f'95th: {np.percentile(carState_to_carControl_times, 95):0.2f}')
  sns.histplot(card_controls_times, kde=True, ax=ax[1], label=f'CI.apply(): \n  minmax: {min(card_controls_times):0.2f}, {max(card_controls_times):>6.2f}, ' +
                                                              f'med: {np.median(card_controls_times):0.2f}, mean: {np.mean(card_controls_times):0.2f}, ' +
                                                              f'95th: {np.percentile(card_controls_times, 95):0.2f}')
  sns.histplot(card_carInterface_update_times, kde=True, ax=ax[1],
               label=f'CI.update(): \n  minmax: {min(card_carInterface_update_times):0.2f}, {max(card_carInterface_update_times):>5.2f}, ' +
                     f'med: {np.median(card_carInterface_update_times):0.2f}, mean: {np.mean(card_carInterface_update_times):0.2f}, ' +
                     f'95th: {np.percentile(card_carInterface_update_times, 95):0.2f}')
  ax[1].legend()

  ax[2].set_title('total card loop time')
  ax[2].set_xlim(0, 16)
  sns.histplot(card_loop_times, kde=True, ax=ax[2], label=f'entire card loop time: \n  minmax: {min(card_loop_times):0.2f}, {max(card_loop_times):>5.2f}, ' +
                                                          f'med: {np.median(card_loop_times):0.2f}, mean: {np.mean(card_loop_times):0.2f}, ' +
                                                          f'95th: {np.percentile(card_loop_times, 95):0.2f}')
  ax[2].legend()
  ax[2].set_xlabel('ms')

  return timestamps, card_loop_times, carState_recv_times, carState_to_carControl_times



if __name__ == "__main__":
  # parser = argparse.ArgumentParser(description="A tool for analyzing openpilot's end-to-end latency",
  #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  # parser.add_argument("route_or_segment_name", nargs='?', help="The route to print")
  #
  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit()
  # args = parser.parse_args()

  # r = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
  # lr = LogReader(r, sort_by_time=True)
  # lr = LogReader('08e4c2a99df165b1/00000016--c3a4ca99ec/0', sort_by_time=True)  # normal
  # lr = LogReader('08e4c2a99df165b1/00000017--e2d24ab118/0', sort_by_time=True)  # polls on carControl
  # lr = LogReader('08e4c2a99df165b1/00000018--cf65e47c24/0', sort_by_time=True)  # polls on carControl, sends it earlier
  # lr = LogReader('08e4c2a99df165b1/00000019--e73e3ab4df/0', sort_by_time=True)  # polls on carControl, more logging

  # lr = LogReader('08e4c2a99df165b1/0000002c--b40eb82d6d/0:-1', sort_by_time=True)  # polls on carControl
  # lr = LogReader('08e4c2a99df165b1/0000002d--ccebe8b617/0:1', sort_by_time=True)  # no poll on carControl
  # lr = LogReader('08e4c2a99df165b1/0000002e--fd98f6603b/:7', sort_by_time=True)  # no poll on carControl (no timestamps)

  POLL = False  # carControl polling or not
  if POLL:
    # lr = LogReader('08e4c2a99df165b1/00000032--2c1d57d894/0', sort_by_time=True)  # carControl poll, w/ reduced timestamps
    # lr = LogReader('08e4c2a99df165b1/00000033--1e2720e55b/0', sort_by_time=True)  # carControl poll, w/ poll flag (FINAL)
    lr = LogReader('08e4c2a99df165b1/00000036--4eb8126f04', sort_by_time=True)  # carControl poll, w/ poll flag & Received can (FINAL v2)
  else:
    # lr = LogReader('08e4c2a99df165b1/00000031--f6f38d1ccf/0', sort_by_time=True)  # no carControl poll, w/ reduced timestamps
    # lr = LogReader('08e4c2a99df165b1/00000035--0abfde9c4a/0', sort_by_time=True)  # no carControl poll, w/ poll flag (FINAL)
    lr = LogReader('08e4c2a99df165b1/00000037--f6294815ac', sort_by_time=True)  # no carControl poll, w/ poll flag & Received can (FINAL v2)

  timestamps = plot_dist(lr, poll=POLL)
  # times, points = plot(lr)
