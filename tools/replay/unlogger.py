#!/usr/bin/env python3
import argparse
import os
import sys
import zmq
import time
import signal
import multiprocessing
from uuid import uuid4
from collections import namedtuple
from collections import deque
from datetime import datetime

from cereal import log as capnp_log
from cereal.services import service_list
from cereal.messaging import pub_sock, MultiplePublishersError
from cereal.visionipc.visionipc_pyx import VisionIpcServer, VisionStreamType  # pylint: disable=no-name-in-module, import-error
from common import realtime
from common.transformations.camera import eon_f_frame_size, tici_f_frame_size

from tools.lib.kbhit import KBHit
from tools.lib.logreader import MultiLogIterator
from tools.lib.route import Route
from tools.lib.framereader import rgb24toyuv420
from tools.lib.route_framereader import RouteFrameReader

# Commands.
SetRoute = namedtuple("SetRoute", ("name", "start_time", "data_dir"))
SeekAbsoluteTime = namedtuple("SeekAbsoluteTime", ("secs",))
SeekRelativeTime = namedtuple("SeekRelativeTime", ("secs",))
TogglePause = namedtuple("TogglePause", ())
StopAndQuit = namedtuple("StopAndQuit", ())
VIPC_RGB = "rgb"
VIPC_YUV = "yuv"


class UnloggerWorker(object):
  def __init__(self):
    self._frame_reader = None
    self._cookie = None
    self._readahead = deque()

  def run(self, commands_address, data_address, pub_types):
    zmq.Context._instance = None
    commands_socket = zmq.Context.instance().socket(zmq.PULL)
    commands_socket.connect(commands_address)

    data_socket = zmq.Context.instance().socket(zmq.PUSH)
    data_socket.connect(data_address)

    poller = zmq.Poller()
    poller.register(commands_socket, zmq.POLLIN)

    # We can't publish frames without roadEncodeIdx, so add when it's missing.
    if "roadCameraState" in pub_types:
      pub_types["roadEncodeIdx"] = None

    # gc.set_debug(gc.DEBUG_LEAK | gc.DEBUG_OBJECTS | gc.DEBUG_STATS | gc.DEBUG_SAVEALL |
    # gc.DEBUG_UNCOLLECTABLE)

    # TODO: WARNING pycapnp leaks memory all over the place after unlogger runs for a while, gc
    # pauses become huge because there are so many tracked objects solution will be to switch to new
    # cython capnp
    try:
      route = None
      while True:
        while poller.poll(0.) or route is None:
          cookie, cmd = commands_socket.recv_pyobj()
          route = self._process_commands(cmd, route, pub_types)

        # **** get message ****
        self._read_logs(cookie, pub_types)
        self._send_logs(data_socket)
    finally:
      if self._frame_reader is not None:
        self._frame_reader.close()
      data_socket.close()
      commands_socket.close()

  def _read_logs(self, cookie, pub_types):
    fullHEVC = capnp_log.EncodeIndex.Type.fullHEVC
    lr = self._lr
    while len(self._readahead) < 1000:
      route_time = lr.tell()
      msg = next(lr)
      typ = msg.which()
      if typ not in pub_types:
        continue

      # **** special case certain message types ****
      if typ == "roadEncodeIdx" and msg.roadEncodeIdx.type == fullHEVC:
        # this assumes the roadEncodeIdx always comes before the frame
        self._frame_id_lookup[
          msg.roadEncodeIdx.frameId] = msg.roadEncodeIdx.segmentNum, msg.roadEncodeIdx.segmentId
        #print "encode", msg.roadEncodeIdx.frameId, len(self._readahead), route_time
      self._readahead.appendleft((typ, msg, route_time, cookie))

  def _send_logs(self, data_socket):
    while len(self._readahead) > 500:
      typ, msg, route_time, cookie = self._readahead.pop()
      smsg = msg.as_builder()

      if typ == "roadCameraState":
        frame_id = msg.roadCameraState.frameId

        # Frame exists, make sure we have a framereader.
        # load the frame readers as needed
        s1 = time.time()
        try:
          img = self._frame_reader.get(frame_id, pix_fmt="rgb24")
        except Exception:
          img = None

        fr_time = time.time() - s1
        if fr_time > 0.05:
          print("FRAME(%d) LAG -- %.2f ms" % (frame_id, fr_time*1000.0))

        if img is not None:

          extra = (smsg.roadCameraState.frameId, smsg.roadCameraState.timestampSof, smsg.roadCameraState.timestampEof)

          # send YUV frame
          if os.getenv("YUV") is not None:
            img_yuv = rgb24toyuv420(img)
            data_socket.send_pyobj((cookie, VIPC_YUV, msg.logMonoTime, route_time, extra), flags=zmq.SNDMORE)
            data_socket.send(img_yuv.flatten().tobytes(), copy=False)

          img = img[:, :, ::-1]  # Convert RGB to BGR, which is what the camera outputs
          img = img.flatten()
          bts = img.tobytes()

          smsg.roadCameraState.image = bts

          # send RGB frame
          data_socket.send_pyobj((cookie, VIPC_RGB, msg.logMonoTime, route_time, extra), flags=zmq.SNDMORE)
          data_socket.send(bts, copy=False)

      data_socket.send_pyobj((cookie, typ, msg.logMonoTime, route_time), flags=zmq.SNDMORE)
      data_socket.send(smsg.to_bytes(), copy=False)

  def _process_commands(self, cmd, route, pub_types):
    seek_to = None
    if route is None or (isinstance(cmd, SetRoute) and route.name != cmd.name):
      seek_to = cmd.start_time
      route = Route(cmd.name, cmd.data_dir)
      self._lr = MultiLogIterator(route.log_paths(), wraparound=True)
      if self._frame_reader is not None:
        self._frame_reader.close()
      if "roadCameraState" in pub_types or "roadEncodeIdx" in pub_types:
        # reset frames for a route
        self._frame_id_lookup = {}
        self._frame_reader = RouteFrameReader(
          route.camera_paths(), None, self._frame_id_lookup, readahead=True)

    # always reset this on a seek
    if isinstance(cmd, SeekRelativeTime):
      seek_to = self._lr.tell() + cmd.secs
    elif isinstance(cmd, SeekAbsoluteTime):
      seek_to = cmd.secs
    elif isinstance(cmd, StopAndQuit):
      exit()

    if seek_to is not None:
      print("seeking", seek_to)
      if not self._lr.seek(seek_to):
        print("Can't seek: time out of bounds")
      else:
        next(self._lr)   # ignore one
    return route

def _get_address_send_func(address):
  sock = pub_sock(address)
  return sock.send

def _get_vipc_server(length):
  sizes = {3 * w * h: (w, h) for (w, h) in [tici_f_frame_size, eon_f_frame_size]} # RGB
  sizes.update({(3 * w * h) / 2: (w, h) for (w, h) in [tici_f_frame_size, eon_f_frame_size]}) # YUV

  w, h = sizes[length]

  vipc_server = VisionIpcServer("camerad")
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_RGB_BACK, 4, True, w, h)
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_YUV_BACK, 40, False, w, h)
  vipc_server.start_listener()
  return vipc_server

def unlogger_thread(command_address, forward_commands_address, data_address, run_realtime,
                    address_mapping, publish_time_length, bind_early, no_loop, no_visionipc):
  # Clear context to avoid problems with multiprocessing.
  zmq.Context._instance = None
  context = zmq.Context.instance()

  command_sock = context.socket(zmq.PULL)
  command_sock.bind(command_address)

  forward_commands_socket = context.socket(zmq.PUSH)
  forward_commands_socket.bind(forward_commands_address)

  data_socket = context.socket(zmq.PULL)
  data_socket.bind(data_address)

  # Set readahead to a reasonable number.
  data_socket.setsockopt(zmq.RCVHWM, 10000)

  poller = zmq.Poller()
  poller.register(command_sock, zmq.POLLIN)
  poller.register(data_socket, zmq.POLLIN)

  if bind_early:
    send_funcs = {
      typ: _get_address_send_func(address)
      for typ, address in address_mapping.items()
    }

    # Give subscribers a chance to connect.
    time.sleep(0.1)
  else:
    send_funcs = {}

  start_time = float("inf")
  printed_at = 0
  generation = 0
  paused = False
  reset_time = True
  prev_msg_time = None
  vipc_server = None

  while True:
    evts = dict(poller.poll())
    if command_sock in evts:
      cmd = command_sock.recv_pyobj()
      if isinstance(cmd, TogglePause):
        paused = not paused
        if paused:
          poller.modify(data_socket, 0)
        else:
          poller.modify(data_socket, zmq.POLLIN)
      else:
        # Forward the command the the log data thread.
        # TODO: Remove everything on data_socket.
        generation += 1
        forward_commands_socket.send_pyobj((generation, cmd))
        if isinstance(cmd, StopAndQuit):
          return

      reset_time = True
    elif data_socket in evts:
      msg_generation, typ, msg_time, route_time, *extra = data_socket.recv_pyobj(flags=zmq.RCVMORE)
      msg_bytes = data_socket.recv()
      if msg_generation < generation:
        # Skip packets.
        continue

      if no_loop and prev_msg_time is not None and prev_msg_time > msg_time + 1e9:
        generation += 1
        forward_commands_socket.send_pyobj((generation, StopAndQuit()))
        return
      prev_msg_time = msg_time

      msg_time_seconds = msg_time * 1e-9
      if reset_time:
        msg_start_time = msg_time_seconds
        real_start_time = realtime.sec_since_boot()
        start_time = min(start_time, msg_start_time)
        reset_time = False

      if publish_time_length and msg_time_seconds - start_time > publish_time_length:
        generation += 1
        forward_commands_socket.send_pyobj((generation, StopAndQuit()))
        return

      # Print time.
      if abs(printed_at - route_time) > 5.:
        print("at", route_time)
        printed_at = route_time

      if typ not in send_funcs and typ not in [VIPC_RGB, VIPC_YUV]:
        if typ in address_mapping:
          # Remove so we don't keep printing warnings.
          address = address_mapping.pop(typ)
          try:
            print("binding", typ)
            send_funcs[typ] = _get_address_send_func(address)
          except Exception as e:
            print(f"couldn't replay {typ}: {e}")
            continue
        else:
          # Skip messages that we are not registered to publish.
          continue

      # Sleep as needed for real time playback.
      if run_realtime:
        msg_time_offset = msg_time_seconds - msg_start_time
        real_time_offset = realtime.sec_since_boot() - real_start_time
        lag = msg_time_offset - real_time_offset
        if lag > 0 and lag < 30:  # a large jump is OK, likely due to an out of order segment
          if lag > 1:
            print("sleeping for", lag)
          time.sleep(lag)
        elif lag < -1:
          # Relax the real time schedule when we slip far behind.
          reset_time = True

      # Send message.
      try:
        if typ in [VIPC_RGB, VIPC_YUV]:
          if not no_visionipc:
            if vipc_server is None:
              vipc_server = _get_vipc_server(len(msg_bytes))

            i, sof, eof = extra[0]
            stream = VisionStreamType.VISION_STREAM_RGB_BACK if typ == VIPC_RGB else VisionStreamType.VISION_STREAM_YUV_BACK
            vipc_server.send(stream, msg_bytes, i, sof, eof)
        else:
          send_funcs[typ](msg_bytes)
      except MultiplePublishersError:
        del send_funcs[typ]

def timestamp_to_s(tss):
  return time.mktime(datetime.strptime(tss, '%Y-%m-%d--%H-%M-%S').timetuple())

def absolute_time_str(s, start_time):
  try:
    # first try if it's a float
    return float(s)
  except ValueError:
    # now see if it's a timestamp
    return timestamp_to_s(s) - start_time

def _get_address_mapping(args):
  if args.min is not None:
    services_to_mock = [
      'deviceState', 'can', 'pandaState', 'sensorEvents', 'gpsNMEA', 'roadCameraState', 'roadEncodeIdx',
      'modelV2', 'liveLocation',
    ]
  elif args.enabled is not None:
    services_to_mock = args.enabled
  else:
    services_to_mock = service_list.keys()

  address_mapping = {service_name: service_name for service_name in services_to_mock}
  address_mapping.update(dict(args.address_mapping))

  for k in args.disabled:
    address_mapping.pop(k, None)

  non_services = set(address_mapping) - set(service_list)
  if non_services:
    print(f"WARNING: Unknown services {list(non_services)}")

  return address_mapping

def keyboard_controller_thread(q, route_start_time):
  print("keyboard waiting for input")
  kb = KBHit()
  while 1:
    c = kb.getch()
    if c == 'm':  # Move forward by 1m
      q.send_pyobj(SeekRelativeTime(60))
    elif c == 'M':  # Move backward by 1m
      q.send_pyobj(SeekRelativeTime(-60))
    elif c == 's':  # Move forward by 10s
      q.send_pyobj(SeekRelativeTime(10))
    elif c == 'S':  # Move backward by 10s
      q.send_pyobj(SeekRelativeTime(-10))
    elif c == 'G':  # Move backward by 10s
      q.send_pyobj(SeekAbsoluteTime(0.))
    elif c == "\x20":  # Space bar.
      q.send_pyobj(TogglePause())
    elif c == "\n":
      try:
        seek_time_input = input('time: ')
        seek_time = absolute_time_str(seek_time_input, route_start_time)

        # If less than 60, assume segment number
        if seek_time < 60:
          seek_time *= 60

        q.send_pyobj(SeekAbsoluteTime(seek_time))
      except Exception as e:
        print(f"Time not understood: {e}")

def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Mock openpilot components by publishing logged messages.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("route_name", type=(lambda x: x.replace("#", "|")), nargs="?",
                      help="The route whose messages will be published.")
  parser.add_argument("data_dir", nargs='?', default=os.getenv('UNLOGGER_DATA_DIR'),
          help="Path to directory in which log and camera files are located.")

  parser.add_argument("--no-loop", action="store_true", help="Stop at the end of the replay.")

  def key_value_pair(x):
    return x.split("=")

  parser.add_argument("address_mapping", nargs="*", type=key_value_pair,
      help="Pairs <service>=<zmq_addr> to publish <service> on <zmq_addr>.")

  def comma_list(x):
    return x.split(",")

  to_mock_group = parser.add_mutually_exclusive_group()
  to_mock_group.add_argument("--min", action="store_true", default=os.getenv("MIN"))
  to_mock_group.add_argument("--enabled", default=os.getenv("ENABLED"), type=comma_list)

  parser.add_argument("--disabled", type=comma_list, default=os.getenv("DISABLED") or ())

  parser.add_argument(
    "--tl", dest="publish_time_length", type=float, default=None,
    help="Length of interval in event time for which messages should be published.")

  parser.add_argument(
    "--no-realtime", dest="realtime", action="store_false", default=True,
    help="Publish messages as quickly as possible instead of realtime.")

  parser.add_argument(
    "--no-interactive", dest="interactive", action="store_false", default=True,
    help="Disable interactivity.")

  parser.add_argument(
    "--bind-early", action="store_true", default=False,
    help="Bind early to avoid dropping messages.")

  parser.add_argument(
    "--no-visionipc", action="store_true", default=False,
    help="Do not output video over visionipc")

  parser.add_argument(
    "--start-time", type=float, default=0.,
    help="Seek to this absolute time (in seconds) upon starting playback.")

  return parser

def main(argv):
  args = get_arg_parser().parse_args(sys.argv[1:])

  command_address = f"ipc:///tmp/{uuid4()}"
  forward_commands_address = f"ipc:///tmp/{uuid4()}"
  data_address = f"ipc:///tmp/{uuid4()}"

  address_mapping = _get_address_mapping(args)

  command_sock = zmq.Context.instance().socket(zmq.PUSH)
  command_sock.connect(command_address)

  if args.route_name is not None:
    route_name_split = args.route_name.split("|")
    if len(route_name_split) > 1:
      route_start_time = timestamp_to_s(route_name_split[1])
    else:
      route_start_time = 0
    command_sock.send_pyobj(
      SetRoute(args.route_name, args.start_time, args.data_dir))
  else:
    print("waiting for external command...")
    route_start_time = 0

  subprocesses = {}
  try:
    subprocesses["data"] = multiprocessing.Process(
      target=UnloggerWorker().run,
      args=(forward_commands_address, data_address, address_mapping.copy()))

    subprocesses["control"] = multiprocessing.Process(
      target=unlogger_thread,
      args=(command_address, forward_commands_address, data_address, args.realtime,
            _get_address_mapping(args), args.publish_time_length, args.bind_early, args.no_loop, args.no_visionipc))

    subprocesses["data"].start()
    subprocesses["control"].start()

    # Exit if any of the children die.
    def exit_if_children_dead(*_):
      for _, p in subprocesses.items():
        if not p.is_alive():
          [p.terminate() for p in subprocesses.values()]
          exit()
      signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    signal.signal(signal.SIGCHLD, exit_if_children_dead)

    if args.interactive:
      keyboard_controller_thread(command_sock, route_start_time)
    else:
      # Wait forever for children.
      while True:
        time.sleep(10000.)
  finally:
    for p in subprocesses.values():
      if p.is_alive():
        try:
          p.join(3.)
        except multiprocessing.TimeoutError:
          p.terminate()
          continue
  return 0

if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
