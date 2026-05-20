#!/usr/bin/env python3
import av
import os
import sys
import argparse
import numpy as np
import multiprocessing
import time
import signal
import zmq

import capnp
from cereal import log
from msgq.visionipc import VisionIpcServer, VisionStreamType

V4L2_BUF_FLAG_KEYFRAME = 8


def fnv1a_hash(s: str) -> int:
  """FNV-1a hash function matching bridge_zmq.cc"""
  h = 0xcbf29ce484222325
  for c in s.encode():
    h ^= c
    h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
  return h


def get_port(endpoint: str) -> int:
  """Calculate port number from endpoint name, matching bridge_zmq.cc"""
  return 8023 + (fnv1a_hash(endpoint) % (65535 - 8023))


class ZmqSubSocket:
  """ZMQ subscriber socket matching BridgeZmqSubSocket from bridge_zmq.cc"""

  def __init__(self, endpoint: str, address: str, conflate: bool = False):
    self.ctx = zmq.Context()
    self.sock = self.ctx.socket(zmq.SUB)
    self.sock.setsockopt(zmq.SUBSCRIBE, b"")

    if conflate:
      self.sock.setsockopt(zmq.CONFLATE, 1)

    reconnect_ivl = 500
    self.sock.setsockopt(zmq.RECONNECT_IVL_MAX, reconnect_ivl)

    port = get_port(endpoint)
    self.endpoint = f"tcp://{address}:{port}"
    self.sock.connect(self.endpoint)

  def set_timeout(self, timeout_ms: int):
    self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)

  def recv(self, non_blocking: bool = False) -> bytes:
    flags = zmq.NOBLOCK if non_blocking else 0
    try:
      return self.sock.recv(flags=flags)
    except zmq.Again:
      return None

  def __del__(self):
    self.sock.close()
    self.ctx.term()


class ZmqPoller:
  """ZMQ poller matching BridgeZmqPoller from bridge_zmq.cc"""

  def __init__(self):
    self.socks = []

  def register(self, sock: ZmqSubSocket):
    self.socks.append(sock)

  def poll(self, timeout_ms: int):
    """Poll registered sockets, return list of sockets with events"""
    items = [sock.sock for sock in self.socks]
    if not items:
      return []

    zmq_poller = zmq.Poller()
    for s in items:
      zmq_poller.register(s, zmq.POLLIN)

    result = zmq_poller.poll(timeout_ms)
    ready_socks = []
    for s, _ in result:
      for sock in self.socks:
        if sock.sock == s:
          ready_socks.append(sock)
          break
    return ready_socks


ENCODE_SOCKETS = {
  VisionStreamType.VISION_STREAM_ROAD: "roadEncodeData",
  VisionStreamType.VISION_STREAM_DRIVER: "driverEncodeData",
  VisionStreamType.VISION_STREAM_WIDE_ROAD: "wideRoadEncodeData",
}


def decoder(addr, vipc_server, vst, nvidia, W, H, debug=False):
  sock_name = ENCODE_SOCKETS[vst]
  if debug:
    print(f"start decoder for {sock_name}, {W}x{H}")

  if nvidia:
    os.environ["NV_LOW_LATENCY"] = "3"    # both bLowLatency and CUVID_PKT_ENDOFPICTURE
    sys.path += os.environ["LD_LIBRARY_PATH"].split(":")
    import PyNvCodec as nvc

    nvDec = nvc.PyNvDecoder(W, H, nvc.PixelFormat.NV12, nvc.CudaVideoCodec.HEVC, 0)
    cc1 = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709, nvc.ColorRange.JPEG)
    conv_yuv = nvc.PySurfaceConverter(W, H, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, 0)
    nvDwn_yuv = nvc.PySurfaceDownloader(W, H, nvc.PixelFormat.YUV420, 0)
    img_yuv = np.ndarray((H*W//2*3), dtype=np.uint8)
  else:
    codec = av.CodecContext.create("hevc", "r")

  sock = ZmqSubSocket(sock_name, addr, conflate=False)
  cnt = 0
  last_idx = -1
  seen_iframe = False

  time_q = []
  while 1:
    dat = sock.recv()
    if dat is None:
      continue

    with log.Event.from_bytes(dat) as evt:
      evta = getattr(evt, evt.which())
      if debug and evta.idx.encodeId != 0 and evta.idx.encodeId != (last_idx+1):
        print("DROP PACKET!")
      last_idx = evta.idx.encodeId
      if not seen_iframe and not (evta.idx.flags & V4L2_BUF_FLAG_KEYFRAME):
        if debug:
          print("waiting for iframe")
        continue
      time_q.append(time.monotonic())
      network_latency = (int(time.time()*1e9) - evta.unixTimestampNanos)/1e6  # noqa: TID251
      frame_latency = ((evta.idx.timestampEof/1e9) - (evta.idx.timestampSof/1e9))*1000
      process_latency = ((evt.logMonoTime/1e9) - (evta.idx.timestampEof/1e9))*1000

      # put in header (first)
      if not seen_iframe:
        if nvidia:
          nvDec.DecodeSurfaceFromPacket(np.frombuffer(evta.header, dtype=np.uint8))
        else:
          codec.decode(av.packet.Packet(evta.header))
        seen_iframe = True

      if nvidia:
        rawSurface = nvDec.DecodeSurfaceFromPacket(np.frombuffer(evta.data, dtype=np.uint8))
        if rawSurface.Empty():
          if debug:
            print("DROP SURFACE")
          continue
        convSurface = conv_yuv.Execute(rawSurface, cc1)
        nvDwn_yuv.DownloadSingleSurface(convSurface, img_yuv)
      else:
        frames = codec.decode(av.packet.Packet(evta.data))
        if len(frames) == 0:
          if debug:
            print("DROP SURFACE")
          continue
        assert len(frames) == 1
        img_yuv = frames[0].to_ndarray(format=av.video.format.VideoFormat('yuv420p')).flatten()
        uv_offset = H*W
        y = img_yuv[:uv_offset]
        uv = img_yuv[uv_offset:].reshape(2, -1).ravel('F')
        img_yuv = np.hstack((y, uv))

      vipc_server.send(vst, img_yuv.data, cnt, int(time_q[0]*1e9), int(time.monotonic()*1e9))
      cnt += 1

      pc_latency = (time.monotonic()-time_q[0])*1000
      time_q = time_q[1:]
      if debug:
        print(f" 1 {evta.idx.encodeId:4d} {evt.logMonoTime/1e9:.3f} {evta.idx.timestampEof/1e6:.3f} "
              f"roll {frame_latency:6.2f} ms latency {process_latency:6.2f} ms + {network_latency:6.2f} ms + {pc_latency:6.2f} ms "
              f"= {process_latency+network_latency+pc_latency:6.2f} ms [{len(evta.data)} bytes] {sock_name}")


class CompressedVipc:
  def __init__(self, addr, vision_streams, server_name, nvidia=False, debug=False):
    print("getting frame sizes")

    # Wait for first frame to get dimensions
    self.poller = ZmqPoller()
    self.socks = {}
    self.data = {}

    for vst in vision_streams:
      sock_name = ENCODE_SOCKETS[vst]
      sock = ZmqSubSocket(sock_name, addr, conflate=True)
      self.poller.register(sock)
      self.socks[sock_name] = sock
      self.data[sock_name] = None

    # Poll until we get at least one message on each socket
    waiting_for = set(ENCODE_SOCKETS[vst] for vst in vision_streams)
    while waiting_for:
      ready = self.poller.poll(5000)
      for sock in ready:
        for name, s in self.socks.items():
          if s == sock:
            dat = sock.recv()
            if dat and self.data[name] is None:
              with log.Event.from_bytes(dat) as evt:
                self.data[name] = getattr(evt, name)
              waiting_for.discard(name)
            break

    print(222)

    self.vipc_server = VisionIpcServer(server_name)
    for vst in vision_streams:
      ed = self.data[ENCODE_SOCKETS[vst]]
      self.vipc_server.create_buffers(vst, 4, ed.width, ed.height)
    self.vipc_server.start_listener()

    print("start decoders")

    self.procs = []
    for vst in vision_streams:
      ed = self.data[ENCODE_SOCKETS[vst]]
      print(f"start decoder for {ENCODE_SOCKETS[vst]}, {ed.width}x{ed.height}")
      p = multiprocessing.Process(target=decoder, args=(addr, self.vipc_server, vst, nvidia, ed.width, ed.height, debug))
      p.start()
      self.procs.append(p)

  def join(self):
    for p in self.procs:
      p.join()

  def kill(self):
    for p in self.procs:
      p.terminate()
    self.join()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Decode video streams and broadcast on VisionIPC")
  parser.add_argument("addr", help="Address of comma three")
  parser.add_argument("--nvidia", action="store_true", help="Use nvidia instead of ffmpeg")
  parser.add_argument("--cams", default="0,1,2", help="Cameras to decode")
  parser.add_argument("--server", default="camerad", help="choose vipc server name")
  parser.add_argument("--silent", action="store_true", help="Suppress debug output")
  args = parser.parse_args()

  vision_streams = [
    VisionStreamType.VISION_STREAM_ROAD,
    VisionStreamType.VISION_STREAM_DRIVER,
    VisionStreamType.VISION_STREAM_WIDE_ROAD,
  ]

  vsts = [vision_streams[int(x)] for x in args.cams.split(",")]
  cvipc = CompressedVipc(args.addr, vsts, args.server, args.nvidia, debug=(not args.silent))

  # register exit handler
  signal.signal(signal.SIGINT, lambda sig, frame: cvipc.kill())

  cvipc.join()
