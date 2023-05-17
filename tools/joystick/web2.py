import asyncio
import json
import logging
import os
import ssl
import uuid
import pyaudio
import io
import cv2

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VideoStreamTrack, AudioStreamTrack, MediaStreamError
from aiortc.contrib.media import MediaBlackhole, MediaRecorder
from aiortc.codecs.opus import OpusEncoder

import numpy as np
from av import VideoFrame, CodecContext, Packet
from pydub import AudioSegment
import sounddevice as sd
import cereal.messaging as messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from system.camerad.snapshot.snapshot import extract_image


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

pcs = set()

IMG_H, IMG_W = 540, 960
AUDIO_RATE = 16000
pm = messaging.PubMaster(['testJoystick'])


class VideoTrack(VideoStreamTrack):
  def __init__(self):
    super().__init__()  # don't forget this!
    self.vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
    self.cnt = 0

  async def recv(self):
    if os.environ.get('FAKE_CAMERA') == '1':
      frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    
    else:
      if not self.vipc_client.is_connected():
        self.vipc_client.connect(True)
      yuv_img_raw = self.vipc_client.recv()
      if yuv_img_raw is None or not yuv_img_raw.any():
        frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
      else:
        frame = extract_image(yuv_img_raw, self.vipc_client.width, self.vipc_client.height, self.vipc_client.stride, self.vipc_client.uv_offset)
        frame = cv2.resize(frame, (IMG_W, IMG_H))
        frame = np.stack([frame] * 3, axis=-1)

    frame = VideoFrame.from_ndarray(frame, format="rgb24")
    pts, time_base = await self.next_timestamp()
    frame.pts = pts
    frame.time_base = time_base
    return frame


class Speaker:
  def __init__(self):
    self.__tracks = {}
    self.p = pyaudio.PyAudio()
    self.buffer = io.BytesIO()
    self.channels = 2
    self.stream = self.p.open(format=pyaudio.paInt16, channels=self.channels, rate=48000, frames_per_buffer=960, output=True, stream_callback=self.pyaudio_callback)
    
  def pyaudio_callback(self, in_data, frame_count, time_info, status):
    # print(self.buffer.getbuffer().nbytes, frame_count)
    if self.buffer.getbuffer().nbytes == 0:
      buff = np.zeros((frame_count, 2), dtype=np.int16).tobytes()
    else:
      self.buffer.seek(0)
      buff = self.buffer.read(frame_count * self.channels * 2)
      self.buffer.seek(2)
    return (buff, pyaudio.paContinue)


  def addTrack(self, track):
    if track not in self.__tracks:
      self.__tracks[track] = None

  async def consume(self, track):
    while True:
      try:
        frame = await track.recv()
        # logging.info(f"{frame.to_ndarray().shape}, {frame.samples}, {frame.rate}, {frame.sample_rate}, {frame.layout}")
        # logging.info(f"{frame.layout.channels}, {frame.layout.name}")
      except MediaStreamError:
        return
      bio = bytes(frame.planes[0])
      self.buffer.write(bio)


  async def start(self):
    # self.stream.start()
    for track, task in self.__tracks.items():
      if task is None:
        self.__tracks[track] = asyncio.ensure_future(self.consume(track))

  async def stop(self):
    for task in self.__tracks.values():
      if task is not None:
        task.cancel()
    self.__tracks = {}


class MicTrack(AudioStreamTrack):
  def __init__(self):
    super().__init__()  # don't forget this!

    self.sample_rate = AUDIO_RATE
    self.AUDIO_PTIME = 0.020  # 20ms audio packetization
    self.samples = int(self.AUDIO_PTIME * self.sample_rate)
    self.FORMAT = pyaudio.paInt16
    self.CHANNELS = 2
    self.RATE = self.sample_rate
    self.CHUNK = int(AUDIO_RATE * 0.020)
    self.p = pyaudio.PyAudio()
    self.mic_stream = self.p.open(format=self.FORMAT, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

    self.codec = CodecContext.create('pcm_s16le', 'r')
    self.codec.sample_rate = self.RATE
    self.codec.channels = 2
    self.audio_samples = 0
    self.chunk_number = 0

  async def recv(self):
    mic_data = self.mic_stream.read(self.CHUNK)
    mic_sound = AudioSegment(mic_data, sample_width=2, channels=1, frame_rate=self.RATE)
    mic_sound = AudioSegment.from_mono_audiosegments(mic_sound, mic_sound)
    packet = Packet(mic_sound.raw_data)
    frame = self.codec.decode(packet)[0]
    frame.pts = self.audio_samples
    self.audio_samples += frame.samples
    self.chunk_number = self.chunk_number + 1
    return frame


async def index(request):
    content = open("./static/index.html", "r").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
  logger.info("\n\n\nnewoffer!\n\n")
  params = await request.json()
  offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
  # speaker = MediaRecorder("default", format="alsa", options={'ar': "48000", 'ac': "2", 'aframes': "960", 'acodec': 'pcm_s16le'})
  speaker = Speaker()

  pc = RTCPeerConnection()
  pc_id = "PeerConnection(%s)" % uuid.uuid4()
  pcs.add(pc)

  def log_info(msg, *args):
    logger.info(pc_id + " " + msg, *args)

  log_info("Created for %s", request.remote)


  @pc.on("datachannel")
  def on_datachannel(channel):
    @channel.on("message")
    def on_message(message):
      print(json.loads(message))

  @pc.on("connectionstatechange")
  async def on_connectionstatechange():
    log_info("Connection state is %s", pc.connectionState)
    if pc.connectionState == "failed":
      await pc.close()
      pcs.discard(pc)

  @pc.on('track')
  def on_track(track):
    print("Track received")
    speaker.addTrack(track)

  pc.addTrack(VideoTrack())
  pc.addTrack(MicTrack())

  await pc.setRemoteDescription(offer)
  await speaker.start()
  answer = await pc.createAnswer()
  await pc.setLocalDescription(answer)

  return web.Response(
    content_type="application/json",
    text=json.dumps(
      {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    ),
  )


async def on_shutdown(app):
  coros = [pc.close() for pc in pcs]
  await asyncio.gather(*coros)
  pcs.clear()


if __name__ == "__main__":
  ssl_context = ssl.SSLContext()
  ssl_context.load_cert_chain("cert.pem", "key.pem")
  app = web.Application()
  app.on_shutdown.append(on_shutdown)
  app.router.add_post("/offer", offer)
  app.router.add_get("/", index)
  app.router.add_static('/static', 'static')
  web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)
