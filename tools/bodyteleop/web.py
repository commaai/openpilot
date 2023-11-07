import asyncio
import json
import logging
import os
import ssl
import uuid
import time
import subprocess
from typing import Dict, Awaitable

# aiortc and its dependencies have lots of internal warnings :(
import warnings
warnings.resetwarnings()
warnings.simplefilter("always")

from aiohttp import web

import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.tools.bodyteleop.bodyav import play_sound
from openpilot.tools.bodyteleop.webrtc import WebRTCStreamBuilder
from openpilot.tools.bodyteleop.webrtc.stream import WebRTCBaseStream
from openpilot.tools.bodyteleop.webrtc.info import parse_info_from_offer
from openpilot.tools.bodyteleop.webrtc.device.video import LiveStreamVideoStreamTrack
from openpilot.tools.bodyteleop.webrtc.device.audio import AudioInputStreamTrack, AudioOutputSpeaker

logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

streams: Dict[str, WebRTCBaseStream] = dict()
stream_connection_tasks: Dict[str, Awaitable[None]] = dict()
pm, sm = None, None
TELEOPDIR = f"{BASEDIR}/tools/bodyteleop"


async def control_body(data, app):
  now = time.monotonic()
  if (data['type'] == 'dummy_controls') and (now < (app['mutable_vals']['last_send_time'] + 0.2)):
    return
  if (data['type'] == 'control_command') and (app['mutable_vals']['prev_command'] == [data['x'], data['y']] and data['x'] == 0 and data['y'] == 0):
    return

  logger.info(str(data))
  x = max(-1.0, min(1.0, data['x']))
  y = max(-1.0, min(1.0, data['y']))
  dat = messaging.new_message('testJoystick')
  dat.testJoystick.axes = [x, y]
  dat.testJoystick.buttons = [False]
  pm.send('testJoystick', dat)
  app['mutable_vals']['last_send_time'] = now
  if (data['type'] == 'control_command'):
    app['mutable_vals']['last_override_time'] = now
    app['mutable_vals']['prev_command'] = [data['x'], data['y']]


async def dummy_controls_msg(app):
  while True:
    if 'last_send_time' in app['mutable_vals']:
      this_time = time.monotonic()
      if (app['mutable_vals']['last_send_time'] + 0.2) < this_time:
        await control_body({'type': 'dummy_controls', 'x': 0, 'y': 0}, app)
    await asyncio.sleep(0.2)


async def start_background_tasks(app):
  app['bgtask_dummy_controls_msg'] = asyncio.create_task(dummy_controls_msg(app))


async def stop_background_tasks(app):
  app['bgtask_dummy_controls_msg'].cancel()
  del app['bgtask_dummy_controls_msg']


async def index(request):
  content = open(TELEOPDIR + "/static/index.html", "r").read()
  now = time.monotonic()
  request.app['mutable_vals']['last_send_time'] = now
  request.app['mutable_vals']['last_override_time'] = now
  request.app['mutable_vals']['prev_command'] = []
  request.app['mutable_vals']['find_person'] = False

  return web.Response(content_type="text/html", text=content)


async def offer(request):
  async def on_webrtc_channel_message(channel, message):
    data = json.loads(message)
    if data['type'] == 'control_command':
      await control_body(data, request.app)
      times = {
        'type': 'ping_time',
        'incoming_time': data['dt'],
        'outgoing_time': int(time.time() * 1000),
      }
      channel.send(json.dumps(times))
    if data['type'] == 'battery_level':
      sm.update(timeout=0)
      if sm.updated['carState']:
        channel.send(json.dumps({'type': 'battery_level', 'value': int(sm['carState'].fuelGauge * 100)}))
    if data['type'] == 'play_sound':
      logger.info(f"Playing sound: {data['sound']}")
      await play_sound(data['sound'])
    if data['type'] == 'find_person':
      request.app['mutable_vals']['find_person'] = data['value']

  async def post_connect(stream, identifier):
    try:
      await stream.wait_for_connection()

      if stream.has_incoming_audio_track():
        track = stream.get_incoming_audio_track(False)
        speaker = AudioOutputSpeaker()
        speaker.add_track(track)
        speaker.start()
    except Exception as e:
      logger.info(f"Connection exception with stream {identifier}: {e}")
      await stream.stop()
      del streams[stream_id]
      del stream_connection_tasks[stream_id]

  logger.info("\n\nNew Offer!\n\n")

  params = await request.json()
  sdp = params["sdp"]

  stream_builder = WebRTCStreamBuilder.answer(sdp)
  media_info = parse_info_from_offer(sdp)
  if media_info.n_expected_camera_tracks >= 0:
    camera_track = LiveStreamVideoStreamTrack("driver")
    stream_builder.add_video_stream("driver", camera_track)
  if media_info.expected_audio_track:
    audio_track = AudioInputStreamTrack()
    stream_builder.add_audio_stream(audio_track)
  if media_info.incoming_audio_track:
    stream_builder.request_audio_stream()

  stream = stream_builder.stream()
  stream.set_message_handler(on_webrtc_channel_message)
  description = await stream.start()
  stream_id = "WebRTCStream(%s)" % uuid.uuid4()

  connection_task = asyncio.create_task(post_connect(stream, stream_id))
  streams[stream_id] = stream
  stream_connection_tasks[stream_id] = connection_task

  response_content = {"sdp": description.sdp, "type": description.type}
  return web.json_response(response_content)


async def on_shutdown(app):
  coroutines = [stream.stop() for stream in streams.values()]
  await asyncio.gather(*coroutines)
  streams.clear()


def create_ssl_cert(cert_path, key_path):
  try:
    proc = subprocess.run(f'openssl req -x509 -newkey rsa:4096 -nodes -out {cert_path} -keyout {key_path} \
                          -days 365 -subj "/C=US/ST=California/O=commaai/OU=comma body"',
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    proc.check_returncode()
  except subprocess.CalledProcessError as ex:
    raise ValueError(f"Error creating SSL certificate:\n[stdout]\n{proc.stdout.decode()}\n[stderr]\n{proc.stderr.decode()}") from ex


def create_ssl_context():
  cert_path = TELEOPDIR + '/cert.pem'
  key_path = TELEOPDIR + '/key.pem'
  if not os.path.exists(cert_path) or not os.path.exists(key_path):
    logger.info("Creating certificate...")
    create_ssl_cert(cert_path, key_path)
  else:
    logger.info("Certificate exists!")
  ssl_context = ssl.SSLContext()
  ssl_context.load_cert_chain(cert_path, key_path)

  return ssl_context


def main():
  global pm, sm
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['carState', 'logMessage'])

  # App needs to be HTTPS for microphone and audio autoplay to work on the browser
  ssl_context = create_ssl_context()

  app = web.Application()
  app['mutable_vals'] = {}
  app['streams'] = {}
  app.on_shutdown.append(on_shutdown)
  app.router.add_get("/", index)
  app.router.add_post("/offer", offer)
  app.router.add_static('/static', TELEOPDIR + '/static')
  app.on_startup.append(start_background_tasks)
  app.on_cleanup.append(stop_background_tasks)
  web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)


if __name__ == "__main__":
  main()
