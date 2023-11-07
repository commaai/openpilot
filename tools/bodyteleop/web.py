import asyncio
import dataclasses
import json
import logging
import os
import ssl
import time
import subprocess

# aiortc and its dependencies have lots of internal warnings :(
import warnings
warnings.resetwarnings()
warnings.simplefilter("always")

from aiohttp import web, ClientSession
import pyaudio
import wave

from openpilot.common.basedir import BASEDIR
from openpilot.tools.bodyteleop.webrtc import WebRTCStreamBuilder
from openpilot.tools.bodyteleop.webrtcd import StreamRequestBody

logger = logging.getLogger("bodyteleop")
logging.basicConfig(level=logging.INFO)

TELEOPDIR = f"{BASEDIR}/tools/bodyteleop"
WEBRTCD_HOST = "http://localhost:5001"

## UTILS
async def play_sound(sound):
  SOUNDS = {
    'engage': '../../selfdrive/assets/sounds/engage.wav',
    'disengage': '../../selfdrive/assets/sounds/disengage.wav',
    'error': '../../selfdrive/assets/sounds/warning_immediate.wav',
  }

  chunk = 5120
  with wave.open(SOUNDS[sound], 'rb') as wf:
    def callback(in_data, frame_count, time_info, status):
      data = wf.readframes(frame_count)
      return data, pyaudio.paContinue

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=chunk,
                    stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
      await asyncio.sleep(0)
    stream.stop_stream()
    stream.close()
    p.terminate()

# TODO body 0,0 messages sent over some interval
# options:
# - endpoint for sending control messages
# - send control on client side, and dummy on server side
# - dummy on webrtcd side

## SSL
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

## APP EVENTS
async def internal_stream_startup(app):
  # this is executed on startup
  async def connection_provider(offer):
    body = StreamRequestBody(offer.sdp, offer.video, ["testJoystick"], [])
    body_json = json.dumps(dataclasses.asdict(body))
    async with ClientSession() as session, session.post("{WEBRTCD_HOST}/stream", data=body_json) as resp:
      if resp.status != 200:
        raise web.HTTPInternalServerError()
      return await resp.json()

  builder = WebRTCStreamBuilder.offer(connection_provider)
  builder.add_messaging()

  stream = builder.stream()
  await stream.start()
  await stream.wait_for_connection()
  assert stream.has_messaging_channel()
  app['control_stream'] = stream

  # yield the execution after startup
  yield
  # this is executed on cleanup
  await stream.stop()


## ENDPOINTS
async def index(request):
  content = open(TELEOPDIR + "/static/index.html", "r").read()
  now = time.monotonic()
  request.app['mutable_vals']['last_send_time'] = now
  request.app['mutable_vals']['last_override_time'] = now
  request.app['mutable_vals']['prev_command'] = []
  request.app['mutable_vals']['find_person'] = False

  return web.Response(content_type="text/html", text=content)


async def control(request):
  assert request.app['control_stream'].is_connected_and_ready, "RTC control stream not ready"
  channel = request.app['control_stream'].get_messaging_channel()
  x, y = request.match_info.get('x'), request.match_info.get('y')
  msg = {"type": "testJoystick", "data": {"axes": [float(x), float(y)]}}
  msg_json = json.dumps(msg)
  channel.send(msg_json)


async def ping(request):
  pass


async def sound(request):
  params = await request.json()
  sound_to_play = params["sound"]

  try:
    await play_sound(sound_to_play)
    return web.json_response({"status": "ok"})
  except Exception as ex:
    return web.json_response({"error": str(ex)}, status=400)


async def offer(request):
  params = await request.json()
  body = StreamRequestBody(params["sdp"], ["driver"], [], ["carState"])
  body_json = json.dumps(dataclasses.asdict(body))

  raise web.HTTPFound("{WEBRTCD_HOST}/stream", text=body_json)


def main():
  # App needs to be HTTPS for microphone and audio autoplay to work on the browser
  ssl_context = create_ssl_context()

  app = web.Application()
  app['mutable_vals'] = {}
  app.router.add_get("/", index)
  app.router.add_get("/control/{x}/{y}", control)
  app.router.add_get("/ping", ping)
  app.router.add_post("/offer", offer)
  app.router.add_post("/sound", sound)
  app.router.add_static('/static', TELEOPDIR + '/static')
  app.cleanup_ctx.append(internal_stream_startup)
  web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)


if __name__ == "__main__":
  main()
