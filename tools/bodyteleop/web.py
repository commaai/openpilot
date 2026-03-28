import dataclasses
import json
import logging
import os
import ssl
import subprocess
import wave

from aiohttp import web
from aiohttp import ClientSession
import numpy as np
import sounddevice as sd

from openpilot.common.basedir import BASEDIR
from openpilot.system.webrtc.webrtcd import StreamRequestBody
from openpilot.common.params import Params

logger = logging.getLogger("bodyteleop")
logging.basicConfig(level=logging.INFO)

TELEOPDIR = f"{BASEDIR}/tools/bodyteleop"
WEBRTCD_HOST, WEBRTCD_PORT = "localhost", 5001
SOUND_FILES = {
  "engage": "engage.wav",
  "disengage": "disengage.wav",
  "error": "warning_immediate.wav",
}


## SSL
def create_ssl_cert(cert_path: str, key_path: str):
  try:
    proc = subprocess.run(f'openssl req -x509 -newkey rsa:4096 -nodes -out {cert_path} -keyout {key_path} \
                          -days 365 -subj "/C=US/ST=California/O=commaai/OU=comma body"',
                          capture_output=True, shell=True)
    proc.check_returncode()
  except subprocess.CalledProcessError as ex:
    raise ValueError(f"Error creating SSL certificate:\n[stdout]\n{proc.stdout.decode()}\n[stderr]\n{proc.stderr.decode()}") from ex


def create_ssl_context():
  cert_path = os.path.join(TELEOPDIR, "cert.pem")
  key_path = os.path.join(TELEOPDIR, "key.pem")
  if not os.path.exists(cert_path) or not os.path.exists(key_path):
    logger.info("Creating certificate...")
    create_ssl_cert(cert_path, key_path)
  else:
    logger.info("Certificate exists!")
  ssl_context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER)
  ssl_context.load_cert_chain(cert_path, key_path)

  return ssl_context

## ENDPOINTS
async def index(request: 'web.Request'):
  with open(os.path.join(TELEOPDIR, "static", "index.html")) as f:
    content = f.read()
    return web.Response(content_type="text/html", text=content)


async def ping(request: 'web.Request'):
  return web.Response(text="pong")


async def offer(request: 'web.Request'):
  params = await request.json()
  body = StreamRequestBody(params["sdp"], ["driver"], ["testJoystick"], ["carState"])
  body_json = json.dumps(dataclasses.asdict(body))

  logger.info("Sending offer to webrtcd...")
  webrtcd_url = f"http://{WEBRTCD_HOST}:{WEBRTCD_PORT}/stream"
  async with ClientSession() as session, session.post(webrtcd_url, data=body_json) as resp:
    assert resp.status == 200
    answer = await resp.json()
    return web.json_response(answer)

async def sound(request: 'web.Request'):
  body = await request.json()
  sound_name = body.get("sound")
  if sound_name not in SOUND_FILES:
    return web.json_response({"ok": False, "error": f"unknown sound: {sound_name}"}, status=400)

  sound_path = os.path.join(BASEDIR, "selfdrive", "assets", "sounds", SOUND_FILES[sound_name])
  with wave.open(sound_path, "rb") as wave_file:
    n_channels = wave_file.getnchannels()
    frames = wave_file.readframes(wave_file.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
      audio = audio.reshape(-1, n_channels)
    sd.play(audio.astype(np.float32) / 32768.0, samplerate=wave_file.getframerate(), blocking=False)

  return web.json_response({"ok": True})


async def autonomy(request: 'web.Request'):
  body = await request.json()
  enabled = bool(body.get("enabled", False))
  params = Params()
  params.put_bool("BodyAutonomyEnabled", enabled)
  params.put_bool("JoystickDebugMode", True)
  return web.json_response({"ok": True, "enabled": enabled})


async def autonomy_config(request: 'web.Request'):
  body = await request.json()
  params = Params()

  if "target_visible" in body:
    params.put_bool("BodyAutonomyTargetVisible", bool(body["target_visible"]))
  for key, param_key in {
    "target_distance_m": "BodyAutonomyTargetDistance",
    "target_bearing_deg": "BodyAutonomyTargetBearingDeg",
    "obstacle_distance_m": "BodyAutonomyObstacleDistance",
  }.items():
    if key in body:
      params.put(param_key, str(float(body[key])))

  return web.json_response({"ok": True})


async def autonomy_status(request: 'web.Request'):
  params = Params()
  status = params.get("BodyAutonomyStatus", encoding="utf8")
  return web.json_response({"status": json.loads(status) if status else {}})


def main():
  # Enable joystick debug mode
  Params().put_bool("JoystickDebugMode", True)

  # App needs to be HTTPS for WebRTC to work on the browser
  ssl_context = create_ssl_context()

  app = web.Application()
  app.router.add_get("/", index)
  app.router.add_get("/ping", ping, allow_head=True)
  app.router.add_post("/offer", offer)
  app.router.add_post("/sound", sound)
  app.router.add_post("/autonomy", autonomy)
  app.router.add_post("/autonomy/config", autonomy_config)
  app.router.add_get("/autonomy/status", autonomy_status)
  app.router.add_static('/static', os.path.join(TELEOPDIR, 'static'))
  web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)


if __name__ == "__main__":
  main()
