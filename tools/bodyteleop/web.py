import asyncio
import dataclasses
import json
import logging
import os
import ssl
import subprocess
import time
from typing import Optional

import numpy as np
from PIL import Image
import pyaudio
import wave
from aiohttp import web
from aiohttp import ClientSession

from openpilot.common.basedir import BASEDIR
from openpilot.system.webrtc.webrtcd import StreamRequestBody
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
import cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType

try:
  import google.generativeai as genai
except ImportError:
  cloudlog.error("google-generativeai not installed. Install with: pip install google-generativeai")
  genai = None

logger = logging.getLogger("bodyteleop")
logging.basicConfig(level=logging.INFO)

TELEOPDIR = f"{BASEDIR}/tools/bodyteleop"
WEBRTCD_HOST, WEBRTCD_PORT = "localhost", 5001

# Gemini control state
gemini_pm = None
gemini_current_x = 0.0
gemini_current_y = 0.0
gemini_task = None
gemini_last_response = ""

# File paths for Gemini state (avoiding params)
GEMINI_ENABLED_FILE = os.path.join(TELEOPDIR, ".gemini_enabled")
GEMINI_PROMPT_FILE = os.path.join(TELEOPDIR, ".gemini_prompt")


## GEMINI UTILS
def yuv_to_rgb(y, u, v):
  """Convert YUV to RGB"""
  ul = np.repeat(np.repeat(u, 2).reshape(u.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
  vl = np.repeat(np.repeat(v, 2).reshape(v.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)

  yuv = np.dstack((y, ul, vl)).astype(np.int16)
  yuv[:, :, 1:] -= 128

  m = np.array([
    [1.00000,  1.00000, 1.00000],
    [0.00000, -0.39465, 2.03211],
    [1.13983, -0.58060, 0.00000],
  ])
  rgb = np.dot(yuv, m).clip(0, 255)
  return rgb.astype(np.uint8)


def get_camera_frame() -> Optional[np.ndarray]:
  """Get a frame from the road camera"""
  try:
    vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    vipc_client.connect(True)
    buf = vipc_client.recv()

    if buf is None:
      return None

    y = np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
    u = np.array(buf.data[buf.uv_offset::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]
    v = np.array(buf.data[buf.uv_offset+1::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]

    rgb = yuv_to_rgb(y, u, v)
    return rgb
  except Exception as e:
    cloudlog.error(f"Error getting camera frame: {e}")
    return None


def image_to_pil(image: np.ndarray, max_size: tuple[int, int] = (640, 480)) -> Image.Image:
  """Convert numpy image array to PIL Image and downscale if needed"""
  pil_image = Image.fromarray(image)

  if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    cloudlog.debug(f"Downscaled image from {image.shape[1]}x{image.shape[0]} to {pil_image.size[0]}x{pil_image.size[1]}")

  return pil_image


def parse_gemini_response(response_text: str) -> dict:
  """Parse Gemini response to extract movement commands"""
  response_text = response_text.strip()

  try:
    data = json.loads(response_text)
    return {
      "forward": data.get("forward", False),
      "backward": data.get("backward", False),
      "left": data.get("left", False),
      "right": data.get("right", False),
    }
  except json.JSONDecodeError:
    pass

  response_upper = response_text.upper()
  return {
    "forward": "W" in response_upper,
    "backward": "S" in response_upper,
    "left": "A" in response_upper,
    "right": "D" in response_upper,
  }


def commands_to_joystick_axes(commands: dict) -> tuple[float, float]:
  """Convert movement commands to joystick axes format"""
  forward = commands.get("forward", False)
  backward = commands.get("backward", False)
  left = commands.get("left", False)
  right = commands.get("right", False)

  if forward and backward:
    forward = backward = False
  if left and right:
    left = right = False

  x = 0.0
  y = 0.0

  if forward:
    x = 1.0
  elif backward:
    x = -1.0

  if left:
    y = -1.0
  elif right:
    y = 1.0

  return (x, y)


def gemini_is_enabled() -> bool:
  """Check if Gemini is enabled via file"""
  return os.path.exists(GEMINI_ENABLED_FILE)


def gemini_get_prompt() -> str:
  """Get Gemini prompt from file"""
  if os.path.exists(GEMINI_PROMPT_FILE):
    try:
      with open(GEMINI_PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()
    except Exception:
      return ""
  return ""


def gemini_set_enabled(enabled: bool):
  """Set Gemini enabled state via file"""
  if enabled:
    with open(GEMINI_ENABLED_FILE, 'w') as f:
      f.write("1")
  else:
    if os.path.exists(GEMINI_ENABLED_FILE):
      os.remove(GEMINI_ENABLED_FILE)


def gemini_set_prompt(prompt: str):
  """Set Gemini prompt via file"""
  if prompt:
    with open(GEMINI_PROMPT_FILE, 'w', encoding='utf-8') as f:
      f.write(prompt)
  else:
    if os.path.exists(GEMINI_PROMPT_FILE):
      os.remove(GEMINI_PROMPT_FILE)


async def gemini_loop():
  """Background task for Gemini control"""
  global gemini_current_x, gemini_current_y, gemini_last_response

  logger.info("Gemini loop starting...")

  if genai is None:
    logger.error("google-generativeai not available - install with: pip install google-generativeai")
    return

  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    logger.error("GEMINI_API_KEY environment variable not set")
    return

  logger.info("Configuring Gemini API...")
  genai.configure(api_key=api_key)
  
  # Try flash models only (cheapest)
  model_names = ['gemini-1.5-flash-latest', 'gemini-1.5-flash', 'gemini-1.5-flash-001']
  model = None
  model_name = None
  
  for name in model_names:
    try:
      model = genai.GenerativeModel(name)
      model_name = name
      logger.info(f"✓ Gemini Flash model initialized: {name}")
      break
    except Exception as e:
      logger.debug(f"Failed to load {name}: {e}")
      continue
  
  if model is None:
    logger.error("Failed to initialize any Gemini Flash model. Available models may have changed.")
    return

  joystick_rk = Ratekeeper(100, print_delay_threshold=None)
  last_gemini_call_time = 0.0
  gemini_call_interval = 5.0

  logger.info("Gemini loop ready, waiting for enable...")

  default_prompt = """You are controlling a robot with two motors in a side-by-side configuration.
Based on the camera image, determine what movement actions to take.

Movement controls:
- W: Move forward (both motors forward)
- S: Move backward (both motors backward)
- A: Rotate left in place (left motor backward, right motor forward)
- D: Rotate right in place (left motor forward, right motor backward)
- WA: Move forward while turning left (left motor slower, right motor faster)
- SD: Move backward while turning right (left motor faster, right motor slower)

You can combine W+A or S+D, but NOT forward+backward or left+right.

Respond with ONLY a JSON object in this format:
{
  "forward": true/false,
  "backward": true/false,
  "left": true/false,
  "right": true/false
}

Or respond with text commands like "W", "S", "A", "D", "WA", "SD", etc.

Analyze the image and respond with the appropriate movement command."""

  while True:
    try:
      gemini_enabled = gemini_is_enabled()

      if gemini_enabled:
        current_time = time.monotonic()
        if current_time - last_gemini_call_time >= gemini_call_interval:
          last_gemini_call_time = current_time

          logger.info("Getting camera frame...")
          frame = get_camera_frame()
          if frame is not None:
            logger.info(f"Got frame: {frame.shape}")
            custom_prompt = gemini_get_prompt()
            active_prompt = custom_prompt if custom_prompt else default_prompt
            logger.info(f"Using {'custom' if custom_prompt else 'default'} prompt (length: {len(active_prompt)})")

            pil_image = image_to_pil(frame)
            logger.info(f"Image prepared: {pil_image.size}")

            logger.info("Sending frame to Gemini API...")
            try:
              response = model.generate_content([active_prompt, pil_image])
              response_text = response.text
              gemini_last_response = response_text
              logger.info(f"✓ Gemini response received: {response_text}")

              commands = parse_gemini_response(response_text)
              logger.info(f"Parsed commands: {commands}")

              gemini_current_x, gemini_current_y = commands_to_joystick_axes(commands)
              logger.info(f"✓ Updated joystick command: x={gemini_current_x:.2f}, y={gemini_current_y:.2f}")

            except Exception as e:
              logger.error(f"✗ Error calling Gemini API: {e}", exc_info=True)
          else:
            logger.warning("Failed to get camera frame")

        # Publish joystick messages continuously at 100Hz
        joystick_msg = messaging.new_message('testJoystick')
        joystick_msg.valid = True
        joystick_msg.testJoystick.axes = [gemini_current_x, gemini_current_y]
        joystick_msg.testJoystick.buttons = [False]
        gemini_pm.send('testJoystick', joystick_msg)

        joystick_rk.keep_time()
      else:
        if gemini_current_x != 0.0 or gemini_current_y != 0.0:
          logger.info("Gemini disabled, resetting joystick to zero")
          gemini_current_x = 0.0
          gemini_current_y = 0.0
          joystick_msg = messaging.new_message('testJoystick')
          joystick_msg.valid = True
          joystick_msg.testJoystick.axes = [0.0, 0.0]
          joystick_msg.testJoystick.buttons = [False]
          gemini_pm.send('testJoystick', joystick_msg)

        await asyncio.sleep(0.1)
        joystick_rk.keep_time()

    except Exception as e:
      logger.exception(f"Error in gemini loop: {e}")
      await asyncio.sleep(1)


## UTILS
async def play_sound(sound: str):
  SOUNDS = {
    "engage": "selfdrive/assets/sounds/engage.wav",
    "disengage": "selfdrive/assets/sounds/disengage.wav",
    "error": "selfdrive/assets/sounds/warning_immediate.wav",
  }
  assert sound in SOUNDS

  chunk = 5120
  with wave.open(os.path.join(BASEDIR, SOUNDS[sound]), "rb") as wf:
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


async def sound(request: 'web.Request'):
  params = await request.json()
  sound_to_play = params["sound"]

  await play_sound(sound_to_play)
  return web.json_response({"status": "ok"})


async def gemini_control(request: 'web.Request'):
  """Toggle or get Gemini control status"""
  if request.method == 'GET':
    # Get current status and prompt
    enabled = gemini_is_enabled()
    prompt = gemini_get_prompt()
    status_info = {
      "enabled": enabled,
      "prompt": prompt,
      "api_key_set": bool(os.getenv("GEMINI_API_KEY")),
      "genai_available": genai is not None,
      "current_x": gemini_current_x,
      "current_y": gemini_current_y,
      "last_response": gemini_last_response[:100] if gemini_last_response else "",  # Truncate for display
    }
    return web.json_response(status_info)
  else:
    # POST: Toggle or set status
    data = await request.json()
    enabled = data.get("enabled")

    if enabled is not None:
      gemini_set_enabled(enabled)
      logger.info(f"✓ Gemini control {'ENABLED' if enabled else 'DISABLED'}")
      if enabled:
        logger.info("  - Waiting for next camera frame (every 5 seconds)")
        logger.info(f"  - API key: {'set' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
        logger.info(f"  - GenAI library: {'available' if genai else 'NOT AVAILABLE'}")
      return web.json_response({"status": "ok", "enabled": enabled})
    else:
      # Toggle if no value provided
      current = gemini_is_enabled()
      new_value = not current
      gemini_set_enabled(new_value)
      logger.info(f"✓ Gemini control toggled to {'ENABLED' if new_value else 'DISABLED'}")
      return web.json_response({"status": "ok", "enabled": new_value})


async def gemini_prompt(request: 'web.Request'):
  """Get or set Gemini prompt"""
  if request.method == 'GET':
    prompt = gemini_get_prompt()
    return web.json_response({"prompt": prompt})
  else:
    # POST: Set prompt
    data = await request.json()
    prompt = data.get("prompt", "")

    gemini_set_prompt(prompt)
    logger.info(f"Gemini prompt updated (length: {len(prompt)})")

    return web.json_response({"status": "ok", "prompt": prompt})


async def offer(request: 'web.Request'):
  params = await request.json()
  body = StreamRequestBody(params["sdp"], ["wideRoad"], ["testJoystick"], ["carState"])
  body_json = json.dumps(dataclasses.asdict(body))

  logger.info("Sending offer to webrtcd...")
  webrtcd_url = f"http://{WEBRTCD_HOST}:{WEBRTCD_PORT}/stream"
  async with ClientSession() as session, session.post(webrtcd_url, data=body_json) as resp:
    assert resp.status == 200
    answer = await resp.json()
    return web.json_response(answer)


def main():
  global gemini_pm, gemini_task

  # Enable joystick debug mode
  from openpilot.common.params import Params
  Params().put_bool("JoystickDebugMode", True)

  # Initialize messaging for Gemini
  gemini_pm = messaging.PubMaster(['testJoystick'])

  # App needs to be HTTPS for microphone and audio autoplay to work on the browser
  ssl_context = create_ssl_context()

  app = web.Application()
  app.router.add_get("/", index)
  app.router.add_get("/ping", ping, allow_head=True)
  app.router.add_post("/offer", offer)
  app.router.add_post("/sound", sound)
  app.router.add_get("/gemini", gemini_control)
  app.router.add_post("/gemini", gemini_control)
  app.router.add_get("/gemini/prompt", gemini_prompt)
  app.router.add_post("/gemini/prompt", gemini_prompt)
  app.router.add_static('/static', os.path.join(TELEOPDIR, 'static'))

  # Start Gemini background task when app starts
  async def startup_background_tasks(app):
    global gemini_task
    gemini_task = asyncio.create_task(gemini_loop())

  app.on_startup.append(startup_background_tasks)

  web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)


if __name__ == "__main__":
  main()
