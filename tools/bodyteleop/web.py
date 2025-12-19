import asyncio
import dataclasses
import json
import logging
import os
import re
import ssl
import subprocess
import time
from typing import Optional, Tuple

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

# Gemini control state (in-memory only, no file persistence)
gemini_pm = None
gemini_current_x = 0.0
gemini_current_y = 0.0
gemini_task = None
gemini_last_response = ""
gemini_enabled = False  # In-memory state, starts disabled
gemini_prompt = ""  # In-memory prompt
gemini_plan = None  # Current plan being executed [(w, a, s, d, end_time), ...]
gemini_plan_start_time = None  # When the current plan started


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


def get_camera_frame():
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


def parse_gemini_response(response_text: str) -> Tuple[Optional[dict], Optional[list]]:
  """Parse Gemini response to extract movement commands or plan
  Returns: (commands_dict, plan_list)
  - commands_dict: {"forward": bool, "backward": bool, "left": bool, "right": bool} or None
  - plan_list: [(w, a, s, d, end_time), ...] or None if no plan
  """
  response_text = response_text.strip()

  # Check for plan format first (lines with w,a,s,d,t)
  plan_lines = []
  lines = response_text.split('\n')
  for line in lines:
    line = line.strip()
    if ',' in line and line.count(',') == 4:
      parts = line.split(',')
      try:
        w, a, s, d, t = [int(float(p.strip())) if '.' not in p.strip() else float(p.strip()) for p in parts]
        if w in [0, 1] and a in [0, 1] and s in [0, 1] and d in [0, 1] and t >= 0:
          plan_lines.append((w, a, s, d, t))
      except (ValueError, IndexError):
        pass

  if plan_lines:
    # Convert to cumulative timestamps
    plan = []
    cumulative_time = 0.0
    for w, a, s, d, duration in plan_lines:
      cumulative_time += duration
      plan.append((w, a, s, d, cumulative_time))
    return (None, plan)

  # Try to parse as JSON
  json_start = response_text.find('{')
  json_end = response_text.rfind('}')
  if json_start != -1 and json_end != -1:
    json_str = response_text[json_start:json_end+1]
    try:
      data = json.loads(json_str)
      commands = {
        "forward": bool(data.get("forward", False)),
        "backward": bool(data.get("backward", False)),
        "left": bool(data.get("left", False)),
        "right": bool(data.get("right", False)),
      }
      # Only return commands if at least one is True
      if any(commands.values()):
        return (commands, None)
    except json.JSONDecodeError:
      pass

  # Look for explicit command words at start of response or after "command:" etc
  response_upper = response_text.upper()

  # Only parse if response is very short (likely a command) or contains explicit markers
  is_short_response = len(response_text.split()) <= 5
  has_command_marker = any(marker in response_upper for marker in ["COMMAND:", "RESPONSE:", "ACTION:", "MOVE:"])

  if is_short_response or has_command_marker:
    # Look for exact command words, not just letters
    commands = {
      "forward": bool(re.search(r'\b(W|FORWARD)\b', response_upper)),
      "backward": bool(re.search(r'\b(S|BACKWARD)\b', response_upper)),
      "left": bool(re.search(r'\b(A|LEFT)\b', response_upper)),
      "right": bool(re.search(r'\b(D|RIGHT)\b', response_upper)),
    }
    if any(commands.values()):
      return (commands, None)

  # No valid commands found
  return ({"forward": False, "backward": False, "left": False, "right": False}, None)


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
  """Check if Gemini is enabled (in-memory state)"""
  global gemini_enabled
  return gemini_enabled


def gemini_get_prompt() -> str:
  """Get Gemini prompt (in-memory state)"""
  global gemini_prompt
  return gemini_prompt


def gemini_set_enabled(enabled: bool):
  """Set Gemini enabled state (in-memory only)"""
  global gemini_enabled
  gemini_enabled = enabled


def gemini_set_prompt(prompt: str):
  """Set Gemini prompt (in-memory only)"""
  global gemini_prompt
  gemini_prompt = prompt


async def gemini_loop():
  """Background task for Gemini control"""
  global gemini_current_x, gemini_current_y, gemini_last_response

  logger.info("Gemini loop starting (will wait for enable)...")

  try:
    if genai is None:
      logger.warning("google-generativeai not available - Gemini control disabled")
      return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      logger.warning("GEMINI_API_KEY environment variable not set - Gemini control disabled")
      return
  except Exception as e:
    logger.error(f"Error checking Gemini prerequisites: {e}", exc_info=True)
    return

  logger.info("Configuring Gemini API...")
  genai.configure(api_key=api_key)

  # Try known working flash models first (cheapest), then discover dynamically
  preferred_models = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash']
  model = None
  model_name = None

  # Try preferred models first
  for name in preferred_models:
    try:
      model = genai.GenerativeModel(name)
      model_name = name
      logger.info(f"✓ Gemini Flash model initialized: {name}")
      break
    except Exception as e:
      logger.debug(f"Failed to load {name}: {e}")
      continue

  # If preferred models don't work, discover dynamically
  if model is None:
    try:
      models = list(genai.list_models())
      logger.info(f"Discovering models... Found {len(models)} total models")

      for model_obj in models:
        name = model_obj.name
        methods = getattr(model_obj, 'supported_generation_methods', [])
        # Model names come as "models/gemini-2.5-flash" format
        if 'flash' in name.lower() and ('generateContent' in methods or hasattr(model_obj, 'generate_content')):
          # Extract just the model name part (remove 'models/' prefix)
          if '/' in name:
            candidate_name = name.split('/')[-1]
          else:
            candidate_name = name

          try:
            model = genai.GenerativeModel(candidate_name)
            model_name = candidate_name
            logger.info(f"✓ Gemini Flash model initialized: {candidate_name} (discovered)")
            break
          except Exception as e:
            logger.debug(f"Failed to use discovered model {candidate_name}: {e}")
            continue

      if not model_name:
        logger.error("No flash model found. Available models with generateContent:")
        for model_obj in models:
          methods = getattr(model_obj, 'supported_generation_methods', [])
          if 'generateContent' in methods or hasattr(model_obj, 'generate_content'):
            logger.error(f"  - {model_obj.name}")
        return
    except Exception as e:
      logger.error(f"Error discovering models: {e}", exc_info=True)
      return

  if model is None:
    logger.error("Failed to initialize any Gemini Flash model")
    return

  joystick_rk = Ratekeeper(100, print_delay_threshold=None)
  last_gemini_call_time = 0.0
  gemini_call_interval = 10.0  # Send frame every 10 seconds

  logger.info("Gemini loop ready, waiting for enable...")

  default_prompt = """You are controlling a robot. Analyze the image and output ONLY a JSON object with movement commands. NO TEXT EXPLANATIONS. NO DESCRIPTIONS.

Movement controls (only ONE direction at a time):
- W/forward: Move forward (~5 mph)
- S/backward: Move backward (~5 mph)
- A/left: Rotate left (~360 deg/s, keep turns SHORT: 0.1-0.3s)
- D/right: Rotate right (~360 deg/s, keep turns SHORT: 0.1-0.3s)

Output ONLY this JSON format (nothing else):
{"forward": false, "backward": false, "left": false, "right": false}

Set exactly ONE to true based on what action to take. If no movement needed, set all to false.

DO NOT write any text explanations. DO NOT describe the image. Output ONLY the JSON object."""

  global gemini_plan, gemini_plan_start_time, gemini_pm, gemini_current_x, gemini_current_y

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

              commands, plan = parse_gemini_response(response_text)

              if plan is not None:
                # Start executing plan
                gemini_plan = plan
                gemini_plan_start_time = time.monotonic()
                logger.info(f"✓ Parsed plan with {len(plan)} steps: {plan}")
              elif commands is not None:
                logger.info(f"Parsed commands: {commands}")
                gemini_current_x, gemini_current_y = commands_to_joystick_axes(commands)
                logger.info(f"✓ Updated joystick command: x={gemini_current_x:.2f}, y={gemini_current_y:.2f}")
                # Clear any existing plan when we get a direct command
                gemini_plan = None
              else:
                logger.info("No valid commands or plan parsed from response")

            except Exception as e:
              logger.error(f"✗ Error calling Gemini API: {e}", exc_info=True)
          else:
            logger.warning("Failed to get camera frame")

        # Execute plan if active
        if gemini_plan is not None and gemini_plan_start_time is not None:
          elapsed = time.monotonic() - gemini_plan_start_time
          # Find current step in plan
          current_commands = {"forward": False, "backward": False, "left": False, "right": False}
          for w, a, s, d, end_time in gemini_plan:
            if elapsed <= end_time:
              current_commands["forward"] = bool(w)
              current_commands["backward"] = bool(s)
              current_commands["left"] = bool(a)
              current_commands["right"] = bool(d)
              break
          else:
            # Plan finished
            gemini_plan = None
            gemini_plan_start_time = None
            current_commands = {"forward": False, "backward": False, "left": False, "right": False}

          gemini_current_x, gemini_current_y = commands_to_joystick_axes(current_commands)

        # Publish joystick messages continuously at 100Hz
        if gemini_pm is not None:
          try:
            joystick_msg = messaging.new_message('testJoystick')
            joystick_msg.valid = True
            joystick_msg.testJoystick.axes = [gemini_current_x, gemini_current_y]
            joystick_msg.testJoystick.buttons = [False]
            gemini_pm.send('testJoystick', joystick_msg)
          except Exception as e:
            logger.error(f"Failed to send joystick message: {e}")
            # Try to reinitialize messaging only if it's None
            if gemini_pm is None:
              try:
                gemini_pm = messaging.PubMaster(['testJoystick'])
              except Exception as e2:
                logger.error(f"Failed to reinitialize messaging: {e2}")

        joystick_rk.keep_time()
      else:
        # Clear plan when disabled
        gemini_plan = None
        gemini_plan_start_time = None

        if gemini_current_x != 0.0 or gemini_current_y != 0.0:
          logger.info("Gemini disabled, resetting joystick to zero")
          gemini_current_x = 0.0
          gemini_current_y = 0.0
          if gemini_pm is not None:
            try:
              joystick_msg = messaging.new_message('testJoystick')
              joystick_msg.valid = True
              joystick_msg.testJoystick.axes = [0.0, 0.0]
              joystick_msg.testJoystick.buttons = [False]
              gemini_pm.send('testJoystick', joystick_msg)
            except Exception as e:
              logger.error(f"Failed to send zero joystick message: {e}")

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
        logger.info("  - Waiting for next camera frame (every 10 seconds)")
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


async def gemini_prompt_handler(request: 'web.Request'):
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

  try:
    # Ensure Gemini starts disabled by default
    # Only enable if file exists AND contains "1"
    # (File will be created when user enables via UI)

    # Enable joystick debug mode
    from openpilot.common.params import Params
    Params().put_bool("JoystickDebugMode", True)

    # Initialize messaging for Gemini (don't fail if this fails)
    # Note: Only create one publisher - reuse if it already exists
    global gemini_pm
    if gemini_pm is None:
      try:
        gemini_pm = messaging.PubMaster(['testJoystick'])
        logger.info("Messaging initialized for Gemini")
      except Exception as e:
        logger.error(f"Failed to initialize messaging: {e}", exc_info=True)
        gemini_pm = None
    else:
      logger.info("Messaging already initialized, reusing existing publisher")

    # App needs to be HTTPS for microphone and audio autoplay to work on the browser
    ssl_context = create_ssl_context()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ping", ping, allow_head=True)
    app.router.add_post("/offer", offer)
    app.router.add_post("/sound", sound)
    app.router.add_get("/gemini", gemini_control)
    app.router.add_post("/gemini", gemini_control)
    app.router.add_get("/gemini/prompt", gemini_prompt_handler)
    app.router.add_post("/gemini/prompt", gemini_prompt_handler)
    app.router.add_static('/static', os.path.join(TELEOPDIR, 'static'))

    # Start Gemini background task when app starts
    async def startup_background_tasks(app):
      global gemini_task
      try:
        gemini_task = asyncio.create_task(gemini_loop())
        logger.info("Gemini background task started")
      except Exception as e:
        logger.error(f"Failed to start Gemini background task: {e}", exc_info=True)
        # Don't fail startup if Gemini fails - web server should still work

    app.on_startup.append(startup_background_tasks)

    logger.info("Starting web server on https://0.0.0.0:5000")
    logger.info("Web server ready - Gemini will start disabled by default")
    web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)
  except KeyboardInterrupt:
    logger.info("Web server shutting down...")
  except Exception as e:
    logger.exception(f"Fatal error starting web server: {e}")
    raise


if __name__ == "__main__":
  main()
