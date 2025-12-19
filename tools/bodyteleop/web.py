import asyncio
import dataclasses
import io
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
from openpilot.common.swaglog import cloudlog
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.tools.bodyteleop.gemini_plan import parse_plan_from_text, compute_next_gemini_call_time

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
gemini_current_x = 0.0
gemini_current_y = 0.0
gemini_task = None
gemini_last_response = ""
gemini_last_summary = ""
gemini_enabled = False  # In-memory state, starts disabled
gemini_prompt = ""  # In-memory prompt
gemini_plan = None  # Current plan being executed [(w, a, s, d, end_time), ...]
gemini_plan_start_time = None  # When the current plan started
gemini_plan_id = 0  # increments every time we receive a new plan


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
  """Convert numpy image array to PIL Image and downscale before sending to Gemini"""
  pil_image = Image.fromarray(image)

  # Always downscale by half to reduce tokens + latency (area becomes ~1/4)
  w, h = pil_image.size
  half_size = (max(1, w // 2), max(1, h // 2))
  if half_size != (w, h):
    pil_image = pil_image.resize(half_size, Image.Resampling.LANCZOS)

  if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    cloudlog.debug(f"Downscaled image from {image.shape[1]}x{image.shape[0]} to {pil_image.size[0]}x{pil_image.size[1]}")

  return pil_image


def pil_to_jpeg_part(pil_image: Image.Image, quality: int = 65) -> tuple[dict, int]:
  """
  Convert PIL image to a compact JPEG payload for Gemini.
  Returns (part_dict, nbytes).
  """
  buf = io.BytesIO()
  pil_image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
  data = buf.getvalue()
  return ({"mime_type": "image/jpeg", "data": data}, len(data))


def parse_gemini_response(response_text: str) -> Optional[list]:
  # Kept for backwards compatibility within this module
  return parse_plan_from_text(response_text)


def extract_gemini_summary(response_text: str) -> str:
  """
  Extract a short, structured summary line from the model output.
  Expected format: a single line starting with "summary:" (case-insensitive).
  """
  if not response_text:
    return ""
  for ln in response_text.splitlines():
    s = ln.strip()
    if not s:
      continue
    low = s.lower()
    if low.startswith("summary:"):
      out = s.split(":", 1)[1].strip()
      # Keep it short and single-line for UI
      out = " ".join(out.split())
      return out[:240]
  return ""


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
  # Prefer Pro first (better reasoning), then fall back to Flash variants.
  # Note: model availability can vary by account/region; we try in order and fall back gracefully.
  preferred_models = ['gemini-2.5-pro', 'gemini-2.0-pro', 'gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
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

  # IMPORTANT: This task runs on the same asyncio event loop as the aiohttp web server.
  # Never use blocking sleeps (time.sleep) here, or the web UI will hang/time out.
  loop_hz = 100.0
  loop_dt = 1.0 / loop_hz
  last_gemini_call_time = 0.0
  gemini_call_interval = 10.0  # Minimum time between Gemini calls (seconds)

  logger.info("Gemini loop ready, waiting for enable...")

  default_prompt = """You are controlling a robot. Analyze the image and output ONLY a structured response with:
1) a brief single-line intention/notes summary, and
2) a movement plan.
No extra text outside that structure.

Movement controls (only ONE direction at a time):
- W: Move forward (~5 mph)
- S: Move backward (~5 mph)
- A: Rotate left (~360 deg/s, keep turns SHORT: 0.1-0.3s)
- D: Rotate right (~360 deg/s, keep turns SHORT: 0.1-0.3s)

Default behavior (unless overridden by Additional instructions):
- Explore and move around to see more of the environment. Prefer safe, gentle exploration rather than standing still.
- If there is a clear open path, make progress forward and occasionally do small turns to look around.

Planning quality / nuance requirements:
- Do NOT output a trivial plan like "all W until 10.0s" unless the scene is extremely open and safe.
- You do NOT need to use only short forward bursts; longer forward segments are OK when the path is clearly open and safe.
- Use short turns (A/D 0.1-0.3s) to re-center in corridors/lanes, avoid nearby obstacles, and aim around corners; make these adjustments proactively.
- If forward motion looks risky/blocked/uncertain, slow down: use stop steps and/or short turns instead of committing to long forward motion.
- Obstacle avoidance is the top priority: if any obstacle/person/object is close ahead or to the side, DO NOT drive into it. Instead stop, turn away, or back up briefly to create clearance, then proceed.

Stuck handling / repeated view:
- If you appear stuck (no progress, blocked, wedged, or not moving), you MUST include at least one backup segment (S) in the plan (typically early).
  Recommended: start with S for 0.4-0.8s, then a short turn (A/D 0.1-0.3s), then reassess/continue.
- If the view looks essentially unchanged from the previous request(s) (same frame / same scene), assume you're not moving and DO NOT keep pushing forward.
  Instead: S (0.4-0.8s) + small turn (A/D 0.1-0.3s) + brief stop (0.1-0.3s), then proceed.
- If you are facing a mostly solid color / textureless close-up (e.g., a wall very close, overexposed surface, or you can't see the environment), treat that as stuck:
  immediately back up (S 0.6-1.0s), then turn (A/D 0.2-0.4s), then stop briefly to reassess.

Output format (EXACTLY):
summary: <one short sentence (<= 140 chars, no newlines) describing your intention + anything noteworthy (e.g. obstacle, stuck, turning, exploring)>
plan
<w,a,s,d,t>
<w,a,s,d,t>
...

STRICT OUTPUT RULES (MUST FOLLOW):
- Do NOT describe the image. Do NOT write bullet points. Do NOT write markdown. Do NOT add numbering.
- The line must be exactly `plan` (no colon, no extra whitespace).
- After the `plan` line, output ONLY CSV lines of exactly: w,a,s,d,t (4 commas). No labels, no extra text.
- If you are uncertain, output a safe minimal plan:
  summary: uncertain, pausing
  plan
  0,0,0,0,0.1

EXAMPLE OUTPUT (copy this structure exactly, but choose actions based on the current image):
summary: exploring forward, slight right to avoid obstacle on left, pause to reassess
plan
1,0,0,0,0.40
1,0,0,0,0.80
0,0,0,0,1.00
0,0,0,1,1.20
0,0,0,0,1.35
1,0,0,0,1.95
0,0,0,0,2.10
0,1,0,0,2.30
0,0,0,0,2.45
1,0,0,0,3.05
0,0,0,0,3.20
0,0,0,1,3.40
0,0,0,0,3.55
1,0,0,0,4.15
0,0,0,0,4.30
0,1,0,0,4.50
0,0,0,0,4.65
1,0,0,0,5.25
0,0,0,0,5.40
0,0,0,1,5.60
0,0,0,0,5.75
1,0,0,0,6.35
0,0,0,0,6.50
0,1,0,0,6.70
0,0,0,0,6.85
1,0,0,0,7.45
0,0,0,0,7.60
0,0,0,1,7.80
0,0,0,0,7.95
1,0,0,0,8.55
0,0,0,0,8.70
0,1,0,0,8.90
0,0,0,0,9.05
1,0,0,0,9.65
0,0,0,0,9.80
1,0,0,0,9.90
0,0,0,0,10.00

EXAMPLE OUTPUT (stuck / repeated view):
summary: stuck, backing up then turning to free clearance
plan
0,0,1,0,0.60
0,0,0,1,0.80
0,0,0,0,1.00
1,0,0,0,1.80
0,0,0,0,2.00
0,1,0,0,2.20
0,0,0,0,2.40
1,0,0,0,3.40
0,0,0,0,3.60
0,0,0,1,3.80
0,0,0,0,4.00
1,0,0,0,5.00
0,0,0,0,5.20
0,1,0,0,5.40
0,0,0,0,5.60
1,0,0,0,6.60
0,0,0,0,6.80
0,0,0,1,7.00
0,0,0,0,7.20
1,0,0,0,8.40
0,0,0,0,8.60
0,1,0,0,8.80
0,0,0,0,9.00
1,0,0,0,9.80
0,0,0,0,10.00

Format: w,a,s,d,t where:
- w,a,s,d are 1 (on) or 0 (off) - only ONE should be 1 per line
- t is cumulative time in seconds from plan start (not duration)
- The plan MUST start near t=0: the FIRST line's t should be a small positive value (e.g. 0.2-0.6), not 10.0
- The plan MUST end at t=10.0: the LAST line's t must be exactly 10.0 (unless no movement needed)
- Times must be strictly increasing each line (cumulative)
- When movement is needed, output AT LEAST 5 steps (5+ lines after "plan"). Use small increments (e.g. 0.2-1.0s per step) and end exactly at 10.0.
- Target plan duration: ~10 seconds when movement is needed (use multiple short steps that add up close to 10.0s)
- Maximum plan duration: 10 seconds (never exceed)
- Example: "1,0,0,0,1.0" means W until 1.0s, then "0,0,0,1,1.3" means D until 1.3s total
- If no movement needed, output: plan\n0,0,0,0,0.1

Do NOT describe the image. Output ONLY the summary line and the plan."""

  global gemini_plan, gemini_plan_start_time, gemini_current_x, gemini_current_y, gemini_plan_id
  global gemini_last_summary

  while True:
    loop_start_t = time.monotonic()
    try:
      gemini_enabled = gemini_is_enabled()

      if gemini_enabled:
        current_time = time.monotonic()

        next_allowed_call_time = compute_next_gemini_call_time(
          last_gemini_call_time,
          gemini_call_interval,
          gemini_plan_start_time,
          gemini_plan,
        )

        # Only request a new plan when we're idle (no plan currently being executed)
        if current_time >= next_allowed_call_time and gemini_plan is None:
          # Mock mode for debugging (set GEMINI_MOCK=1 to enable)
          plan = None
          if os.getenv("GEMINI_MOCK") == "1":
            logger.info("MOCK MODE: Using fake Gemini response (skipping camera)")
            response_text = """summary: mock plan (forward then left)
plan
1,0,0,0,1.0
0,1,0,0,2.0
0,0,0,0,2.1"""
            gemini_last_response = response_text
            gemini_last_summary = extract_gemini_summary(response_text)
            logger.info(f"✓ Mock Gemini response:")
            logger.info(f"{response_text}")
            plan = parse_gemini_response(response_text)
            logger.info(f"✓ Parsed plan result: {plan}")
          else:
            logger.info("Getting camera frame...")
            # Offload potentially-blocking vision IPC + image handling to a thread to keep aiohttp responsive.
            frame = await asyncio.to_thread(get_camera_frame)
            if frame is not None:
              logger.info(f"Got frame: {frame.shape}")
              custom_prompt = gemini_get_prompt()
              # Stateless request: full instructions each time (no chat history).
              active_prompt = default_prompt
              if custom_prompt:
                active_prompt = active_prompt + "\n\nAdditional instructions: " + custom_prompt
                logger.info(f"Using default + custom prompt (custom length: {len(custom_prompt)})")
              else:
                logger.info("Using default prompt (no custom prompt)")

              pil_image = await asyncio.to_thread(image_to_pil, frame)
              # Compress image to JPEG to reduce upload + latency.
              jpeg_quality = int(os.getenv("GEMINI_JPEG_QUALITY", "65"))
              image_part, image_nbytes = await asyncio.to_thread(pil_to_jpeg_part, pil_image, jpeg_quality)
              logger.info(f"Image prepared: {pil_image.size}, jpeg_bytes={image_nbytes}, q={jpeg_quality}")

              logger.info("Sending frame to Gemini API...")
              try:
                # Offload the blocking Gemini SDK call to a thread to avoid stalling the event loop.
                # Send image + full prompt every time (stateless).
                response = await asyncio.to_thread(model.generate_content, [active_prompt, image_part])
                response_text = response.text
                gemini_last_response = response_text
                gemini_last_summary = extract_gemini_summary(response_text)

                # Print full response in terminal
                logger.info("✓ Gemini response received:")
                logger.info(f"{response_text}")

                plan = parse_gemini_response(response_text)

              except Exception as e:
                logger.error(f"✗ Error calling Gemini API: {e}", exc_info=True)
            else:
              logger.warning("Failed to get camera frame")

          if plan is not None:
            # Start executing plan
            gemini_plan = plan
            gemini_plan_start_time = time.monotonic()
            # Pace Gemini calls off plan start time so a slow Gemini response doesn't
            # artificially add extra delay on top of the plan duration.
            last_gemini_call_time = gemini_plan_start_time
            gemini_plan_id += 1
            plan_str = "\n".join([f"  {w},{a},{s},{d},{t:.2f}" for w, a, s, d, t in plan])
            logger.info(f"✓ SETTING PLAN: plan_id={gemini_plan_id}, steps={len(plan)}, start_time={gemini_plan_start_time}")
            logger.info(f"✓ Plan details:")
            logger.info(f"{plan_str}")
            logger.info(f"✓ Plan will be available for browser to fetch")
          else:
            logger.warning("No valid plan parsed from response")
            # Avoid hammering the API if we keep getting invalid output.
            last_gemini_call_time = time.monotonic()
            gemini_plan = None
            gemini_plan_start_time = None
            # Reset joystick when plan parsing fails
            gemini_current_x = 0.0
            gemini_current_y = 0.0

        # Execute plan if active
        if gemini_plan is not None and gemini_plan_start_time is not None:
          elapsed = time.monotonic() - gemini_plan_start_time
          # Find current step in plan (times are cumulative from start)
          current_commands = {"forward": False, "backward": False, "left": False, "right": False}
          plan_active = False
          for w, a, s, d, end_time in gemini_plan:
            if elapsed <= end_time:
              current_commands["forward"] = bool(w)
              current_commands["backward"] = bool(s)
              current_commands["left"] = bool(a)
              current_commands["right"] = bool(d)
              plan_active = True
              break

          if not plan_active:
            # Plan finished: clear immediately so we can request the next plan right away.
            # The browser should have already fetched the plan during execution (it polls frequently).
            logger.info("Plan finished, clearing it (ready for next plan)")
            gemini_plan = None
            gemini_plan_start_time = None
            current_commands = {"forward": False, "backward": False, "left": False, "right": False}

          gemini_current_x, gemini_current_y = commands_to_joystick_axes(current_commands)
          logger.debug(f"Plan execution: elapsed={elapsed:.2f}s, x={gemini_current_x:.2f}, y={gemini_current_y:.2f}")

        # Gemini values are sent via WebRTC data channel from the browser
        # The browser polls /gemini endpoint and updates getXY() in controls.js

        # Keep a steady loop rate without blocking the event loop.
        elapsed = time.monotonic() - loop_start_t
        await asyncio.sleep(max(0.0, loop_dt - elapsed))
      else:
        # Clear plan when disabled
        gemini_plan = None
        gemini_plan_start_time = None

        if gemini_current_x != 0.0 or gemini_current_y != 0.0:
          logger.info("Gemini disabled, resetting joystick to zero")
          gemini_current_x = 0.0
          gemini_current_y = 0.0
          # Reset handled via browser polling /gemini endpoint

        # When disabled, run slower and yield to the web server.
        await asyncio.sleep(0.1)

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
    # Ensure plan_id is always an integer
    plan_id = gemini_plan_id if gemini_plan_id is not None else 0

    # Capture plan at request time to avoid race conditions
    plan_at_request = gemini_plan
    plan_start_at_request = gemini_plan_start_time

    # Build current_plan list
    current_plan_list = None
    if plan_at_request:
      try:
        current_plan_list = [[w, a, s, d, t] for w, a, s, d, t in plan_at_request]
        logger.debug(f"GET /gemini: Built plan list with {len(current_plan_list)} steps")
      except Exception as e:
        logger.error(f"GET /gemini: Error building plan list: {e}", exc_info=True)
        current_plan_list = None

    status_info = {
      "enabled": enabled,
      "prompt": prompt,
      "api_key_set": bool(os.getenv("GEMINI_API_KEY")),
      "genai_available": genai is not None,
      "current_x": gemini_current_x,
      "current_y": gemini_current_y,
      "summary": gemini_last_summary if gemini_last_summary else "",
      "last_response": gemini_last_response if gemini_last_response else "",  # Full response
      "current_plan": current_plan_list,
      "current_plan_id": plan_id,
      "plan_active": plan_at_request is not None and plan_start_at_request is not None,
      "plan_elapsed": (time.monotonic() - plan_start_at_request) if (plan_start_at_request is not None) else None,
    }
    plan_steps = len(plan_at_request) if plan_at_request else 0
    # This endpoint is polled frequently (200ms when enabled), so keep logging at debug to avoid log spam.
    logger.debug(f"GET /gemini: enabled={enabled}, plan_id={plan_id}, has_plan={plan_at_request is not None}, plan_steps={plan_steps}")
    if plan_at_request:
      logger.debug(f"  ✓ SENDING PLAN TO BROWSER: plan_id={plan_id}, steps={plan_steps}")
      logger.debug(f"  Plan steps (first 3): {[f'{w},{a},{s},{d},{t:.2f}' for w,a,s,d,t in plan_at_request[:3]]}")
      logger.debug(f"  current_plan in JSON: {status_info['current_plan'] is not None}, length={len(status_info['current_plan']) if status_info['current_plan'] else 0}")
      logger.debug(f"  JSON serialized plan type: {type(status_info['current_plan'])}")
    else:
      logger.debug("  ✗ NO PLAN TO SEND (gemini_plan is None at request time)")
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
  global gemini_task

  try:
    # Ensure Gemini starts disabled by default
    # Only enable if file exists AND contains "1"
    # (File will be created when user enables via UI)

    # Enable joystick debug mode
    from openpilot.common.params import Params
    Params().put_bool("JoystickDebugMode", True)

    # Gemini sends joystick commands via WebRTC data channel (same as keyboard input)
    # No PubMaster needed - browser polls /gemini endpoint and sends via data channel

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
        # Start in background - don't await, let it run independently
        gemini_task = asyncio.create_task(gemini_loop())
        logger.info("Gemini background task started")
      except Exception as e:
        logger.error(f"Failed to start Gemini background task: {e}", exc_info=True)
        # Don't fail startup if Gemini fails - web server should still work
        gemini_task = None

    app.on_startup.append(startup_background_tasks)

    # Also handle shutdown
    async def shutdown_background_tasks(app):
      global gemini_task
      if gemini_task is not None and isinstance(gemini_task, asyncio.Task):
        gemini_task.cancel()
        try:
          await gemini_task
        except (asyncio.CancelledError, AttributeError, TypeError):
          pass
        logger.info("Gemini background task stopped")

    app.on_shutdown.append(shutdown_background_tasks)

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
