#!/usr/bin/env python3

import json
import os
import time
from typing import Optional

import numpy as np
from PIL import Image

from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.panda.board.body import PandaBody

try:
  import google.generativeai as genai
except ImportError:
  cloudlog.error("google-generativeai not installed. Install with: pip install google-generativeai")
  genai = None


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
  vipc_client = None
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
  finally:
    # Note: VisionIpcClient doesn't have explicit close, connection is managed internally
    pass


def image_to_pil(image: np.ndarray, max_size: tuple[int, int] = (640, 480)) -> Image.Image:
  """Convert numpy image array to PIL Image and downscale if needed

  Args:
    image: numpy array image
    max_size: maximum (width, height) to downscale to. Default 640x480 for faster API calls.
  """
  pil_image = Image.fromarray(image)

  # Downscale if image is larger than max_size
  if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    cloudlog.debug(f"Downscaled image from {image.shape[1]}x{image.shape[0]} to {pil_image.size[0]}x{pil_image.size[1]}")

  return pil_image


def parse_gemini_response(response_text: str) -> dict:
  """Parse Gemini response to extract movement commands

  Expected format: JSON with keys like:
  {
    "forward": true/false,
    "backward": true/false,
    "left": true/false,
    "right": true/false
  }
  Or text like "W", "S", "A", "D", "WA", "SD", etc.
  """
  response_text = response_text.strip()

  # Try to parse as JSON first
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

  # Parse text commands like "W", "S", "A", "D", "WA", "SD"
  response_upper = response_text.upper()
  return {
    "forward": "W" in response_upper,
    "backward": "S" in response_upper,
    "left": "A" in response_upper,
    "right": "D" in response_upper,
  }


def apply_motor_control(panda: PandaBody, commands: dict, speed: int = 50):
  """Apply motor control based on commands

  W = forward: both motors forward
  S = backward: both motors backward
  A = left: left motor backward, right motor forward (rotate left)
  D = right: left motor forward, right motor backward (rotate right)
  WA = forward + left: left motor slower, right motor faster
  SD = backward + right: left motor faster, right motor slower
  """
  forward = commands.get("forward", False)
  backward = commands.get("backward", False)
  left = commands.get("left", False)
  right = commands.get("right", False)

  # Validate: can't have both forward and backward
  if forward and backward:
    cloudlog.warning("Invalid command: both forward and backward")
    forward = backward = False

  # Validate: can't have both left and right
  if left and right:
    cloudlog.warning("Invalid command: both left and right")
    left = right = False

  # Calculate motor speeds
  left_speed = 0
  right_speed = 0

  # Base forward/backward movement
  if forward:
    left_speed += speed
    right_speed += speed
  elif backward:
    left_speed -= speed
    right_speed -= speed

  # Rotation adjustments (applied on top of forward/backward)
  if left:
    # Turn left: reduce left motor, increase right motor
    left_speed -= speed * 0.5
    right_speed += speed * 0.5
  elif right:
    # Turn right: increase left motor, reduce right motor
    left_speed += speed * 0.5
    right_speed -= speed * 0.5

  # Clamp speeds
  left_speed = max(-100, min(100, left_speed))
  right_speed = max(-100, min(100, right_speed))

  # Apply motor control
  try:
    panda.motor_set_speed(PandaBody.MOTOR_LEFT, int(left_speed))
    panda.motor_set_speed(PandaBody.MOTOR_RIGHT, int(right_speed))
    cloudlog.info(f"Motor control: left={left_speed}, right={right_speed}")
  except Exception as e:
    cloudlog.error(f"Error setting motor speeds: {e}")


def geminid_thread():
  """Main thread for Gemini-based control"""
  if genai is None:
    cloudlog.error("google-generativeai not available. Exiting.")
    return

  # Get API key from environment
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    cloudlog.error("GEMINI_API_KEY environment variable not set")
    return

  genai.configure(api_key=api_key)

  # Use the cheapest model: gemini-1.5-flash
  model = genai.GenerativeModel('gemini-1.5-flash')

  # Connect to Panda
  try:
    panda = PandaBody()
    cloudlog.info("Connected to PandaBody")
  except Exception as e:
    cloudlog.error(f"Failed to connect to PandaBody: {e}")
    return

  # Camera connection will be done per-frame
  params = Params()

  rk = Ratekeeper(0.2, print_delay_threshold=None)  # 5 seconds = 0.2 Hz

  prompt = """You are controlling a robot with two motors in a side-by-side configuration.
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
      # Check if Gemini control is enabled
      gemini_enabled = params.get_bool("GeminiControlEnabled")
      if not gemini_enabled:
        cloudlog.debug("Gemini control disabled, waiting...")
        time.sleep(1)
        rk.keep_time()
        continue

      # Get camera frame
      frame = get_camera_frame()
      if frame is None:
        cloudlog.warning("Failed to get camera frame, skipping this cycle")
        rk.keep_time()
        continue

      # Convert to PIL Image
      pil_image = image_to_pil(frame)

      # Send to Gemini
      cloudlog.info("Sending frame to Gemini...")
      try:
        response = model.generate_content([
          prompt,
          pil_image
        ])

        response_text = response.text
        cloudlog.info(f"Gemini response: {response_text}")

        # Parse response
        commands = parse_gemini_response(response_text)

        # Apply motor control
        apply_motor_control(panda, commands, speed=50)

      except Exception as e:
        cloudlog.error(f"Error calling Gemini API: {e}")

      rk.keep_time()

    except KeyboardInterrupt:
      cloudlog.info("Shutting down geminid")
      break
    except Exception as e:
      cloudlog.exception(f"Error in geminid loop: {e}")
      time.sleep(1)


def main():
  geminid_thread()


if __name__ == "__main__":
  main()

