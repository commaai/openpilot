import asyncio
import json
import logging
import os
import ssl
import uuid
import time

# aiortc and its dependencies have lots of internal warnings :(
import warnings
warnings.resetwarnings()
warnings.simplefilter("always")

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.tools.bodyteleop.bodyav import BodyMic, WebClientSpeaker, force_codec, play_sound, MediaBlackhole, EncodedBodyVideo

logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)

pcs = set()
pm, sm = None, None
TELEOPDIR = f"{BASEDIR}/tools/bodyteleop"


async def index(request):
  content = open(TELEOPDIR + "/static/index.html", "r").read()
  request.app['mutable_vals']['prev_command'] = []

  return web.Response(content_type="text/html", text=content)


async def control_body(data, app):
  if (data['type'] == 'control_command') and (app['mutable_vals']['prev_command'] == [data['x'], data['y']] and data['x'] == 0 and data['y'] == 0):
    return

  x = max(-1.0, min(1.0, data['x']))
  y = max(-1.0, min(1.0, data['y']))
  dat = messaging.new_message()
  dat.customReservedRawData0 = json.dumps({'x': x, 'y': y}).encode()
  pm.send('customReservedRawData0', dat)
  if (data['type'] == 'control_command'):
    app['mutable_vals']['prev_command'] = [data['x'], data['y']]


async def offer(request):
  logger.info("\n\n\nnewoffer!\n\n")

  params = await request.json()
  offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
  speaker = WebClientSpeaker()
  blackhole = MediaBlackhole()

  pc = RTCPeerConnection()
  pc_id = "PeerConnection(%s)" % uuid.uuid4()
  pcs.add(pc)

  def log_info(msg, *args):
    logger.info(pc_id + " " + msg, *args)

  log_info("Created for %s", request.remote)

  @pc.on("datachannel")
  def on_datachannel(channel):
    request.app['mutable_vals']['remote_channel'] = channel

    @channel.on("message")
    async def on_message(message):
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


  @pc.on("connectionstatechange")
  async def on_connectionstatechange():
    log_info("Connection state is %s", pc.connectionState)
    if pc.connectionState == "failed":
      await pc.close()
      pcs.discard(pc)

  @pc.on('track')
  def on_track(track):
    logger.info(f"Track received: {track.kind}")
    if track.kind == "audio":
      speaker.addTrack(track)
    elif track.kind == "video":
      blackhole.addTrack(track)

    @track.on("ended")
    async def on_ended():
      log_info("Remote %s track ended", track.kind)
      if track.kind == "audio":
        await speaker.stop()
      elif track.kind == "video":
        await blackhole.stop()

  video_sender = pc.addTrack(EncodedBodyVideo())
  force_codec(pc, video_sender, forced_codec='video/H264')
  _ = pc.addTrack(BodyMic())

  await pc.setRemoteDescription(offer)
  await speaker.start()
  await blackhole.start()
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


async def run(cmd):
  proc = await asyncio.create_subprocess_shell(
    cmd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
  )
  stdout, stderr = await proc.communicate()
  logger.info("Created key and cert!")
  if stdout:
    logger.info(f'[stdout]\n{stdout.decode()}')
  if stderr:
    logger.info(f'[stderr]\n{stderr.decode()}')


def main():
  global pm, sm
  pm = messaging.PubMaster(['customReservedRawData0'])
  sm = messaging.SubMaster(['carState', 'logMessage'])
  # App needs to be HTTPS for microphone and audio autoplay to work on the browser
  cert_path = TELEOPDIR + '/cert.pem'
  key_path = TELEOPDIR + '/key.pem'
  if (not os.path.exists(cert_path)) or (not os.path.exists(key_path)):
    asyncio.run(run(f'openssl req -x509 -newkey rsa:4096 -nodes -out {cert_path} -keyout {key_path} \
                     -days 365 -subj "/C=US/ST=California/O=commaai/OU=comma body"'))
  else:
    logger.info("Certificate exists!")
  ssl_context = ssl.SSLContext()
  ssl_context.load_cert_chain(cert_path, key_path)
  app = web.Application()
  app['mutable_vals'] = {}
  app.on_shutdown.append(on_shutdown)
  app.router.add_post("/offer", offer)
  app.router.add_get("/", index)
  app.router.add_static('/static', TELEOPDIR + '/static')
  # app.on_startup.append(start_background_tasks)
  # app.on_cleanup.append(stop_background_tasks)
  web.run_app(app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context)


if __name__ == "__main__":
  main()
