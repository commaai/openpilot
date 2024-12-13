import asyncio
import threading
import json
import queue
import subprocess
import logging

from aiortc import (
  RTCPeerConnection, RTCConfiguration, RTCSessionDescription,
  RTCDataChannel, RTCRtpCodecCapability, RTCIceServer
)
from aiortc.sdp import candidate_from_sdp
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
from openpilot.system.manager.process_config import NativeProcess
from openpilot.common.params import Params
from openpilot.common.api import Api, api_get

# Configure logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_offer(pc: RTCPeerConnection) -> None:
  """
  Create an SDP offer and set it as the local description of the provided PeerConnection.
  """
  if not pc:
    logger.error("PeerConnection is None, cannot create offer.")
    return
  try:
    offer: RTCSessionDescription = await pc.createOffer()
    await pc.setLocalDescription(offer)
    logger.info("Created and set local SDP offer.")
  except Exception:
    logger.exception("Failed to create or set local SDP offer:")
    if pc:
      await pc.close()

async def set_answer(pc: RTCPeerConnection, data: dict) -> None:
  """
  Set the remote answer SDP on the provided PeerConnection.
  """
  if not pc:
    logger.error("PeerConnection is None, cannot set answer.")
    return
  try:
    if 'sdp' not in data or 'type' not in data:
      raise ValueError("Answer data is missing 'sdp' or 'type' field.")
    logger.debug("Received answer data: %s", data)
    answer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
    await pc.setRemoteDescription(answer)
    logger.info("Successfully set remote answer description.")
  except Exception:
    logger.exception("Failed to set remote answer. Data: %s", data)
    if pc:
      await pc.close()

async def set_candidate(pc: RTCPeerConnection, candidate_data: dict) -> None:
  """
  Add a remote ICE candidate to the provided PeerConnection.
  """
  if not pc:
    logger.error("PeerConnection is None, cannot set candidate.")
    return
  logger.debug("Received candidate data: %s", candidate_data)
  try:
    if "candidate" not in candidate_data:
      raise ValueError("Candidate data missing 'candidate' field.")
    candidate_sdp = candidate_data["candidate"]
    parsed_candidate = candidate_from_sdp(candidate_sdp)
    parsed_candidate.sdpMid = candidate_data.get("sdpMid", None)
    parsed_candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex", None)
    await pc.addIceCandidate(parsed_candidate)
    logger.info("Added remote ICE candidate.")
  except Exception:
    logger.exception("Failed to add ICE candidate:")
    if pc:
      await pc.close()

def capture_pane(session_window_pane: str) -> str | None:
  """
  Capture the output of a tmux pane and return it as JSON.
  """
  try:
    result = subprocess.run(
      ["tmux", "capture-pane", "-peJS-7200", "-t", session_window_pane],
      capture_output=True,
      text=True,
      check=False
    )
    if result.returncode != 0:
      logger.warning("tmux capture-pane returned code %d: %s", result.returncode, result.stderr)
      return None
    return json.dumps({"tmuxCapture": result.stdout})
  except Exception:
    logger.exception("Exception while capturing pane '%s':", session_window_pane)
    return None

class Streamer:
  def __init__(self, sdp_send_queue: queue.Queue, sdp_recv_queue: queue.Queue, ice_send_queue: queue.Queue):
    logger.info("Initializing Streamer instance.")
    self.lock = asyncio.Lock()
    self.pc: RTCPeerConnection | None = None
    self.data_channel: RTCDataChannel | None = None
    self.sdp_send_queue = sdp_send_queue
    self.sdp_recv_queue = sdp_recv_queue
    self.ice_send_queue = ice_send_queue
    self.camera: NativeProcess | None = None
    self.encoder: NativeProcess | None = None
    self.params = Params()
    self.onroad = self.params.get_bool("IsOnroad")
    self.api = Api(self.params.get("DongleId", encoding='utf8'))

    # Initialize video tracks
    self.tracks: dict[str, LiveStreamVideoStreamTrack] = {}
    if self.params.get_bool("RecordFront"):
      self.tracks["driver"] = LiveStreamVideoStreamTrack("driver")

    if self.params.get_bool("RecordRoad"):
      self.tracks["road"] = LiveStreamVideoStreamTrack("road")
      self.tracks["wideRoad"] = LiveStreamVideoStreamTrack("wideRoad")

  def add_tracks(self) -> None:
    """
    Add video tracks to the PeerConnection with H264 codec preferences.
    """
    if not self.pc:
      logger.error("Cannot add tracks without a PeerConnection.")
      return

    try:
      for track in self.tracks.values():
        track.paused |= self.onroad
        transceiver = self.pc.addTransceiver("video", direction="sendonly")
        h264_capability = RTCRtpCodecCapability(
          mimeType="video/H264",
          clockRate=90000,
          parameters={
            "level-asymmetry-allowed": "1",
            "packetization-mode": "1",
            "profile-level-id": "42e01f"
          }
        )
        transceiver.setCodecPreferences([h264_capability])
        self.pc.addTrack(track)
      logger.info("All video tracks added successfully.")
    except Exception:
      logger.exception("Failed to add tracks:")
      if self.pc:
        asyncio.ensure_future(self.pc.close())

  def send_track_states(self) -> None:
    """
    Send the current track pause states via the data channel.
    """
    if not self.data_channel:
      logger.warning("Data channel not established. Cannot send track states.")
      return

    track_state = {
      "trackState": {name: track.paused for name, track in self.tracks.items()}
    }

    try:
      self.data_channel.send(json.dumps(track_state))
      logger.debug("Sent updated track states: %s", track_state)
    except Exception:
      logger.exception("Failed to send track states:")

  def attach_event_handlers(self):

    def on_open():
      self.send_track_states()

    def on_message(message: str):
      logger.debug("Received data channel message: %s", message)
      try:
        msg = json.loads(message)
        action = msg.get("action")
        track_type = msg.get("trackType")
        updated = False

        if action in ("startTrack", "stopTrack") and track_type in self.tracks:
          self.tracks[track_type].paused = (action == "stopTrack")
          updated = True
          logger.info("Track '%s' %s", track_type, "stopped" if self.tracks[track_type].paused else "started")
        elif action == "captureTmux":
          capture_result = capture_pane("comma:0.0")
          if capture_result:
            try:
              self.data_channel.send(capture_result) # type: ignore[union-attr]
              logger.debug("Sent tmux capture result.")
            except Exception:
              logger.exception("Failed to send tmux capture result:")
          else:
            logger.warning("No tmux capture result to send.")

        if updated:
          self.send_track_states()

      except Exception:
        logger.exception("Error handling data channel message:")

    async def on_close():
      logger.info("Data channel closed. Stopping streamer...")
      await self.stop()

    async def on_negotiationneeded():
      logger.debug("Negotiation needed. Creating new SDP offer.")
      await create_offer(self.pc)

    self.pc.on("negotiationneeded", on_negotiationneeded)
    self.data_channel.on("open", on_open)
    self.data_channel.on("message", on_message)
    self.data_channel.on("close", on_close)

  def get_ice_configuration(self):
    try:
      ice_servers_data = api_get('/v1/iceservers', timeout=5,access_token=self.api.get_token()).content
      ice_servers = [
        RTCIceServer(urls=server["urls"],username=server["username"],credential=server["credential"])
        for server in json.loads(ice_servers_data)
      ]
    except Exception:
      logger.exception("Failed to fetch ICE servers:")
      # Fallback to Google STUN server
      ice_servers = [
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
      ]

    configuration = RTCConfiguration(iceServers=ice_servers)
    logger.debug("RTCConfiguration: %s", configuration)
    return configuration

  async def build(self):
    self.pc = RTCPeerConnection(self.get_ice_configuration())
    self.add_tracks()
    self.data_channel = self.pc.createDataChannel("data")
    self.attach_event_handlers()
    await create_offer(self.pc)
    while not self.pc.localDescription:
      await asyncio.sleep(0.1)
    while self.pc.iceGatheringState != "complete":
      await asyncio.sleep(0.1)
    message = json.dumps({
      'type': self.pc.localDescription.type,
      'sdp': self.pc.localDescription.sdp,
    })
    self.sdp_send_queue.put_nowait(message)

  async def stop(self) -> None:
    """
    Stop the PeerConnection, camera, encoder, and clean up state.
    """
    async with self.lock:
      logger.info("Stopping streamer...")
      if self.camera:
        self.camera.stop()
      if self.encoder:
        self.encoder.stop()
      try:
        # Clear any pending messages in sdp_send_queue
        while not self.sdp_send_queue.empty():
          self.sdp_send_queue.get()

        if self.data_channel is not None:
          self.data_channel.close()
          self.data_channel = None
        if self.pc:
          await self.pc.close()
          self.pc = None

        await asyncio.sleep(1)
        logger.info("Streamer stopped successfully.")
      except Exception:
        logger.exception("Error during stop:", )

  async def event_loop(self, end_event: threading.Event):
    """
    Main event loop that processes signaling messages and maintains the PeerConnection.
    Runs until end_event is set.
    """
    self.camera = NativeProcess("camerad", "system/camerad", ["./camerad"], True)
    self.encoder = NativeProcess("encoderd", "system/loggerd", ["./encoderd", "--stream"], True)
    logger.info("Native processes for camera and encoder initialized.")

    stop_states = ['failed', 'closed']
    connecting_states = ['connecting', 'new']

    while not end_event.is_set():
      self.onroad = self.params.get_bool("IsOnroad") # support some functions while onroad
      try:
        try:
          data = self.sdp_recv_queue.get_nowait()
        except queue.Empty:
          data = None

        if data:
          logger.debug("Received signaling message: %s", data)
          interaction_timeout = 6000 # interaction_timeout 10mins
          match data.get('type'):
            case 'start':
              try:
                await asyncio.wait_for(self.build(), timeout=30)
                connection_timeout = 600 # 1min
                if not self.onroad:
                  self.camera.start()
                  self.encoder.start()
              except Exception:
                logger.exception("Error during 'start' handling:")
                await self.stop()

            case 'answer':
              await set_answer(self.pc, data)
            case 'candidate' if 'candidate' in data:
              await set_candidate(self.pc, data['candidate'])
            case 'bye':
              await self.stop()
        else:

          await asyncio.sleep(0.1 if self.pc else 2)

        if self.pc:
          transeivers = self.pc.getTransceivers()
          dtls_state = None
          if len(transeivers):
            dtls_state = transeivers[0].receiver.transport.state

          if self.pc.connectionState in stop_states or dtls_state in stop_states:
            raise TimeoutError("The connection ended")
          if self.pc.connectionState in connecting_states or dtls_state in connecting_states:
            if connection_timeout:
              connection_timeout -= 1
            else:
              raise TimeoutError("Connection took too long to establish. Closing")
          if interaction_timeout:
            interaction_timeout -= 1
          else:
            self.data_channel.send("bye") # type: ignore[union-attr]
            raise TimeoutError("Interaction timeout. Closing")

      except Exception:
        logger.exception("Stopping:")
        await self.stop()
    await self.stop()
