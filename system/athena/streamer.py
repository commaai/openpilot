import asyncio
import threading
import json
import queue
import subprocess
import logging
import pty
import os
from aiortc import (
  RTCPeerConnection, RTCConfiguration, RTCSessionDescription,
  RTCDataChannel, RTCRtpCodecCapability, RTCIceServer
)
from aiortc.sdp import candidate_from_sdp
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
from openpilot.system.manager.process_config import NativeProcess
from openpilot.common.params import Params
from openpilot.common.api import Api, api_get
import cereal.messaging as messaging


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

class RemoteShellHandler:
    """
    Allows remote shell command execution via the RTC data channel.
    Streams stdout/stderr output in real time.
    Maintains a persistent shell session using a PTY.
    """
    def __init__(self, send_func, loop=None, timeout=30):
        self.send_func = send_func
        self.loop = loop or asyncio.get_event_loop()
        self.timeout = timeout
        self.proc = None
        self._output_task = None
        self._lock = asyncio.Lock()
        self._start_shell()

    def _start_shell(self):
        self.proc = asyncio.ensure_future(self._launch_shell())

    async def _launch_shell(self):
        import os
        import pty
        import fcntl
        import termios

        master_fd, slave_fd = pty.openpty()

        # Set the slave PTY as the controlling terminal for the child process
        def preexec():
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

        self.shell = await asyncio.create_subprocess_exec(
            "/bin/bash",
            cwd="/data/openpilot",
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            preexec_fn=preexec
        )
        os.close(slave_fd)
        self.master_fd = master_fd
        self._output_task = asyncio.create_task(self._stream_output(master_fd))

    async def handle_message(self, msg):
        action = msg.get('action')
        if action == 'shell_input':
            data = msg.get('data')
            if data is not None:
                await self._send_input(data)
            return True
        elif action == 'shell':
            command = msg.get('command')
            if not command:
                await self._send_result("No command provided.", error=True, done=True)
                return True
            await self._send_command(command)
            return True
        return False

    async def _send_command(self, command):
        if not hasattr(self, 'master_fd'):
            await self._send_result("Shell not available.", error=True, done=True)
            return
        try:
            os.write(self.master_fd, (command + "\n").encode())
        except Exception as e:
            await self._send_result(f"Shell error: {e}", error=True, done=True)

    async def _send_input(self, data):
        if not hasattr(self, 'master_fd'):
            await self._send_result("Shell not available.", error=True, done=True)
            return
        try:
            os.write(self.master_fd, data.encode())
        except Exception as e:
            await self._send_result(f"Shell input error: {e}", error=True, done=True)

    async def _stream_output(self, master_fd):
        loop = asyncio.get_event_loop()
        try:
            while True:
                data = await loop.run_in_executor(None, os.read, master_fd, 1024)
                if not data:
                    break
                await self._send_result(data.decode(errors='replace'), error=False, done=False)
        except Exception as e:
            await self._send_result(f"Shell output error: {e}", error=True, done=False)

    async def _send_result(self, output, error=False, done=False):
        result = json.dumps({
            "action": "shell_result",
            "output": output,
            "error": error,
            "done": done,
        })
        if asyncio.iscoroutinefunction(self.send_func):
            await self.send_func(result)
        else:
            self.send_func(result)

    async def close(self):
        if hasattr(self, 'shell') and self.shell:
            self.shell.terminate()
            await self.shell.wait()
        if self._output_task:
            self._output_task.cancel()
        if hasattr(self, 'master_fd'):
            os.close(self.master_fd)


class ControllerHandler:
    """
    Safety-focused handler for Xbox controller inputs using asyncio.
    """
    def __init__(self, safety_timeout: float = 0.5):
        self._throttle = 0.0
        self._steering = 0.0
        self._safety_timeout = safety_timeout
        self._input_active = False
        self._lock = asyncio.Lock()
        self._running = True

        self._joystick_sock = messaging.pub_sock('testJoystick')
        self._safety_task = asyncio.create_task(self._monitor())
        logger.info("ControllerHandler initialized with safety timeout: %.2f seconds", self._safety_timeout)

    async def handle_message(self, msg: dict) -> bool:
        """
        Process a controller input message.

        Args:
            msg: Message dict with 'action', 'throttle', 'steering'.

        Returns:
            True if the message was handled as a controller action.
        """
        if msg.get('action') != 'controller':
            return False

        throttle = float(msg.get('throttle', 0.0))
        steering = float(msg.get('steering', 0.0))

        async with self._lock:
            self._throttle = throttle
            self._steering = steering
            self._input_active = True

        logger.info("Controller input: throttle=%.2f, steering=%.2f", throttle, steering)
        await self._apply(throttle, steering)
        return True

    async def close(self):
        """
        Safely shutdown the handler and reset controls.
        """
        self._running = False
        self._safety_task.cancel()
        try:
            await self._safety_task
        except asyncio.CancelledError:
            pass
        await self._reset()

    async def _monitor(self):
        """
        Reset controls if no input is received within the safety timeout.
        """
        while self._running:
            await asyncio.sleep(self._safety_timeout)
            async with self._lock:
                if not self._input_active:
                    await self._apply(0.0, 0.0)
                self._input_active = False

    async def _apply(self, throttle: float, steering: float):
        """
        Publish control values to the joystick socket.
        """
        logger.debug("Applying control: throttle=%.2f, steering=%.2f", throttle, steering)
        msg = messaging.new_message('testJoystick')
        msg.testJoystick.axes = [throttle, steering]
        msg.testJoystick.buttons = [False]
        self._joystick_sock.send(msg.to_bytes())

    async def _reset(self):
        """
        Reset controls to neutral.
        """
        async with self._lock:
            self._throttle = 0.0
            self._steering = 0.0
            await self._apply(0.0, 0.0)

class Streamer:
  def __init__(self, sdp_send_queue: queue.Queue, sdp_recv_queue: queue.Queue, ice_send_queue: queue.Queue):
    logger.info("Initializing Streamer instance.")
    self.lock = asyncio.Lock()
    self.pc: RTCPeerConnection | None = None
    self.data_channel: RTCDataChannel | None = None
    self.controller_handler: ControllerHandler | None = None
    self.shell_handler: RemoteShellHandler | None = None
    self.sdp_send_queue = sdp_send_queue
    self.sdp_recv_queue = sdp_recv_queue
    self.ice_send_queue = ice_send_queue
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

    async def on_open():
      self.send_track_states()
      if self.controller_handler is not None:
        await self.controller_handler.close()
      if self.params.get_bool("RemoteAccess"):
        self.shell_handler = RemoteShellHandler(self.data_channel.send)
        self.controller_handler = ControllerHandler(safety_timeout=0.5)

    def on_message(message: str):
      logger.debug("Received data channel message: %s", message)
      try:
        msg = json.loads(message)
        action = msg.get("action")
        track_type = msg.get("trackType")

        if action == "controller":
          # Process controller input and apply controls with safety measures
          if not self.params.get_bool("JoystickDebugMode"):
            self.params.put_bool("JoystickDebugMode", True)
          asyncio.create_task(self.controller_handler.handle_message(msg))
          return # Don't send track states

        if action == "shell" or action == "shell_input" and self.shell_handler is not None:
          asyncio.create_task(self.shell_handler.handle_message(msg))
          return

        if action in ("startTrack", "stopTrack") and track_type in self.tracks:
          self.tracks[track_type].paused = (action == "stopTrack")
          logger.info("Track '%s' %s", track_type, "stopped" if self.tracks[track_type].paused else "started")
          self.send_track_states()

        if action == "captureTmux":
          capture_result = capture_pane("comma:0.0")
          if capture_result:
            try:
              self.data_channel.send(capture_result) # type: ignore[union-attr]
              logger.debug("Sent tmux capture result.")
            except Exception:
              logger.exception("Failed to send tmux capture result:")
          else:
            logger.warning("No tmux capture result to send.")

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
    #TODO for privacy, we can disable the turn server and always use the stun server. This forces a local connection.
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
      self.params.put_bool("LiveStreamRunning", False)
      self.params.put_bool("JoystickDebugMode", False)
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
        if self.controller_handler:
          await self.controller_handler.close()
          self.controller_handler = None
        if self.shell_handler:
          await self.shell_handler.close()
          self.shell_handler = None

        await asyncio.sleep(1)
        logger.info("Streamer stopped successfully.")
      except Exception:
        logger.exception("Error during stop:", )

  async def event_loop(self, exit_event: threading.Event):
    """
    Main event loop that processes signaling messages and maintains the PeerConnection.
    Runs until exit_event is set. Should keep running even if webbsocket connection is lost.
    """
    self.camera = NativeProcess("camerad", "system/camerad", ["./camerad"], True)
    self.encoder = NativeProcess("encoderd", "system/loggerd", ["./encoderd", "--stream"], True)
    logger.info("Native processes for camera and encoder initialized.")
    stop_states = ['failed', 'closed']
    connecting_states = ['connecting', 'new']
    while exit_event is None or not exit_event.is_set():
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
                  self.params.put_bool("LiveStreamRunning", True)
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
