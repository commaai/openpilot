import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openpilot.system.webrtc.webrtcd import CerealIncomingMessageProxy, DynamicPubMaster, StreamSession


class TestIncomingStartup(unittest.IsolatedAsyncioTestCase):
  async def test_joystick_arriving_during_connection_is_published(self):
    session = StreamSession.__new__(StreamSession)
    session.identifier = 'startup-test'
    session.params = Mock()
    session.logger = Mock()
    session.shared_pub_master = DynamicPubMaster([])
    session.incoming_bridge_services = ['testJoystick']
    session.incoming_bridge = CerealIncomingMessageProxy(session.shared_pub_master)
    session.outgoing_bridge = None
    session.bitrate_controller = None
    session.is_body = True
    session.run_body_session = AsyncMock()
    session.post_run_cleanup = AsyncMock()
    session.stream = Mock()

    async def connect():
      # A data channel can deliver input before wait_for_connection resumes.
      handler = session.stream.set_message_handler.call_args.args[0]
      handler(json.dumps({'type': 'testJoystick', 'data': {'axes': [0, 0], 'buttons': [False]}}).encode())

    session.stream.wait_for_connection = connect
    socket = Mock()
    with patch('openpilot.system.webrtc.webrtcd.messaging.pub_sock', return_value=socket):
      await session.run()
    session.logger.exception.assert_not_called()
    socket.send.assert_called_once()
