import sys
import unittest
from unittest import mock

from openpilot.tools.sim.bridge.metadrive.metadrive_world import get_metadrive_process_context


class TestMetaDriveWorld(unittest.TestCase):
  def test_uses_spawn_context_on_macos(self):
    # Panda3D must initialize Cocoa in a fresh interpreter, not a forked child.
    with mock.patch.object(sys, "platform", "darwin"):
      self.assertEqual(get_metadrive_process_context().get_start_method(), "spawn")
