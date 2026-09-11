import unittest
from unittest.mock import patch

import pyray as rl

from openpilot.selfdrive.ui.mici.layouts.settings.device import PairBigButton
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.label import UnifiedLabel


class TestPairBigButton(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.object(gui_app, "font", return_value=rl.Font()))
    self.enterContext(patch.object(gui_app, "texture", return_value=rl.Texture()))
    self.enterContext(patch.object(UnifiedLabel, "get_content_height", return_value=32))

  def test_pair_button_subtitles(self):
    button = PairBigButton()
    with patch.object(ui_state, "prime_state") as state:
      for paired, prime, full, trial, expected in (
        (False, False, False, False, "connect.comma.ai"),
        (True, False, False, True, "claim prime trial"),
        (True, False, False, False, "upgrade to prime"),
        (True, True, False, False, "lite"),
        (True, True, True, False, "prime"),
      ):
        with self.subTest(expected=expected):
          state.is_paired.return_value = paired
          state.is_prime.return_value = prime
          state.is_full_prime.return_value = full
          state.can_claim_prime_trial.return_value = trial
          state.get_pairing_provider.return_value = "google"
          button._update_state()
          self.assertEqual(button.get_value(), expected)
          self.assertEqual(button.get_text(), "paired" if paired else "pair to connect")
