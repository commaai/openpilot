import numpy as np
import pytest

from openpilot.cereal import log, messaging
from openpilot.selfdrive.ui.soundd import (ALERT_RAMP_TIME, CRITICAL_ESCALATION_TIME,
                                         SELFDRIVE_STATE_TIMEOUT, ESCALATION_ALERTS, MaxAlert, Soundd)

AudibleAlert = log.SelfdriveState.AudibleAlert


class AlertState:
  """A selfdrive state input without sockets or an audio device."""
  def __init__(self, sound=AudibleAlert.warningImmediate):
    self.state = messaging.new_message('selfdriveState').selfdriveState
    self.state.alertStatus = 'critical'
    self.state.alertSize = 'full'
    self.state.alertSound = sound
    self.state.enabled = True
    self.updated = {'selfdriveState': True}
    self.recv_time = {'selfdriveState': 0.}

  def __getitem__(self, name):
    assert name == 'selfdriveState'
    return self.state


@pytest.mark.parametrize('sound', [AudibleAlert.warningImmediate, AudibleAlert.warningSoft])
@pytest.mark.parametrize('starting_volume', [.1, .5, 1.])
def test_critical_escalation_at_full_volume(mocker, sound, starting_volume):
  clock = mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic', return_value=0.)
  sd = Soundd()
  sm = AlertState(sound)
  sd.current_volume = starting_volume
  sd.get_audible_alert(sm)
  clock.return_value = CRITICAL_ESCALATION_TIME - .01
  sd.get_audible_alert(sm)
  assert sd.current_alert == sound
  clock.return_value = CRITICAL_ESCALATION_TIME
  sd.get_audible_alert(sm)
  assert sd.current_alert == ESCALATION_ALERTS[sound]
  assert sd.current_volume == 1.
  np.testing.assert_allclose(sd.get_sound_data(100), sd.loaded_sounds[ESCALATION_ALERTS[sound]][:100])

  # Repeated updates must keep full volume without restarting the clip.
  clock.return_value += .1
  sd.get_audible_alert(sm)
  sd.update_volume()
  assert sd.current_sound_frame == 100
  assert sd.current_volume == 1.
  clock.return_value += ALERT_RAMP_TIME
  sd.update_volume()
  assert sd.current_volume == 1.


@pytest.mark.parametrize('status,size,sound', [
  ('critical', 'full', AudibleAlert.none),
  ('userPrompt', 'full', AudibleAlert.warningSoft),
  ('normal', 'full', AudibleAlert.warningImmediate),
  ('critical', 'none', AudibleAlert.warningImmediate),
])
def test_only_visible_audible_red_alerts_escalate(mocker, status, size, sound):
  clock = mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic', return_value=0.)
  sd = Soundd()
  sm = AlertState(sound)
  sm.state.alertStatus = status
  sm.state.alertSize = size
  sd.get_audible_alert(sm)
  clock.return_value = 20.
  sd.get_audible_alert(sm)
  assert sd.current_alert == sound
  assert sd.critical_start_time is None


@pytest.mark.parametrize('interrupt', ['clear', 'silent', 'orange'])
def test_escalation_resets_after_interruption(mocker, interrupt):
  clock = mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic', return_value=0.)
  sd = Soundd()
  sm = AlertState()
  sd.get_audible_alert(sm)
  clock.return_value = CRITICAL_ESCALATION_TIME
  sd.get_audible_alert(sm)
  assert sd.current_alert == MaxAlert.driver
  clock.return_value = CRITICAL_ESCALATION_TIME + 1.
  if interrupt == 'clear':
    sm.state.alertStatus = 'normal'
    sm.state.alertSize = 'none'
    sm.state.alertSound = AudibleAlert.none
  elif interrupt == 'silent':
    sm.state.alertSound = AudibleAlert.none
  else:
    sm.state.alertStatus = 'userPrompt'
    sm.state.alertSound = AudibleAlert.warningSoft
  sd.get_audible_alert(sm)
  assert sd.critical_start_time is None
  assert sd.current_alert != MaxAlert.driver
  sm = AlertState()
  restart_time = CRITICAL_ESCALATION_TIME + 2.
  clock.return_value = restart_time
  sd.get_audible_alert(sm)
  clock.return_value = restart_time + CRITICAL_ESCALATION_TIME - .01
  sd.get_audible_alert(sm)
  assert sd.current_alert == AudibleAlert.warningImmediate
  clock.return_value = restart_time + CRITICAL_ESCALATION_TIME
  sd.get_audible_alert(sm)
  assert sd.current_alert == MaxAlert.driver


def test_continuous_red_alert_changes_keep_timer(mocker):
  clock = mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic', return_value=0.)
  sd = Soundd()
  sm = AlertState()
  sd.get_audible_alert(sm)
  clock.return_value = 3.
  sm.state.alertSound = AudibleAlert.warningSoft
  sm.state.alertType = 'fcw/permanent'
  sd.get_audible_alert(sm)
  clock.return_value = CRITICAL_ESCALATION_TIME
  sd.get_audible_alert(sm)
  assert sd.current_alert == MaxAlert.critical


def test_timeout_escalation_and_recovery(mocker):
  clock = mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic', return_value=0.)
  sd = Soundd()
  sm = AlertState(AudibleAlert.none)
  sm.state.alertStatus = 'normal'
  sm.updated['selfdriveState'] = False
  clock.return_value = SELFDRIVE_STATE_TIMEOUT + .01
  sd.get_audible_alert(sm)
  assert sd.current_alert == AudibleAlert.warningImmediate
  clock.return_value += CRITICAL_ESCALATION_TIME
  sd.get_audible_alert(sm)
  assert sd.current_alert == MaxAlert.driver
  # The existing timeout window still ends, even without a new message.
  clock.return_value = SELFDRIVE_STATE_TIMEOUT + 10
  sd.get_audible_alert(sm)
  assert sd.current_alert == AudibleAlert.none
  assert sd.critical_start_time is None
  sm.updated['selfdriveState'] = True
  sm.state.alertStatus = 'critical'
  sm.state.alertSound = AudibleAlert.warningImmediate
  sd.get_audible_alert(sm)
  assert sd.current_alert == AudibleAlert.warningImmediate


def test_original_immediate_ramp(mocker):
  clock = mocker.patch('openpilot.selfdrive.ui.soundd.time.monotonic', return_value=0.)
  sd = Soundd()
  sd.current_volume = .2
  sd.get_audible_alert(AlertState())
  clock.return_value = ALERT_RAMP_TIME / 2
  sd.update_volume()
  assert sd.current_volume == pytest.approx(.6)
  clock.return_value = ALERT_RAMP_TIME
  sd.update_volume()
  assert sd.current_volume == 1.
