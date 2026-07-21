from openpilot.cereal import log
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.common.realtime import DT_CTRL

State = log.SelfdriveState.OpenpilotState

SOFT_DISABLE_TIME = 3  # seconds
ACTIVE_STATES = (State.enabled, State.lateralEnabled, State.softDisabling, State.overriding)
ENABLED_STATES = (State.preEnabled, *ACTIVE_STATES)

class StateMachine:
  def __init__(self):
    self.current_alert_types = [ET.PERMANENT]
    self.state = State.disabled
    self.soft_disable_timer = 0
    self.mads_blocked = False

  def update(self, events: Events, lateral_only: bool | None = None, mads_requested: bool = False):
    if not mads_requested:
      self.mads_blocked = False

    if lateral_only is None:
      lateral_only = self.state == State.lateralEnabled
    nominal_state = State.lateralEnabled if lateral_only else State.enabled

    # decrement the soft disable timer at every step, as it's reset on
    # entrance in SOFT_DISABLING state
    self.soft_disable_timer = max(0, self.soft_disable_timer - 1)

    self.current_alert_types = [ET.PERMANENT]

    # ENABLED, SOFT DISABLING, PRE ENABLING, OVERRIDING
    if self.state != State.disabled:
      # user and immediate disable always have priority in a non-disabled state
      if events.contains(ET.IMMEDIATE_DISABLE):
        self.state = State.disabled
        self.mads_blocked = mads_requested
        self.current_alert_types.append(ET.IMMEDIATE_DISABLE)

      elif events.contains(ET.USER_DISABLE):
        self.state = State.disabled
        self.mads_blocked = mads_requested
        self.current_alert_types.append(ET.USER_DISABLE)

      else:
        # ENABLED
        if self.state in (State.enabled, State.lateralEnabled):
          if events.contains(ET.SOFT_DISABLE):
            self.state = State.softDisabling
            self.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)
            self.current_alert_types.append(ET.SOFT_DISABLE)

          elif events.contains(ET.OVERRIDE_LATERAL) or events.contains(ET.OVERRIDE_LONGITUDINAL):
            self.state = State.overriding
            self.current_alert_types += [ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL]
          else:
            self.state = nominal_state

        # SOFT DISABLING
        elif self.state == State.softDisabling:
          if not events.contains(ET.SOFT_DISABLE):
            # no more soft disabling condition, so go back to ENABLED
            self.state = nominal_state

          elif self.soft_disable_timer > 0:
            self.current_alert_types.append(ET.SOFT_DISABLE)

          elif self.soft_disable_timer <= 0:
            self.state = State.disabled
            self.mads_blocked = mads_requested

        # PRE ENABLING
        elif self.state == State.preEnabled:
          if not events.contains(ET.PRE_ENABLE):
            self.state = nominal_state
          else:
            self.current_alert_types.append(ET.PRE_ENABLE)

        # OVERRIDING
        elif self.state == State.overriding:
          if events.contains(ET.SOFT_DISABLE):
            self.state = State.softDisabling
            self.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)
            self.current_alert_types.append(ET.SOFT_DISABLE)
          elif not (events.contains(ET.OVERRIDE_LATERAL) or events.contains(ET.OVERRIDE_LONGITUDINAL)):
            self.state = nominal_state
          else:
            self.current_alert_types += [ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL]

    # DISABLED
    elif self.state == State.disabled:
      # MADS is level-triggered by the main switch. This lets lateral control
      # engage once a transient no-entry/soft-disable condition clears without
      # requiring the driver to cycle the main switch again.
      blocking_event = any(events.contains(et) for et in (ET.IMMEDIATE_DISABLE, ET.USER_DISABLE, ET.SOFT_DISABLE))
      if (events.contains(ET.ENABLE) or mads_requested) and not (mads_requested and (self.mads_blocked or blocking_event)):
        if events.contains(ET.NO_ENTRY):
          self.current_alert_types.append(ET.NO_ENTRY)

        else:
          if events.contains(ET.PRE_ENABLE):
            self.state = State.preEnabled
          elif events.contains(ET.OVERRIDE_LATERAL) or events.contains(ET.OVERRIDE_LONGITUDINAL):
            self.state = State.overriding
          else:
            self.state = nominal_state
          self.current_alert_types.append(ET.ENABLE)

    # Check if openpilot is engaged and actuators are enabled
    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES
    if active:
      self.current_alert_types.append(ET.WARNING)
    return enabled, active
