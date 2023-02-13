#!/usr/bin/env python3
# type: ignore

import os
import argparse
import signal
import sys

import cereal.messaging as messaging
from cereal import log
from selfdrive.monitoring.hands_on_wheel_monitor import HandsOnWheelStatus
from selfdrive.controls.lib.events import Events

HandsOnWheelState = log.DriverMonitoringState.HandsOnWheelState


def sigint_handler(signal, frame):
  print("handler!")
  exit(0)


signal.signal(signal.SIGINT, sigint_handler)


def status_monitor():
  # use driverState socker to drive timing.
  driverState = messaging.sub_sock('driverState', addr=args.addr, conflate=True)
  sm = messaging.SubMaster(['carState', 'dMonitoringState'], addr=args.addr)
  steering_status = HandsOnWheelStatus()
  v_cruise_last = 0

  while messaging.recv_one(driverState):
    try:
      sm.update()

      v_cruise = sm['carState'].cruiseState.speed
      steering_wheel_engaged = len(sm['carState'].buttonEvents) > 0 or \
          v_cruise != v_cruise_last or sm['carState'].steeringPressed
      v_cruise_last = v_cruise

      # Get status from our own instance of SteeringStatus
      steering_status.update(Events(), steering_wheel_engaged, sm['carState'].cruiseState.enabled, sm['carState'].vEgo)
      steering_state = steering_status.hands_on_wheel_state
      state_name = "Unknown                   "
      if steering_state == HandsOnWheelState.none:
        state_name = "Not Active                "
      elif steering_state == HandsOnWheelState.ok:
        state_name = "Hands On Wheel            "
      elif steering_state == HandsOnWheelState.minor:
        state_name = "Hands Off Wheel - Minor   "
      elif steering_state == HandsOnWheelState.warning:
        state_name = "Hands Off Wheel - Warning "
      elif steering_state == HandsOnWheelState.critical:
        state_name = "Hands Off Wheel - Critical"
      elif steering_state == HandsOnWheelState.terminal:
        state_name = "Hands Off Wheel - Terminal"

      # Get events from `dMonitoringState`
      events = sm['dMonitoringState'].events
      event_name = events[0].name if len(events) else "None"
      event_name = "{:<30}".format(event_name[:30])

      # Print output
      sys.stdout.write(f'\rSteering State: {state_name} | event: {event_name}')
      sys.stdout.flush()

    except Exception as e:
      print(e)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Sniff a communication socket')
  parser.add_argument('--addr', default='127.0.0.1')
  args = parser.parse_args()

  if args.addr != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.context = messaging.Context()

  status_monitor()
