#!/usr/bin/env python3
from collections import deque
from tools.lib.route import Route
from tools.lib.logreader import MultiLogIterator
import sys

MIN_ENABLED = 10

# Finds the speed thresholds at which OP can begin steering and then how slowly the vehicle can travel before losing steering control.
# Returns the minimum steering enable and disable speeds from a given set of routes.

# Application: vehicles with a variable CP.minSteerSpeed such as Honda Odyssey 2021-2022, Acura RDX 2022, and 2022 Mazda CX9.

# Requirements:
# 1. OP must have been enabled for the last 10 steps (to account for latency) prior to lateral control becoming available.
# 2. Lateral control is active in a sample 'n', but not in 'n-1': the speed at which the vehicle allows OP to steer.
# 3. Lateral control is not active in sample 'n', but was in  'n-1': the speed where the torque request will be ignored (CP.minSteerSpeed).
# 4. rlogs
# 5. OP must detect when steering capability has been lost and set 'latActive' False in the carControls packet.

# Usage: ./find_steering_speeds.py "dongleid|routeid" "dongleid|routeid2" "dongleid|routeid3"

if __name__ == '__main__':
  steerAvailble, steerRejected = [], []

  routes = sys.argv[1:]
  assert(len(routes) != 0)

  print(f"Checking {len(routes)} routes...")
  for route in routes:
    currentSegment, lastSegment = 0, -1
    enabled = deque(MIN_ENABLED*[0], maxlen=MIN_ENABLED)
    latActive, latActive_last = False, False
    vEgo, vEgo_last = 0., 0.

    r = Route(route)
    lr = MultiLogIterator(r.log_paths()[6:])
    print(''), print(f"route: {route}")

    for msg in lr:
      currentSegment = lr._current_log
      if currentSegment != lastSegment:
        print(f'Current segment: {currentSegment}/{len(r.log_paths())}')
        lastSegment = currentSegment
      if msg.which() == "controlsState":
        enabled.append(bool(msg.controlsState.enabled))
      if msg.which() == "carControl":
        latActive_last = latActive
        latActive = msg.carControl.latActive
      if msg.which() == "carState":
        if len(enabled) == MIN_ENABLED and all(enabled):
          vEgo_last = vEgo
          vEgo = msg.carState.vEgo
          # rising edge
          if latActive and not latActive_last:
            steerAvailble.append(vEgo)
            print(f'Transition found: steering initially available at: {vEgo} m/s')
          # falling edge
          if not latActive and latActive_last:
            steerRejected.append(vEgo_last)
            print(f'Transition found: steering became unavailable below: {vEgo_last} m/s')

  if len(steerAvailble) != 0:
    print(f'The minimum speed required to begin steering for this session: {min(steerAvailble)} m/s')
  else:
    print('No steering available transitions were found. Try different route(s).')
  if len(steerRejected) != 0:
    print(f'The absolute minimum steering speed (CP.minSteerSpeed) for this session: {min(steerRejected)} m/s')
  else:
    print('No rejected steering commands rejected transitions speeds found. Try a different route(s).')
