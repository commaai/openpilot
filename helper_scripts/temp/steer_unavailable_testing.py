# cp.vl["EPS_STATUS"]['LKA_STATE'] status meanings

# 1: steer req is 0, torque is not being applied by eps. generally ok and fault-free
  # carcontroller waits 2s after last 5 state, so state is 1 for 2s after fault. can we change the duration?

# 2: unknown
# 3: unknown
# 4: unknown
# 5: engaged and applying torque, no faults

# 9: steer fault ocurred, not applying torque for duration

# 25: most often occurs for one frame. on rising edge of a steering fault
