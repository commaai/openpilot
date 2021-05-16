def std_feedforward(_speed):
  return _speed ** 2


def acc_feedforward(_speed):
  CUSTOM_FIT_KF = 0.00006908923778520113 * 0.88  # fit kf
  ORIG_KF = 0.00003  # current kf
  comp_mult = CUSTOM_FIT_KF / ORIG_KF

  return (0.35189607550172824 * _speed ** 2 + 7.506201251644202 * _speed + 69.226826411091) * comp_mult


# torq uni was 0.1, but that's using higher more accurate ff at low speeds (so less error correcting)
gain_mult = 0.2
speeds = [9, 16, 32]  # 20, 35, 75 mph
mults = [acc_feedforward(i) / std_feedforward(i) for i in speeds]

k_p = [round(0.2 * gain_mult * m, 3) for m in mults]
k_i = [round(0.05 * gain_mult * m, 3) for m in mults]

MPH_TO_MS = 0.447
high_speeds = [60 * MPH_TO_MS, 80 * MPH_TO_MS]  # mean mult isn't 70
ffs = [acc_feedforward(hs) / std_feedforward(hs) * 0.00003 for hs in high_speeds]
mean_ff = sum(ffs) / len(ffs)

print(f'{k_p=}\n{k_i=}')
print(f'{mean_ff=}')
