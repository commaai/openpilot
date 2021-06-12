from numpy import interp


def to_mph(x):
  return [i * 2.23694 for i in x]


def to_ms(x):
  return [i * 0.44704 for i in x]


p_mod_x = [5., 30., 55., 80.]
for v_ego in p_mod_x:
  if v_ego != 80.:
    continue
  # profile to tune mods for
  x_vel_tuning = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]
  x_vel_tuning = to_mph(x_vel_tuning)
  y_dist_tuning = [1.5486, 1.556, 1.5655, 1.5773, 1.5964, 1.6246, 1.6715, 1.7057, 1.7859, 1.8542, 1.8697, 1.8833, 1.8961]

  TR_tuning = interp(v_ego, x_vel_tuning, y_dist_tuning)

  traffic_mod_pos = [0.5, 0.35, 0.1, 0.03]
  traffic_mod_neg = [1.3, 1.4, 1.8, 2.0]

  traffic_mod_pos = interp(v_ego, p_mod_x, traffic_mod_pos)
  traffic_mod_neg = interp(v_ego, p_mod_x, traffic_mod_neg)

  # base profile to compare to (relaxed)
  x_vel_base = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]
  x_vel_base = to_mph(x_vel_base)
  y_dist_base = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.516, 1.534, 1.546, 1.568, 1.579, 1.593, 1.614]
  TR_base = interp(v_ego, x_vel_base, y_dist_base)
  relaxed_mod_pos = [1.0, 1.0, 1.0, 1.0]
  relaxed_mod_neg = [1.0, 1.0, 1.0, 1.0]
  relaxed_mod_pos = interp(v_ego, p_mod_x, relaxed_mod_pos)
  relaxed_mod_neg = interp(v_ego, p_mod_x, relaxed_mod_neg)


  x_rel = [-20.0288, -15.6871, -11.1965, -7.8645, -4.9472, -3.0541, -2.2244, -1.5045, -0.7908, -0.3196, 0.0, 0.5588, 1.3682, 1.898, 2.7316, 4.4704]  # relative velocity values
  x_rel = to_mph(x_rel)
  y_rel = [0.62323, 0.49488, 0.40656, 0.32227, 0.23914, 0.12269, 0.10483, 0.08074, 0.04886, 0.0072, 0.0, -0.05648, -0.0792, -0.15675, -0.23289, -0.315]  # modification values

  TR_mod_pos = interp(-10, x_rel, y_rel)
  TR_mod_neg = interp(3.6, x_rel, y_rel)
  print('v_ego: {}'.format(v_ego))
  print('TUNING TR: {}'.format(TR_tuning))
  pos_traffic = TR_tuning + TR_mod_pos * traffic_mod_pos
  neg_traffic = TR_tuning + TR_mod_neg * traffic_mod_neg
  print('pos: {}, neg: {}'.format(pos_traffic, neg_traffic))
  print()
  print('BASE TR: {}'.format(TR_base))
  pos_relaxed = TR_base + TR_mod_pos * relaxed_mod_pos
  neg_relaxed = TR_base + TR_mod_neg * relaxed_mod_neg
  print('pos: {}, neg: {}'.format(pos_relaxed, neg_relaxed))
  print('\npos difference: {}%'.format(100*(1 - (pos_traffic / pos_relaxed))))
  print('neg difference: {}%'.format(100*(1 - (neg_traffic / neg_relaxed))))
  print('------')
