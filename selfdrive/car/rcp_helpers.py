
def create_radar_signals(*signals):
  # accepts multiple namedtuples in the form ([('name', value)],[msg])
  name_value = []
  msgs = []
  repetitions = []
  for s in signals:
    name_value += [nv for nv in s.name_value]
    name_value_n = len(s.name_value)
    msgs_n = [len(s.msg)]
    repetitions += msgs_n * name_value_n
    msgs += s.msg * name_value_n

  name_value = sum([[nv] * r for nv, r in zip(name_value, repetitions)], [])
  names = [n for n, v in name_value]
  vals = [v for n, v in name_value]
  return zip(names, msgs, vals)

def create_radar_checks(msgs, select, rate=20):
  if select == "all":
    return zip(msgs, [rate] * len(msgs))
  if select == "last":
    return zip([msgs[-1]], [rate])
  if select == "none":
    return []
  return []
