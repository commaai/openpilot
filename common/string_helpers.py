def replace_right(s, old, new, occurrence):
  # replace_right('1232425', '2', ' ', 1) -> '12324 5'
  # replace_right('1232425', '2', ' ', 2) -> '123 4 5'

  split = s.rsplit(old, occurrence)
  return new.join(split)