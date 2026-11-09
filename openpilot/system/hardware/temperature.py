def format_temperature(temp_c: float, is_metric: bool) -> str:
  if is_metric:
    temp, unit = temp_c, "°C"
  else:
    temp, unit = temp_c * 9 / 5 + 32, "°F"
  return f"{temp:.1f}{unit}"
