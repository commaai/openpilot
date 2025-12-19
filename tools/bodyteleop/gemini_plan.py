import re


def parse_plan_from_text(response_text: str):
  """
  Parse a Gemini response and extract a plan.

  Expected format:
    plan
    w,a,s,d,t
    ...

  where t is cumulative seconds from plan start (strictly increasing).

  Returns:
    list[(w,a,s,d,t)] with w/a/s/d ints (0/1) and t float
    or None if no valid plan was found.
  """
  if response_text is None:
    return None

  text = response_text.strip()
  if not text:
    return None

  lines = [ln.strip() for ln in text.split("\n")]

  # Prefer parsing after "plan" keyword, but fall back to any plan-like lines.
  plan_lines = _extract_plan_lines(lines, require_plan_keyword=True)
  if not plan_lines:
    plan_lines = _extract_plan_lines(lines, require_plan_keyword=False)

  if not plan_lines:
    return None

  plan = []
  for w, a, s, d, t in plan_lines:
    plan.append((w, a, s, d, float(t)))

  if not validate_plan(plan):
    return None

  return plan


def _extract_plan_lines(lines, require_plan_keyword: bool):
  found_plan = not require_plan_keyword
  out = []

  for line in lines:
    if not found_plan and "plan" in line.lower():
      found_plan = True
      continue

    if not found_plan:
      continue

    # Must be "x,x,x,x,x" (4 commas)
    if line.count(",") != 4:
      continue

    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
      continue

    try:
      w = int(float(parts[0]))
      a = int(float(parts[1]))
      s = int(float(parts[2]))
      d = int(float(parts[3]))
      t = float(parts[4])
    except Exception:
      continue

    if w not in (0, 1) or a not in (0, 1) or s not in (0, 1) or d not in (0, 1):
      continue
    if t < 0:
      continue

    out.append((w, a, s, d, t))

  return out


def validate_plan(plan):
  """
  Validate a plan:
  - non-empty
  - each row has at most one of w/a/s/d == 1
  - t strictly increasing
  - final t <= 5.0 seconds
  """
  if not plan:
    return False

  last_t = -1.0
  for w, a, s, d, t in plan:
    if (w + a + s + d) > 1:
      return False
    if t <= last_t:
      return False
    last_t = t

  if last_t > 5.0:
    return False

  return True


def plan_end_time_seconds(plan):
  if not plan:
    return 0.0
  return float(plan[-1][4])


def compute_next_gemini_call_time(last_call_time, min_interval_s, plan_start_time, plan):
  """
  Determine the earliest time we are allowed to call Gemini again.
  Enforces:
  - >= last_call_time + min_interval_s
  - >= plan_start_time + plan_end_time_seconds(plan)  (if plan active)
  """
  t = last_call_time + float(min_interval_s)
  if plan_start_time is not None and plan:
    t = max(t, float(plan_start_time) + plan_end_time_seconds(plan))
  return t


