import math
import os
import time
import unicodedata

import raylib_nb as rl

from openpilot.cereal import log
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached

State = log.VoiceAssistant.State

# none of these are openpilot's engaged green or alert amber/red
COLORS = {
  State.listening: rl.Color(43, 200, 255, 255),
  State.thinking: rl.Color(139, 124, 255, 255),
  State.speaking: rl.Color(232, 244, 255, 255),
}
DEMO = {'listening': State.listening, 'thinking': State.thinking, 'speaking': State.speaking}.get(os.getenv('VOICE_UI_DEMO', ''))

DEMO_TEXT = os.getenv('VOICE_UI_TEXT', '')
DEMO_RTT = os.getenv('VOICE_UI_RTT')

PILL_W, PILL_H = 112, 40
BAR_W, BAR_GAP, BAR_MIN, BAR_MAX = 6, 8, 6, 28
BAR_GAIN = (0.6, 0.8, 1.0, 0.8, 0.6)
USER_FONT, REPLY_FONT, TOP_FONT = 22, 24, 36
# the ping line on top, the caption rows and the pill each get their own band so nothing overlaps
TOP_H = 46
CAPTION_LEFT, CAPTION_RIGHT = 150, 50
MAX_ROWS = 4
ATTACK_S, RELEASE_S, FADE_S = 0.06, 0.2, 0.15
SWEEP_S = 1.2
# the reply stays readable a moment after the voice ends
LINGER_S, LINGER_PER_WORD_S, LINGER_MAX_S = 1.0, 0.1, 3.0


class _Anim:
  def __init__(self):
    self.t = time.monotonic()
    self.level = 0.
    self.alpha = 0.
    self.color = COLORS[State.listening]
    self.rows: tuple[str, str] = ('', '')
    self.idle_since = 0.


_anim = _Anim()


def _source() -> tuple[int | None, float, str, str]:
  if DEMO is not None:
    t = time.monotonic()
    lvl = 0.5 + 0.4 * math.sin(t * 9) * math.sin(t * 2.3) if DEMO != State.thinking else 0.
    return DEMO, lvl, DEMO_TEXT, os.getenv('VOICE_UI_REPLY', '')
  sm = ui_state.sm
  if not sm.alive["voiceAssistant"]:
    return None, 0., '', ''
  va = sm["voiceAssistant"]
  # received enums don't hash like the schema constants, compare raw values
  state = va.state.raw
  return (state if state in COLORS else None), va.level, va.transcript, va.reply


def _alert_active() -> bool:
  return ui_state.sm["selfdriveState"].alertSize != log.SelfdriveState.AlertSize.none


def _fade(c: rl.Color, a: float) -> rl.Color:
  return rl.Color(c.r, c.g, c.b, int(c.a * max(0., min(1., a))))


def _ascii(text: str) -> str:
  # the ui font only has ascii, llm replies use curly quotes and dashes
  text = text.translate(str.maketrans({'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2013': '-', '\u2014': '-', '\u2026': '...'}))
  return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()


def _wrap(font: rl.Font, text: str, size: int, width: float) -> list[str]:
  lines, cur = [], ''
  for w in text.split():
    nxt = f'{cur} {w}'.strip()
    if cur and measure_text_cached(font, nxt, size).x > width:
      lines.append(cur)
      cur = w
    else:
      cur = nxt
  return lines + [cur] if cur else lines


def _tail(font: rl.Font, text: str, size: int, width: float) -> str:
  # one line, the end of what was said matters most
  if measure_text_cached(font, text, size).x <= width:
    return text
  while text and measure_text_cached(font, '...' + text, size).x > width:
    text = text[1:]
  return '...' + text.lstrip()


def _answer_ms() -> float:
  if DEMO_RTT is not None:
    return 1240.
  return ui_state.sm["voiceAssistant"].replyLatencyMs if ui_state.sm.alive["voiceAssistant"] else 0.


def _draw_caption(rect: rl.Rectangle, bottom: float, user: str, reply: str, a: float) -> None:
  # between the set speed on the left and the status dot on the right
  left, right = rect.x + CAPTION_LEFT, rect.x + rect.width - CAPTION_RIGHT
  width = right - left - 32
  user, reply = _ascii(user).strip(), _ascii(reply).strip()
  rows = _wrap(gui_app.font(FontWeight.BOLD), reply, REPLY_FONT, width) if reply else []
  # long replies keep the newest lines, the one being spoken, and push out what the driver said
  lines: list[tuple[str, int, rl.Color]] = [(ln, REPLY_FONT, rl.WHITE) for ln in rows[-MAX_ROWS:]]
  if user and len(lines) < MAX_ROWS:
    lines.insert(0, (_tail(gui_app.font(FontWeight.MEDIUM), user, USER_FONT, width), USER_FONT, rl.Color(170, 185, 205, 255)))
  # how long the answer took, from the last word to the speaker
  if reply and _answer_ms() > 0 and len(lines) < MAX_ROWS:
    lines.insert(0, (f'answered in {_answer_ms() / 1000:.1f} s', USER_FONT, rl.Color(170, 185, 205, 255)))
  if not lines:
    return
  sizes = [(gui_app.font(FontWeight.BOLD if sz == REPLY_FONT else FontWeight.MEDIUM), sz) for _, sz, _ in lines]
  w = max(measure_text_cached(f, ln, sz).x for (ln, _, _), (f, sz) in zip(lines, sizes, strict=True)) + 32
  h = sum(sz + 4 for _, sz, _ in lines) + 12
  box = rl.Rectangle(left + (right - left - w) / 2, max(rect.y + TOP_H, bottom - h), w, h)
  rl.draw_rectangle_rounded(box, 0.3, 12, _fade(rl.Color(0, 0, 0, 200), a))
  y = box.y + 7
  for (ln, sz, c), (f, _) in zip(lines, sizes, strict=True):
    tw = measure_text_cached(f, ln, sz).x
    rl.draw_text_ex(f, ln, rl.Vector2(box.x + (w - tw) / 2, y), sz, 0, _fade(c, a))
    y += sz + 4


def _draw_pill(rect: rl.Rectangle, state: int, a: float, now: float) -> rl.Rectangle:
  pill = rl.Rectangle(rect.x + (rect.width - PILL_W) / 2, rect.y + rect.height - PILL_H - 10, PILL_W, PILL_H)
  c = _anim.color
  rl.draw_rectangle_rounded(pill, 1.0, 16, _fade(rl.Color(0, 0, 0, 170), a))
  rl.draw_rectangle_rounded_lines_ex(pill, 1.0, 16, 2, _fade(rl.Color(c.r, c.g, c.b, 110), a))
  n = len(BAR_GAIN)
  x0 = pill.x + (PILL_W - n * BAR_W - (n - 1) * BAR_GAP) / 2
  for i, g in enumerate(BAR_GAIN):
    if state == State.thinking:
      # a highlight sweeping across flat bars, waiting not hearing
      d = abs(((now / SWEEP_S) % 1.0) * (n + 1) - 0.5 - i)
      h, bright = BAR_MIN + 2, 0.4 + 0.6 * max(0., 1 - d)
    else:
      wobble = 0.85 + 0.15 * math.sin(now * 11 + i * 1.7)
      h, bright = BAR_MIN + (BAR_MAX - BAR_MIN) * _anim.level * g * wobble, 1.0
    bar = rl.Rectangle(x0 + i * (BAR_W + BAR_GAP), pill.y + (PILL_H - h) / 2, BAR_W, h)
    rl.draw_rectangle_rounded(bar, 1.0, 8, _fade(c, a * bright))
  return pill


def _draw_top(rect: rl.Rectangle) -> None:
  # always there, red when slow or offline, short so it clears the speed on the left
  if DEMO_RTT is not None:
    rtt = float(DEMO_RTT)
  elif ui_state.sm.alive["voiceAssistant"]:
    rtt = ui_state.sm["voiceAssistant"].serverRttMs
  else:
    return
  # 0 means not connected, a stalled link counts up until voiced reconnects
  bad = not 0 < rtt < 300
  text = f'ping: {rtt:.0f} ms' if rtt else 'offline'
  font = gui_app.font(FontWeight.BOLD)
  w = measure_text_cached(font, text, TOP_FONT).x
  rl.draw_text_ex(font, text, rl.Vector2(rect.x + (rect.width - w) / 2, rect.y + 4), TOP_FONT, 0,
                  rl.Color(255, 90, 70, 255) if bad else rl.Color(255, 255, 255, 220))


def draw_voice_indicator(rect: rl.Rectangle) -> None:
  now = time.monotonic()
  dt, _anim.t = min(now - _anim.t, 0.1), now
  state, lvl, user, reply = _source()

  # fast attack, slow release, like a level meter
  k = 1 - math.exp(-dt / (ATTACK_S if lvl > _anim.level else RELEASE_S))
  _anim.level += (lvl - _anim.level) * k
  _anim.alpha = max(0., min(1., _anim.alpha + (dt / FADE_S) * (1 if state is not None else -1)))
  if state is not None:
    _anim.color = COLORS[state]
    _anim.idle_since = 0.
    if user or reply:
      _anim.rows = (user, reply)
  elif not _anim.idle_since:
    _anim.idle_since = now

  _draw_top(rect)
  # nothing to show once the drive is over
  if not ui_state.started and DEMO is None:
    _anim.rows = ('', '')
    return

  if _anim.alpha > 0:
    _draw_pill(rect, state if state is not None else State.listening, _anim.alpha, now)

  # a short reading time after the voice is done, then fade out
  user, reply = _anim.rows
  a = 1.
  if state is None and (user or reply):
    linger = min(LINGER_MAX_S, LINGER_S + LINGER_PER_WORD_S * len(reply.split()))
    a = min(1., (_anim.idle_since + linger - now) / 0.3)
    if a <= 0:
      _anim.rows = ('', '')
      return
  if (user or reply) and state != State.listening and not _alert_active():
    # always above where the pill sits, so the text never moves while the pill fades
    _draw_caption(rect, rect.y + rect.height - PILL_H - 18, user, reply, a)
