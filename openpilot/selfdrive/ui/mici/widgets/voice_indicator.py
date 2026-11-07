import math
import os
import time

import raylib_nb as rl

from openpilot.cereal import log
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached

State = log.VoiceAssistant.State

# speaking green is softer than the engaged path green so the two do not read the same
COLORS = {
  State.listening: rl.Color(40, 140, 255, 255),
  State.thinking: rl.Color(40, 140, 255, 255),
  State.speaking: rl.Color(40, 210, 110, 255),
}
DEMO = {'listening': State.listening, 'thinking': State.thinking, 'speaking': State.speaking}.get(os.getenv('VOICE_UI_DEMO', ''))

DEMO_TEXT = os.getenv('VOICE_UI_TEXT', '')
DEMO_RTT = os.getenv('VOICE_UI_RTT')

GLOW_PX = 18
CAPTION_FONT = 26
ROUNDNESS = 0.12


def _caption() -> list[tuple[str, str]]:
  # chat log, what whisper heard then the reply as it is spoken
  if DEMO_TEXT:
    return [('User', DEMO_TEXT), ('Clanker', os.getenv('VOICE_UI_REPLY', ''))]
  va = ui_state.sm["voiceAssistant"]
  return [('User', va.transcript), ('Clanker', va.reply)]


def _wrap(font: rl.Font, text: str, width: float) -> list[str]:
  lines, cur = [], ''
  for w in text.split():
    nxt = f'{cur} {w}'.strip()
    if cur and measure_text_cached(font, nxt, CAPTION_FONT).x > width:
      lines.append(cur)
      cur = w
    else:
      cur = nxt
  return lines + [cur]


def _alert_active() -> bool:
  return ui_state.sm["selfdriveState"].alertSize != log.SelfdriveState.AlertSize.none


def _draw_caption(rect: rl.Rectangle, rows: list[tuple[str, str]], max_lines: int = 4) -> None:
  font = gui_app.font(FontWeight.MEDIUM)
  lines: list[tuple[str, str]] = []
  for who, text in rows:
    if text.strip():
      wrapped = _wrap(font, f'{who}: {text.strip()}', rect.width - 120)
      lines += [(who if i == 0 else '', ln) for i, ln in enumerate(wrapped)]
  if not lines:
    return
  # keep the newest lines so the reply being spoken stays visible
  lines = lines[-max_lines:]
  line_h = CAPTION_FONT + 4
  w = max(measure_text_cached(font, ln, CAPTION_FONT).x for _, ln in lines) + 28
  h = line_h * len(lines) + 14
  box = rl.Rectangle(rect.x + (rect.width - w) / 2, rect.y + rect.height - 46 - h, w, h)
  rl.draw_rectangle_rounded(box, 0.35, 12, rl.Color(0, 0, 0, 190))
  for k, (who, ln) in enumerate(lines):
    pos = rl.Vector2(box.x + 14, box.y + 7 + k * line_h)
    if who:
      label = f'{who}:'
      color = COLORS[State.thinking] if who == 'User' else COLORS[State.speaking]
      rl.draw_text_ex(font, label, pos, CAPTION_FONT, 0, color)
      pos.x += measure_text_cached(font, label, CAPTION_FONT).x
      ln = ln[len(label):]
    rl.draw_text_ex(font, ln, pos, CAPTION_FONT, 0, rl.WHITE)


def _state():
  if DEMO is not None:
    return DEMO
  sm = ui_state.sm
  # received enums don't hash like the schema constants, compare raw values
  state = sm["voiceAssistant"].state.raw
  return state if sm.alive["voiceAssistant"] and state in COLORS else None


def _draw_rtt(rect: rl.Rectangle) -> None:
  if DEMO_RTT is not None:
    rtt = float(DEMO_RTT)
  elif ui_state.sm.alive["voiceAssistant"]:
    rtt = ui_state.sm["voiceAssistant"].serverRttMs
  else:
    return
  if _alert_active():
    return
  font = gui_app.font(FontWeight.MEDIUM)
  text = f'{rtt:.0f} ms' if rtt > 0 else '-- ms'
  color = rl.Color(255, 255, 255, 200) if 0 < rtt < 300 else rl.Color(255, 120, 90, 230)
  w = measure_text_cached(font, text, CAPTION_FONT).x
  rl.draw_text_ex(font, text, rl.Vector2(rect.x + (rect.width - w) / 2, rect.y + 8), CAPTION_FONT, 0, color)


def draw_voice_indicator(rect: rl.Rectangle) -> None:
  _draw_rtt(rect)
  state = _state()
  if state is None:
    return
  c = COLORS[state]
  t = time.monotonic()
  # only thinking breathes, listening and speaking stay steady
  a = 0.6 + 0.4 * math.sin(t * 4) if state == State.thinking else 1.0

  if state != State.listening and not _alert_active():
    _draw_caption(rect, _caption())

  # stacked rounded outlines fading inwards read as a soft glow hugging the rounded screen
  for i in range(GLOW_PX):
    k = 1 - i / GLOW_PX
    r = rl.Rectangle(rect.x + i, rect.y + i, rect.width - 2 * i, rect.height - 2 * i)
    rl.draw_rectangle_rounded_lines_ex(r, ROUNDNESS, 16, 1.5, rl.Color(c.r, c.g, c.b, int(255 * a * k ** 2.2)))

