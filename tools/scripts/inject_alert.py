#!/usr/bin/env python3
import argparse
import curses
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openpilot.common.utils import atomic_write
from openpilot.selfdrive.selfdrived.debug_events import DEBUG_EVENT_PATH, parse_debug_event
from openpilot.selfdrive.selfdrived.events import EVENTS, EVENT_NAME


def alert_choices() -> dict[str, list[str]]:
  return {EVENT_NAME[event]: sorted(event_types) for event, event_types in EVENTS.items()}


def print_choices(choices: dict[str, list[str]], search: str = "") -> None:
  matches = [name for name in sorted(choices, key=str.casefold) if search.casefold() in name.casefold()]
  for name in matches:
    print(f"{name}: {', '.join(choices[name])}")
  if not matches:
    print(f"No events match {search!r}")


def resolve_choice(query: str, choices: list[str], description: str) -> str:
  exact = [choice for choice in choices if choice.casefold() == query.casefold()]
  if exact:
    return exact[0]

  matches = [choice for choice in choices if query.casefold() in choice.casefold()]
  if len(matches) == 1:
    return matches[0]
  if not matches:
    raise ValueError(f"unknown {description}: {query}")
  raise ValueError(f"ambiguous {description} {query!r}: {', '.join(matches)}")


def inject(path: str, event: str) -> None:
  command = {"event": event}
  with atomic_write(path, overwrite=True) as f:
    json.dump(command, f, separators=(",", ":"))
    f.write("\n")


def clear(path: str) -> bool:
  try:
    os.unlink(path)
  except FileNotFoundError:
    return False
  return True


def random_event(choices: dict[str, list[str]], exclude: str | None = None) -> str:
  candidates = [event for event, behaviors in choices.items() if behaviors and event != exclude]
  if not candidates:
    candidates = [event for event, behaviors in choices.items() if behaviors]
  return random.choice(candidates)


def read_active_event(path: str) -> str | None:
  try:
    with open(path) as f:
      event = parse_debug_event(f.read())
  except (OSError, ValueError):
    return None
  return EVENT_NAME[event]


def flattened_choices(choices: dict[str, list[str]]) -> list[tuple[str, str]]:
  return [(event, ", ".join(choices[event]) or "no behavior") for event in sorted(choices, key=str.casefold)]


def fuzzy_score(query: str, choice: tuple[str, str]) -> int | None:
  """Score space-separated substring or subsequence matches."""
  event, behaviors = choice
  event_search = event.casefold()
  searchable = f"{event} {behaviors}".casefold()
  score = 0

  for token in query.casefold().split():
    substring_index = searchable.find(token)
    if substring_index >= 0:
      score += 1000 - substring_index - len(searchable)
      if event_search.startswith(token):
        score += 500
      continue

    positions = []
    next_index = 0
    for char in token:
      next_index = searchable.find(char, next_index)
      if next_index < 0:
        return None
      positions.append(next_index)
      next_index += 1
    score += 100 - (positions[-1] - positions[0])

  return score


def filtered_choices(query: str, choices: list[tuple[str, str]]) -> list[tuple[str, str]]:
  if not query.strip():
    return choices

  matches = [(score, choice) for choice in choices if (score := fuzzy_score(query, choice)) is not None]
  return [choice for _, choice in sorted(matches, key=lambda match: (-match[0], match[1][0].casefold(), match[1][1].casefold()))]


def addstr(screen, y: int, x: int, text: str, attr: int = 0) -> None:
  height, width = screen.getmaxyx()
  if y < 0 or y >= height or x < 0 or x >= width - 1:
    return
  try:
    screen.addnstr(y, x, text, width - x - 1, attr)
  except curses.error:
    pass


def setup_colors() -> dict[str, int]:
  colors = {
    "header": curses.A_BOLD,
    "selected": curses.A_REVERSE | curses.A_BOLD,
    "active": curses.A_BOLD,
    "search": curses.A_BOLD,
    "muted": curses.A_DIM,
    "success": curses.A_BOLD,
  }
  if not curses.has_colors():
    return colors

  curses.start_color()
  default_background = -1
  try:
    curses.use_default_colors()
  except curses.error:
    default_background = curses.COLOR_BLACK
  curses.init_pair(1, curses.COLOR_CYAN, default_background)
  curses.init_pair(2, curses.COLOR_GREEN, default_background)
  curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_CYAN)
  curses.init_pair(4, curses.COLOR_YELLOW, default_background)
  colors.update(
    {
      "header": curses.color_pair(1) | curses.A_BOLD,
      "selected": curses.color_pair(3) | curses.A_BOLD,
      "active": curses.color_pair(2) | curses.A_BOLD,
      "search": curses.color_pair(4) | curses.A_BOLD,
      "success": curses.color_pair(2) | curses.A_BOLD,
    }
  )
  return colors


def draw_picker(
  screen,
  path: str,
  all_choices: list[tuple[str, str]],
  matches: list[tuple[str, str]],
  query: str,
  selected: int,
  scroll: int,
  message: str,
  colors: dict[str, int],
) -> int:
  screen.erase()
  height, width = screen.getmaxyx()
  list_top = 5
  list_height = max(1, height - 9)

  addstr(screen, 0, 1, "selfdrived event injector", colors["header"])
  addstr(screen, 0, 30, path, colors["muted"])

  active = read_active_event(path)
  active_text = active or "none"
  addstr(screen, 1, 1, "ACTIVE  ", colors["muted"])
  addstr(screen, 1, 9, active_text, colors["active"] if active is not None else colors["muted"])

  addstr(screen, 2, 1, "SEARCH  ", colors["muted"])
  addstr(screen, 2, 9, (query or "Type to filter by event name or behavior") + ("_" if query else ""), colors["search"] if query else colors["muted"])
  count_text = f"{len(matches)} of {len(all_choices)} events"
  addstr(screen, 3, max(1, width - len(count_text) - 2), count_text, colors["muted"])
  addstr(screen, 4, 0, "-" * max(0, width - 1), colors["muted"])

  if not matches:
    addstr(screen, list_top + 1, 3, "No matching alerts. Backspace to broaden the search or Esc to reset it.", colors["muted"])
  else:
    selected = max(0, min(selected, len(matches) - 1))
    if selected < scroll:
      scroll = selected
    elif selected >= scroll + list_height:
      scroll = selected - list_height + 1
    scroll = max(0, min(scroll, max(0, len(matches) - list_height)))

    event_width = max(12, min(42, width - 28))
    for row, (event, behaviors) in enumerate(matches[scroll : scroll + list_height], list_top):
      choice_index = scroll + row - list_top
      is_active = active == event
      marker = "*" if is_active else " "
      text = f" {marker}  {event:<{event_width}}  {behaviors}"
      if choice_index == selected:
        attr = colors["selected"]
      elif is_active:
        attr = colors["active"]
      else:
        attr = 0
      addstr(screen, row, 0, text.ljust(max(0, width - 1)), attr)

  status_y = max(list_top + 1, height - 4)
  addstr(screen, status_y, 1, message, colors["success"])
  addstr(screen, height - 3, 0, "-" * max(0, width - 1), colors["muted"])
  addstr(screen, height - 2, 1, "Up/Down move   PgUp/PgDn scroll   Enter toggle   F3 random   Esc reset", colors["muted"])
  addstr(screen, height - 1, 1, "Type to search   Backspace edit   F2/Ctrl-D clear   F10/Ctrl-C quit", colors["muted"])
  screen.refresh()
  return scroll


def curses_interactive(screen, path: str, choices: dict[str, list[str]]) -> None:
  try:
    curses.curs_set(0)
  except curses.error:
    pass
  screen.keypad(True)
  screen.timeout(250)
  colors = setup_colors()
  all_choices = flattened_choices(choices)
  query = ""
  matches = all_choices
  selected = 0
  scroll = 0
  message = "Select an event and press Enter to inject it; press Enter again to clear it."

  while True:
    matches = filtered_choices(query, all_choices)
    if matches:
      selected = max(0, min(selected, len(matches) - 1))
    else:
      selected = 0
    scroll = draw_picker(screen, path, all_choices, matches, query, selected, scroll, message, colors)

    try:
      key = screen.get_wch()
    except curses.error:
      continue

    page_size = max(1, screen.getmaxyx()[0] - 9)
    if key == curses.KEY_UP:
      selected = max(0, selected - 1)
    elif key == curses.KEY_DOWN:
      selected = min(max(0, len(matches) - 1), selected + 1)
    elif key == curses.KEY_PPAGE:
      selected = max(0, selected - page_size)
    elif key == curses.KEY_NPAGE:
      selected = min(max(0, len(matches) - 1), selected + page_size)
    elif key == curses.KEY_HOME:
      selected = 0
    elif key == curses.KEY_END:
      selected = max(0, len(matches) - 1)
    elif key in (curses.KEY_ENTER, "\n", "\r"):
      if matches:
        event, _ = matches[selected]
        if read_active_event(path) == event:
          clear(path)
          message = f"Cleared {event}"
        else:
          inject(path, event)
          message = f"Injected {event}"
    elif key in (curses.KEY_F2, curses.KEY_DC, "\x04"):
      message = "Injected event cleared" if clear(path) else "No injected event was set"
    elif key == curses.KEY_F3:
      event = random_event(choices, read_active_event(path))
      inject(path, event)
      message = f"Randomly injected {event}"
    elif key in (curses.KEY_F10, "\x03"):
      return
    elif key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
      query = query[:-1]
      selected = 0
      scroll = 0
    elif key in ("\x1b", "\x15"):
      query = ""
      selected = 0
      scroll = 0
    elif isinstance(key, str) and key.isprintable():
      query += key
      selected = 0
      scroll = 0


def choose_text_interactively(choices: dict[str, list[str]]) -> str | None:
  while True:
    query = input("Event name or search (random/list/clear/quit): ").strip()
    if query.casefold() == "quit":
      return None
    if query.casefold() == "clear":
      return ""
    if query.casefold() == "random":
      return random_event(choices)
    if query.casefold() == "list":
      print_choices(choices)
      continue
    if not query:
      continue

    names = sorted(choices, key=str.casefold)
    exact = [name for name in names if query.casefold() == name.casefold()]
    matches = [name for name in names if query.casefold() in name.casefold()]
    if exact:
      event = exact[0]
    elif len(matches) == 1:
      event = matches[0]
    elif not matches:
      print(f"No events match {query!r}")
      continue
    else:
      for i, name in enumerate(matches, 1):
        print(f"  {i:>2}. {name} [{', '.join(choices[name])}]")
      selection = input("Choose an event number (or Enter to search again): ").strip()
      if not selection:
        continue
      try:
        index = int(selection)
        if not 1 <= index <= len(matches):
          raise ValueError
        event = matches[index - 1]
      except (ValueError, IndexError):
        print("Invalid selection")
        continue

    return event


def text_interactive(path: str, choices: dict[str, list[str]]) -> None:
  print(f"selfdrived event injector ({path})")
  print("Injected events follow the real selfdrived state-machine and alert behavior.")
  print("The current event remains injected after this script exits; run with --clear to remove it.")

  while True:
    selection = choose_text_interactively(choices)
    if selection is None:
      return
    if not selection:
      print("Cleared" if clear(path) else "No injected event was set")
      continue
    inject(path, selection)
    print(f"Injected {selection}. Choose another event, type 'clear', or type 'quit'.")


def interactive(path: str, choices: dict[str, list[str]]) -> None:
  if not sys.stdin.isatty() or not sys.stdout.isatty():
    text_interactive(path, choices)
    return

  try:
    curses.wrapper(curses_interactive, path, choices)
  except curses.error as e:
    print(f"Full-screen picker unavailable ({e}); falling back to the text prompt.", file=sys.stderr)
    text_interactive(path, choices)


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Inject a real event into a running selfdrived",
    epilog="Run interactively on a device, or from a laptop with: ssh -t comma@DEVICE 'cd /data/openpilot && ./tools/scripts/inject_alert.py'",
  )
  parser.add_argument("event", nargs="?", help="event name (unique substrings are accepted)")
  parser.add_argument("--path", default=DEBUG_EVENT_PATH, help=argparse.SUPPRESS)
  parser.add_argument("--list", action="store_true", help="list injectable events and their behaviors")
  parser.add_argument("--search", default="", help="filter --list output")
  parser.add_argument("--clear", action="store_true", help="clear the injected event")
  parser.add_argument("--random", action="store_true", help="inject a random event with defined behavior")
  args = parser.parse_args()

  choices = alert_choices()
  if args.list:
    print_choices(choices, args.search)
    return 0
  if args.clear:
    print("Cleared" if clear(args.path) else "No injected event was set")
    return 0
  if args.random:
    if args.event is not None:
      parser.error("event and --random cannot be used together")
    event = random_event(choices, read_active_event(args.path))
    inject(args.path, event)
    print(f"Randomly injected {event}; run with --clear to remove it")
    return 0
  if args.event is None:
    try:
      interactive(args.path, choices)
    except (EOFError, KeyboardInterrupt):
      print("\nExiting; the injected alert was left unchanged.", file=sys.stderr)
    return 0

  try:
    event = resolve_choice(args.event, sorted(choices, key=str.casefold), "event")
  except ValueError as e:
    parser.error(str(e))

  inject(args.path, event)
  print(f"Injected {event}; run with --clear to remove it")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
