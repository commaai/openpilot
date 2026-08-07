import re
from pathlib import Path
from typing import Any

from openpilot.common.swaglog import cloudlog


ASOUND_ROOT = Path("/proc/asound")
CARD_RE = re.compile(r"(?:\(hw:|\bhw:)(\d+)[,)]")


def usb_card_indices(asound_root: Path = ASOUND_ROOT) -> set[int]:
  """Return ALSA card indices backed by USB devices."""
  cards: set[int] = set()
  try:
    for card_path in asound_root.glob("card[0-9]*"):
      card = int(card_path.name[4:])
      device_path = card_path / "device"
      try:
        if "usb" in str(device_path.resolve()).lower():
          cards.add(card)
          continue
      except OSError:
        pass

      # Some ALSA layouts don't expose cardN/device, but do identify the card
      # as USB in /proc/asound/cards.
    cards_text = (asound_root / "cards").read_text(errors="replace")
    for match in re.finditer(r"(?m)^\s*(\d+)\s+\[[^]]+\].*?(?=^\s*\d+\s+\[|\Z)", cards_text, re.S):
      if "usb" in match.group(0).lower():
        cards.add(int(match.group(1)))
  except (OSError, ValueError):
    pass
  return cards


def device_card_index(device: Any) -> int | None:
  name = str(device.get("name", ""))
  match = CARD_RE.search(name)
  return int(match.group(1)) if match else None


class AutomaticAudioDevice:
  """Select USB endpoints locally, preferring newly attached cards."""
  def __init__(self, direction: str):
    assert direction in ("input", "output")
    self.direction = direction
    self.selected: int | None = None
    self._known_usb_cards: set[int] | None = None
    self._preferred_card: int | None = None

  @property
  def channels_key(self) -> str:
    return f"max_{self.direction}_channels"

  def select(self, sd, force: bool = False, exclude: set[int | None] | None = None) -> tuple[int | None, bool]:
    devices = list(sd.query_devices())
    usb_cards = usb_card_indices()
    exclude = exclude or set()
    topology_changed = self._known_usb_cards is not None and usb_cards != self._known_usb_cards
    eligible = [(i, device_card_index(device)) for i, device in enumerate(devices)
                if i not in exclude and device.get(self.channels_key, 0) > 0 and device_card_index(device) in usb_cards]

    if self._known_usb_cards is None:
      self._preferred_card = max(usb_cards, default=None)
    else:
      attached = usb_cards - self._known_usb_cards
      if attached:
        self._preferred_card = max(attached)
      elif self._preferred_card not in usb_cards:
        self._preferred_card = max(usb_cards, default=None)
    self._known_usb_cards = usb_cards

    candidates = [i for i, card in eligible if card == self._preferred_card]
    if not candidates:
      candidates = [i for i, _ in eligible]
    selected = max(candidates, default=None)
    if selected is None and None in exclude:
      selected = self.selected
    changed = force or topology_changed or selected != self.selected
    if changed:
      cloudlog.info(f"automatic audio {self.direction} selected device={selected}, usb_cards={sorted(usb_cards)}")
      self.selected = selected
    return selected, changed

  def alternatives(self, sd) -> list[int | None]:
    devices = list(sd.query_devices())
    usb_cards = usb_card_indices()
    usb = [i for i, device in enumerate(devices)
           if device.get(self.channels_key, 0) > 0 and device_card_index(device) in usb_cards and i != self.selected]
    return sorted(usb, reverse=True) + [None]
