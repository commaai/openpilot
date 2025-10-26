#!/usr/bin/env python3
from pathlib import Path
import json

import pyray as rl

FONT_DIR = Path(__file__).resolve().parent
SELFDRIVE_DIR = FONT_DIR.parent.parent
TRANSLATIONS_DIR = SELFDRIVE_DIR / "ui" / "translations"
LANGUAGES_FILE = TRANSLATIONS_DIR / "languages.json"

FONT_SIZE = 200
GLYPH_PADDING = 2
EXTRA_CHARS = "–‑✓×°§•"
UNIFONT_LANGUAGES = {"ar", "th", "zh-CHT", "zh-CHS", "ko", "ja"}
SKIP = {"NotoColorEmoji.ttf"}

KEYBOARD_LAYOUTS = {
  "lowercase": [
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    ["SHIFT_OFF", "z", "x", "c", "v", "b", "n", "m", "<-"],
    ["123", "/", "-", " ", ".", "->"],
  ],
  "uppercase": [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["SHIFT_ON", "Z", "X", "C", "V", "B", "N", "M", "<-"],
    ["123", "/", "-", " ", ".", "->"],
  ],
  "numbers": [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["-", "/", ":", ";", "(", ")", "$", "&", "@", "\""],
    ["#+=", "_", ",", "?", "!", "`", "<-"],
    ["ABC", " ", ".", "->"],
  ],
  "specials": [
    ["[", "]", "{", "}", "#", "%", "^", "*", "+", "="],
    ["_", "\\", "|", "~", "<", ">", "€", "£", "¥", "•"],
    ["123", "-", ",", "?", "!", "'", "<-"],
    ["ABC", " ", ".", "->"],
  ],
}


def _keyboard_chars():
  chars = set()
  for layout in KEYBOARD_LAYOUTS.values():
    for row in layout:
      for key in row:
        chars.update(key)
  return chars


def _load_translation_map():
  if not LANGUAGES_FILE.exists():
    return {}
  with LANGUAGES_FILE.open(encoding="utf-8") as f:
    return json.load(f)


def _load_translation_chars(code: str):
  po_path = TRANSLATIONS_DIR / f"app_{code}.po"
  try:
    return set(po_path.read_text(encoding="utf-8"))
  except FileNotFoundError:
    print(f"Translation file for language '{code}' not found when loading fonts.")
    return set()


def _build_codepoint_sets():
  base_chars = {chr(cp) for cp in range(32, 127)}
  base_chars |= _keyboard_chars()
  base_chars |= set(EXTRA_CHARS)

  unifont_chars = set(base_chars)
  languages = _load_translation_map()
  for language, code in languages.items():
    unifont_chars |= set(language)
    lang_chars = _load_translation_chars(code)
    if code in UNIFONT_LANGUAGES:
      unifont_chars |= lang_chars
    else:
      base_chars |= lang_chars

  return (
    tuple(sorted({ord(ch) for ch in base_chars})),
    tuple(sorted({ord(ch) for ch in unifont_chars})),
  )

def _glyph_metrics(glyphs, rects, codepoints):
  entries = []
  min_offset_y = None
  max_extent = 0

  for idx, codepoint in enumerate(codepoints):
    glyph = glyphs[idx]
    rect = rects[idx]

    width = int(round(rect.width))
    height = int(round(rect.height))
    offset_y = int(round(glyph.offsetY))

    min_offset_y = offset_y if min_offset_y is None else min(min_offset_y, offset_y)
    max_extent = max(max_extent, offset_y + height)

    entries.append({
      "id": codepoint,
      "x": int(round(rect.x)),
      "y": int(round(rect.y)),
      "width": width,
      "height": height,
      "xoffset": int(round(glyph.offsetX)),
      "yoffset": offset_y,
      "xadvance": int(round(glyph.advanceX)),
    })

  if min_offset_y is None:
    raise RuntimeError("No glyphs were generated")

  line_height = int(round(max_extent - min_offset_y))
  base = int(round(max_extent))
  return entries, line_height, base


def _write_bmfont(target, face, atlas_name, lh, base, texture_size, entries):
  lines = [
    f"info face=\"{face}\" size=-{FONT_SIZE} bold=0 italic=0 charset=\"\" unicode=1 stretchH=100 smooth=0 aa=1 padding=0,0,0,0 spacing=0,0 outline=0",
    f"common lineHeight={lh} base={base} scaleW={texture_size[0]} scaleH={texture_size[1]} pages=1 packed=0 alphaChnl=0 redChnl=4 greenChnl=4 blueChnl=4",
    f"page id=0 file=\"{atlas_name}\"",
    f"chars count={len(entries)}",
  ]

  for entry in entries:
    lines.append(
      ("char id={id:<4} x={x:<5} y={y:<5} width={width:<5} height={height:<5} " +
       "xoffset={xoffset:<5} yoffset={yoffset:<5} xadvance={xadvance:<5} page=0  chnl=15").format(**entry),
    )

  target.write_text("\n".join(lines) + "\n")


def _process_font(font_path: Path, codepoints: tuple[int, ...]):
  print(f"Processing {font_path.name}...")
  data = font_path.read_bytes()
  file_buf = rl.ffi.new("unsigned char[]", data)
  codepoint_buffer = rl.ffi.new("int[]", codepoints)
  codepoint_ptr = rl.ffi.cast("int *", codepoint_buffer)
  glyphs = rl.load_font_data(rl.ffi.cast("unsigned char *", file_buf), len(data), FONT_SIZE, codepoint_ptr, len(codepoints), rl.FontType.FONT_DEFAULT)
  if glyphs == rl.ffi.NULL:
    raise RuntimeError("raylib failed to load font data")

  rects_ptr = rl.ffi.new("Rectangle **")
  # padding avoids sampling neighboring glyphs when the atlas is filtered
  image = rl.gen_image_font_atlas(glyphs, rects_ptr, len(codepoints), FONT_SIZE, GLYPH_PADDING, 0)
  assert image.width > 0 and image.height > 0

  rects = rects_ptr[0]
  atlas_name = f"{font_path.stem}.png"
  atlas_path = FONT_DIR / atlas_name
  atlas_size = (image.width, image.height)

  entries, line_height, base = _glyph_metrics(glyphs, rects, codepoints)
  if not rl.export_image(image, atlas_path.as_posix()):
    raise RuntimeError("Failed to export atlas image")

  fnt_path = FONT_DIR / f"{font_path.stem}.fnt"
  _write_bmfont(fnt_path, font_path.stem, atlas_name, line_height, base, atlas_size, entries)


def main():
  base_codepoints, unifont_codepoints = _build_codepoint_sets()
  fonts = sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.otf"))
  for font in fonts:
    if font.name in SKIP:
      continue
    codepoints = unifont_codepoints if font.stem.lower().startswith("unifont") else base_codepoints
    _process_font(font, codepoints)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
