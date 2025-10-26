#!/usr/bin/env python3
from pathlib import Path

import pyray as rl

FONT_DIR = Path(__file__).resolve().parent
FONT_SIZE = 200
GLYPH_PADDING = 2
CODEPOINTS = tuple(range(32, 127))
SKIP = {"NotoColorEmoji.ttf"}

_CODEPOINT_BUFFER = rl.ffi.new("int[]", CODEPOINTS)
_CODEPOINT_PTR = rl.ffi.cast("int *", _CODEPOINT_BUFFER)

def _glyph_metrics(glyphs, rects):
  entries = []
  min_offset_y = None
  max_extent = 0

  for idx, codepoint in enumerate(CODEPOINTS):
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


def _process_font(font_path: Path):
  print(f"Processing {font_path.name}...")
  data = font_path.read_bytes()
  file_buf = rl.ffi.new("unsigned char[]", data)
  glyphs = rl.load_font_data(rl.ffi.cast("unsigned char *", file_buf), len(data), FONT_SIZE, _CODEPOINT_PTR, len(CODEPOINTS), rl.FontType.FONT_DEFAULT)
  if glyphs == rl.ffi.NULL:
    raise RuntimeError("raylib failed to load font data")

  rects_ptr = rl.ffi.new("Rectangle **")
  # padding avoids sampling neighboring glyphs when the atlas is filtered
  image = rl.gen_image_font_atlas(glyphs, rects_ptr, len(CODEPOINTS), FONT_SIZE, GLYPH_PADDING, 0)
  assert image.width > 0 and image.height > 0

  rects = rects_ptr[0]
  atlas_name = f"{font_path.stem}.png"
  atlas_path = FONT_DIR / atlas_name
  atlas_size = (image.width, image.height)

  entries, line_height, base = _glyph_metrics(glyphs, rects)
  if not rl.export_image(image, atlas_path.as_posix()):
    raise RuntimeError("Failed to export atlas image")

  fnt_path = FONT_DIR / f"{font_path.stem}.fnt"
  _write_bmfont(fnt_path, font_path.stem, atlas_name, line_height, base, atlas_size, entries)


def main():
  fonts = sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.otf"))
  for font in fonts:
    if font.name in SKIP:
      continue
    _process_font(font)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
