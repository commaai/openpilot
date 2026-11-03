#!/usr/bin/env python3
"""Contact sheet of viewer thumbnails. usage: sheet.py OUT.jpg KEY:FRAME:LABEL [...]  (KEY = thumbs dir name)"""
import sys
from PIL import Image, ImageDraw

from openpilot.tools.weight_lab.lab_paths import DATA

T = f'{DATA}/viewer/thumbs'
items = [a.split(':', 2) for a in sys.argv[2:]]
w, h, cols = 336, 189, 3
rows = (len(items) + cols - 1) // cols
sheet = Image.new('RGB', (w * cols, (h + 18) * rows), 'black')
dr = ImageDraw.Draw(sheet)
for i, (key, f, label) in enumerate(items):
  f = int(round(int(f) / 10) * 10)
  img = Image.open(f'{T}/{key}/{f}.jpg')
  # brighten night frames so lights and signs show
  img = img.point(lambda p: min(255, int(p * 2.2)))
  x, y = (i % cols) * w, (i // cols) * (h + 18)
  sheet.paste(img, (x, y + 18))
  dr.text((x + 4, y + 3), label, fill='white')
sheet.save(sys.argv[1], quality=85)
print(sys.argv[1])
