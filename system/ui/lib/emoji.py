import io
import re

from PIL import Image, ImageDraw, ImageFont
import pyray as rl

_cache: dict[str, rl.Texture] = {}

EMOJI_REGEX = re.compile(
"""[\U0001F600-\U0001F64F
\U0001F300-\U0001F5FF
\U0001F680-\U0001F6FF
\U0001F1E0-\U0001F1FF
\U00002700-\U000027BF
\U0001F900-\U0001F9FF
\U00002600-\U000026FF
\U00002300-\U000023FF
\U00002B00-\U00002BFF
\U0001FA70-\U0001FAFF
\U0001F700-\U0001F77F
\u2640-\u2642
\u2600-\u2B55
\u200d
\u23cf
\u23e9
\u231a
\ufe0f
\u3030
]+""",
  flags=re.UNICODE
)

def find_emoji(text):
  return [(m.start(), m.end(), m.group()) for m in EMOJI_REGEX.finditer(text)]

def emoji_tex(emoji):
  if emoji not in _cache:
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("NotoColorEmoji", 109)
    draw.text((0, 0), emoji, font=font, embedded_color=True)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    l = buffer.tell()
    buffer.seek(0)
    _cache[emoji] = rl.load_texture_from_image(rl.load_image_from_memory(".png", buffer.getvalue(), l))
  return _cache[emoji]
