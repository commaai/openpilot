import os
import gettext
import pyray as rl
from typing import Union
from openpilot.common.params import Params
from openpilot.common.basedir import BASEDIR


UI_DIR = os.path.join(BASEDIR, "selfdrive", "ui")
TRANSLATIONS_DIR = os.path.join(UI_DIR, "translations")
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, "languages.json")


class Multilang:
  def __init__(self):
    self._language: str = "en"
    self._translations: dict[str, dict[str, str]] = {}

    self._load_languages()
    self._hook_draw_text()

  def translate(self, text: str) -> str:
    if self._language not in self._translations:
      return text
    return self._translations[self._language].get(text, text)

  def _load_languages(self):
    self._language = Params().get("LanguageSetting")

    LANGUAGE_DIR = os.path.join(BASEDIR, "selfdrive", "ui", "translations")
    for file in os.listdir(LANGUAGE_DIR):
      if file.endswith(".ts"):
        pass

  def _get_translated_text(self, text: str) -> str:
    # print("Translating:", text, "to", self._language, self._translations.keys())
    if self._language not in self._translations:
      return text
    return self._translations[self._language].get(text, text)

  def _hook_draw_text(self):
    # hook rl.draw_text* to get text for multilang
    # TODO: and measure text
    original_draw_text = rl.draw_text
    original_draw_text_ex = rl.draw_text_ex

    def draw_text_wrapper(text: str, posX: int, posY: int, fontSize: int, color: Union[rl.Color, list, tuple]) -> None:
      assert False
      text = self._get_translated_text(text)
      return original_draw_text(text, posX, posY, fontSize, color)

    def draw_text_ex_wrapper(font: Union[rl.Font, list, tuple], text: str, position: Union[rl.Vector2, list, tuple], fontSize: float, spacing: float,
                             tint: Union[rl.Color, list, tuple]) -> None:
      text = self._get_translated_text(text)
      return original_draw_text_ex(font, text, position, fontSize, spacing, tint)

    rl.draw_text = draw_text_wrapper
    rl.draw_text_ex = draw_text_ex_wrapper


# multilang = Multilang()
# # l = gettext.translation('app_de', localedir=TRANSLATIONS_DIR, languages=['de'])
# with open(os.path.join(TRANSLATIONS_DIR, 'app_de.mo'), 'rb') as fh:
#   l = gettext.GNUTranslations(fh)
# l.install()
# # tr = multilang.translate
# # tr = gettext.gettext
# tr = l.gettext
# trn = l.ngettext
