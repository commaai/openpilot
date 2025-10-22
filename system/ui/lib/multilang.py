import os
import json
import gettext
from openpilot.common.params import Params
from openpilot.common.basedir import BASEDIR

SYSTEM_UI_DIR = os.path.join(BASEDIR, "system", "ui")
UI_DIR = os.path.join(BASEDIR, "selfdrive", "ui")
TRANSLATIONS_DIR = os.path.join(UI_DIR, "translations")
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, "languages.json")

SUPPORTED_LANGUAGES = [
  "en",
  "de",
  "fr",
  "pt-BR",
  "es",
  "tr",
]


class Multilang:
  def __init__(self):
    self._params = Params()
    self.languages = {}
    self.codes = {}
    self._translation: gettext.NullTranslations | gettext.GNUTranslations = gettext.NullTranslations()
    self._load_languages()

  @property
  def language(self) -> str:
    return self._params.get("LanguageSetting") or "en"

  def setup(self):
    language = self.language
    try:
      with open(os.path.join(TRANSLATIONS_DIR, f'app_{language}.mo'), 'rb') as fh:
        translation = gettext.GNUTranslations(fh)
      translation.install()
      self._translation = translation
      print(f"Loaded translations for language: {language}")
    except FileNotFoundError:
      print(f"No translation file found for language: {language}, using default.")
      gettext.install('app')
      self._translation = gettext.NullTranslations()
    return None

  def change_language(self, language_code: str) -> None:
    # Reinstall gettext with the selected language
    self._params.put("LanguageSetting", language_code)
    self.setup()

  def tr(self, text: str) -> str:
    return self._translation.gettext(text)

  def trn(self, singular: str, plural: str, n: int) -> str:
    return self._translation.ngettext(singular, plural, n)

  def _load_languages(self):
    with open(LANGUAGES_FILE, encoding='utf-8') as f:
      self.languages = {k: v for k, v in json.load(f).items() if v in SUPPORTED_LANGUAGES}
    self.codes = {v: k for k, v in self.languages.items() if v in SUPPORTED_LANGUAGES}


multilang = Multilang()
multilang.setup()

tr, trn = multilang.tr, multilang.trn


# no-op marker for static strings translated later
def tr_noop(s: str) -> str:
  return s
