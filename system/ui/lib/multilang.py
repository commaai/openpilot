from importlib.resources import files
import os
import json
import gettext
from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog

try:
  from openpilot.common.params import Params
except ImportError:
  Params = None

SYSTEM_UI_DIR = os.path.join(BASEDIR, "system", "ui")
UI_DIR = files("openpilot.selfdrive.ui")
TRANSLATIONS_DIR = UI_DIR.joinpath("translations")
LANGUAGES_FILE = TRANSLATIONS_DIR.joinpath("languages.json")

UNIFONT_LANGUAGES = [
  "th",
  "zh-CHT",
  "zh-CHS",
  "ko",
  "ja",
]


class Multilang:
  def __init__(self):
    self._params = Params() if Params is not None else None
    self._language: str = "en"
    self.languages = {}
    self.codes = {}
    self._translation: gettext.NullTranslations | gettext.GNUTranslations = gettext.NullTranslations()
    self._load_languages()

  @property
  def language(self) -> str:
    return self._language

  def requires_unifont(self) -> bool:
    """Certain languages require unifont to render their glyphs."""
    return self._language in UNIFONT_LANGUAGES

  def setup(self):
    try:
      with TRANSLATIONS_DIR.joinpath(f'app_{self._language}.mo').open('rb') as fh:
        translation = gettext.GNUTranslations(fh)
      translation.install()
      self._translation = translation
      cloudlog.debug(f"Loaded translations for language: {self._language}")
    except FileNotFoundError:
      cloudlog.error(f"No translation file found for language: {self._language}, using default.")
      gettext.install('app')
      self._translation = gettext.NullTranslations()

  def change_language(self, language_code: str) -> None:
    # Reinstall gettext with the selected language
    self._params.put("LanguageSetting", language_code)
    self._language = language_code
    self.setup()

  def tr(self, text: str) -> str:
    return self._translation.gettext(text)

  def trn(self, singular: str, plural: str, n: int) -> str:
    return self._translation.ngettext(singular, plural, n)

  def _load_languages(self):
    with LANGUAGES_FILE.open(encoding='utf-8') as f:
      self.languages = json.load(f)
    self.codes = {v: k for k, v in self.languages.items()}

    if self._params is not None:
      lang = str(self._params.get("LanguageSetting")).removeprefix("main_")
      if lang in self.codes:
        self._language = lang


multilang = Multilang()
multilang.setup()

tr, trn = multilang.tr, multilang.trn


# no-op marker for static strings translated later
def tr_noop(s: str) -> str:
  return s
