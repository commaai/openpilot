from importlib.resources import files
import os
import json
import gettext
from typing import Any
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
  "ar",
  "th",
  "zh-CHT",
  "zh-CHS",
  "ko",
  "ja",
]


class Translatable(str):
  """A lazy translation wrapper that behaves like a str and a callable.

  - As a str: it renders to the current translation at creation time (for
    immediate uses that require a concrete Python str).
  - As a callable: calling it re-fetches the current translation from the
    active gettext catalog, optionally applying format() arguments.
  - format(): returns a concrete str using the latest translation.
  """

  def __new__(cls, msgid: str):
    # Create a str value using the current translation so immediate consumers
    # expecting a concrete str keep working.
    value = multilang._translation.gettext(msgid)
    obj = super().__new__(cls, value)
    obj._msgid = msgid  # type: ignore[attr-defined]
    return obj

  def __call__(self, *args: Any, **kwargs: Any) -> str:
    translated = multilang._translation.gettext(self._msgid)  # type: ignore[attr-defined]
    if args or kwargs:
      return translated.format(*args, **kwargs)
    return translated

  def format(self, *args: Any, **kwargs: Any) -> str:  # type: ignore[override]
    translated = multilang._translation.gettext(self._msgid)  # type: ignore[attr-defined]
    return translated.format(*args, **kwargs)


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
      cloudlog.warning(f"Loaded translations for language: {self._language}")
    except FileNotFoundError:
      cloudlog.error(f"No translation file found for language: {self._language}, using default.")
      gettext.install('app')
      self._translation = gettext.NullTranslations()

  def change_language(self, language_code: str) -> None:
    # Reinstall gettext with the selected language
    self._params.put("LanguageSetting", language_code)
    self._language = language_code
    self.setup()

  def tr(self, text: str) -> Translatable:
    return Translatable(text)

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

# Public API
tr, trn = multilang.tr, multilang.trn


# Backwards-compat: no-op marker for static strings translated later
def tr_noop(s: str) -> Translatable:
  # Return a Translatable so call sites can be simplified to just `tr(...)`.
  return Translatable(s)
