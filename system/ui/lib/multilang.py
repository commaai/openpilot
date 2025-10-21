import os
import json
import gettext
from openpilot.common.params import Params
from openpilot.common.basedir import BASEDIR

UI_DIR = os.path.join(BASEDIR, "selfdrive", "ui")
TRANSLATIONS_DIR = os.path.join(UI_DIR, "translations")
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, "languages.json")


class Multilang:
  def __init__(self):
    self._params = Params()
    self.languages = {}
    self.codes = {}
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
      tr = translation.gettext
      trn = translation.ngettext
      print(f"Loaded translations for language: {language}")
    except FileNotFoundError:
      print(f"No translation file found for language: {language}, using default.")
      gettext.install('app')
      tr = gettext.gettext
      trn = gettext.ngettext

    return tr, trn

  def _load_languages(self):
    with open(LANGUAGES_FILE, encoding='utf-8') as f:
      self.languages = json.load(f)
    self.codes = {v: k for k, v in self.languages.items()}


multilang = Multilang()
tr, trn = multilang.setup()
