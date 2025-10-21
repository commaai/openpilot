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
    print(f"Multilang initialized with language: {self.language}")

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

  # def translate(self, text: str) -> str:
  #   if self._language not in self._translations:
  #     return text
  #   return self._translations[self._language].get(text, text)
  #
  def _load_languages(self):
    with open(LANGUAGES_FILE, encoding='utf-8') as f:
      self.languages = json.load(f)
    self.codes = {v: k for k, v in self.languages.items()}
    print(f"Available languages: {self.languages}")


# # l = gettext.translation('app_de', localedir=TRANSLATIONS_DIR, languages=['de'])
# with open(os.path.join(TRANSLATIONS_DIR, 'app_de.mo'), 'rb') as fh:
#   l = gettext.GNUTranslations(fh)
# l.install()
# # tr = multilang.translate
# # tr = gettext.gettext
# tr = l.gettext
# trn = l.ngettext

# tr, trn = None, None
# multilang = Multilang()

multilang = Multilang()
tr, trn = multilang.setup()
