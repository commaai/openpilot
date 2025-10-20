import os
import gettext
from openpilot.common.params import Params
from openpilot.common.basedir import BASEDIR


UI_DIR = os.path.join(BASEDIR, "selfdrive", "ui")
TRANSLATIONS_DIR = os.path.join(UI_DIR, "translations")
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, "languages.json")


class Multilang:
  def __init__(self):
    self._params = Params()
    self._language: str = self._params.get("LanguageSetting").strip("main_")
    print(f"Multilang initialized with language: {self._language}")

    # self._translations: dict[str, dict[str, str]] = {}

    # self._load_languages()

  def setup(self):
    # global tr, trn
    try:
      with open(os.path.join(TRANSLATIONS_DIR, f'app_{self._language}.mo'), 'rb') as fh:
        translation = gettext.GNUTranslations(fh)
      translation.install()
      tr = translation.gettext
      trn = translation.ngettext
      print(f"Loaded translations for language: {self._language}")
    except FileNotFoundError:
      print(f"No translation file found for language: {self._language}, using default.")
      gettext.install('app')
      tr = gettext.gettext
      trn = gettext.ngettext

    return tr, trn

  def install_language(self, language: str):
    # install and set globals
    global tr, trn
    self._language = language
    tr, trn = self.setup()

  # def translate(self, text: str) -> str:
  #   if self._language not in self._translations:
  #     return text
  #   return self._translations[self._language].get(text, text)
  #
  # def _load_languages(self):
  #   self._language = Params().get("LanguageSetting")
  #
  #   LANGUAGE_DIR = os.path.join(BASEDIR, "selfdrive", "ui", "translations")
  #   for file in os.listdir(LANGUAGE_DIR):
  #     if file.endswith(".ts"):
  #       pass


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
