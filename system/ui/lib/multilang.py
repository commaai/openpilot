import os
import gettext
from openpilot.common.basedir import BASEDIR

UI_DIR = os.path.join(BASEDIR, "selfdrive", "ui")
TRANSLATIONS_DIR = os.path.join(UI_DIR, "translations")
LANGUAGES_FILE = os.path.join(TRANSLATIONS_DIR, "languages.json")

tr = gettext.gettext
trn = gettext.ngettext
