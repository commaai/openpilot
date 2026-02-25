from importlib.resources import files
import json
import os
import re
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

# Plural form selectors for supported languages
PLURAL_SELECTORS = {
  'en': lambda n: 0 if n == 1 else 1,
  'de': lambda n: 0 if n == 1 else 1,
  'fr': lambda n: 0 if n <= 1 else 1,
  'pt-BR': lambda n: 0 if n <= 1 else 1,
  'es': lambda n: 0 if n == 1 else 1,
  'tr': lambda n: 0 if n == 1 else 1,
  'uk': lambda n: 0 if n % 10 == 1 and n % 100 != 11 else (1 if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else 2),
  'th': lambda n: 0,
  'zh-CHT': lambda n: 0,
  'zh-CHS': lambda n: 0,
  'ko': lambda n: 0,
  'ja': lambda n: 0,
}


def _parse_quoted(s: str) -> str:
  """Parse a PO-format quoted string."""
  s = s.strip()
  if not (s.startswith('"') and s.endswith('"')):
    raise ValueError(f"Expected quoted string: {s!r}")
  s = s[1:-1]
  result: list[str] = []
  i = 0
  while i < len(s):
    if s[i] == '\\' and i + 1 < len(s):
      c = s[i + 1]
      if c == 'n':
        result.append('\n')
      elif c == 't':
        result.append('\t')
      elif c == '"':
        result.append('"')
      elif c == '\\':
        result.append('\\')
      else:
        result.append(s[i:i + 2])
      i += 2
    else:
      result.append(s[i])
      i += 1
  return ''.join(result)


def load_translations(path) -> tuple[dict[str, str], dict[str, list[str]]]:
  """Parse a .po file and return (translations, plurals) dicts.

  translations: msgid -> msgstr
  plurals: msgid -> [msgstr[0], msgstr[1], ...]
  """
  with open(str(path), encoding='utf-8') as f:
    lines = f.readlines()

  translations: dict[str, str] = {}
  plurals: dict[str, list[str]] = {}

  # Parser state
  msgid = msgid_plural = msgstr = ""
  msgstr_plurals: dict[int, str] = {}
  field: str | None = None
  plural_idx = 0

  def finish():
    nonlocal msgid, msgid_plural, msgstr, msgstr_plurals, field
    if msgid:  # skip header (empty msgid)
      if msgid_plural:
        max_idx = max(msgstr_plurals.keys()) if msgstr_plurals else 0
        plurals[msgid] = [msgstr_plurals.get(i, '') for i in range(max_idx + 1)]
      else:
        translations[msgid] = msgstr
    msgid = msgid_plural = msgstr = ""
    msgstr_plurals = {}
    field = None

  for raw in lines:
    line = raw.strip()

    if not line:
      finish()
      continue

    if line.startswith('#'):
      continue

    if line.startswith('msgid_plural '):
      msgid_plural = _parse_quoted(line[len('msgid_plural '):])
      field = 'msgid_plural'
      continue

    if line.startswith('msgid '):
      msgid = _parse_quoted(line[len('msgid '):])
      field = 'msgid'
      continue

    m = re.match(r'msgstr\[(\d+)]\s+(.*)', line)
    if m:
      plural_idx = int(m.group(1))
      msgstr_plurals[plural_idx] = _parse_quoted(m.group(2))
      field = 'msgstr_plural'
      continue

    if line.startswith('msgstr '):
      msgstr = _parse_quoted(line[len('msgstr '):])
      field = 'msgstr'
      continue

    if line.startswith('"'):
      val = _parse_quoted(line)
      if field == 'msgid':
        msgid += val
      elif field == 'msgid_plural':
        msgid_plural += val
      elif field == 'msgstr':
        msgstr += val
      elif field == 'msgstr_plural':
        msgstr_plurals[plural_idx] += val

  finish()
  return translations, plurals


class Multilang:
  def __init__(self):
    self._params = Params() if Params is not None else None
    self._language: str = "en"
    self.languages: dict[str, str] = {}
    self.codes: dict[str, str] = {}
    self._translations: dict[str, str] = {}
    self._plurals: dict[str, list[str]] = {}
    self._plural_selector = PLURAL_SELECTORS.get('en', lambda n: 0)
    self._load_languages()

  @property
  def language(self) -> str:
    return self._language

  def requires_unifont(self) -> bool:
    """Certain languages require unifont to render their glyphs."""
    return self._language in UNIFONT_LANGUAGES

  def setup(self):
    try:
      po_path = TRANSLATIONS_DIR.joinpath(f'app_{self._language}.po')
      self._translations, self._plurals = load_translations(po_path)
      self._plural_selector = PLURAL_SELECTORS.get(self._language, lambda n: 0)
      cloudlog.debug(f"Loaded translations for language: {self._language}")
    except FileNotFoundError:
      cloudlog.error(f"No translation file found for language: {self._language}, using default.")
      self._translations = {}
      self._plurals = {}

  def change_language(self, language_code: str) -> None:
    self._params.put("LanguageSetting", language_code)
    self._language = language_code
    self.setup()

  def tr(self, text: str) -> str:
    return self._translations.get(text, text)

  def trn(self, singular: str, plural: str, n: int) -> str:
    if singular in self._plurals:
      idx = self._plural_selector(n)
      forms = self._plurals[singular]
      if idx < len(forms) and forms[idx]:
        return forms[idx]
    return singular if n == 1 else plural

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
