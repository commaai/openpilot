import gettext
from selfdrive.hardware import EON
from selfdrive.hardware.eon.hardware import getprop

locale_dir = '/data/openpilot/selfdrive/assets/locales'
supported_language = ['en-US', 'zh-TW', 'zh-CN', 'ja-JP', 'ko-KR']

def get_locale():
  return getprop("persist.sys.locale") if EON else 'en-US'

def events():
  i18n = gettext.translation('events', localedir=locale_dir, fallback=True, languages=[get_locale()])
  i18n.install()
  return i18n.gettext