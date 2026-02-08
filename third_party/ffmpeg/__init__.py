import os
import platform

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

_arch = platform.machine()
if os.path.isfile('/TICI'):
  _arch = 'larch64'
elif platform.system() == 'Darwin':
  _arch = 'Darwin'

FFMPEG_BIN_DIR = os.path.join(BASEDIR, 'third_party', 'ffmpeg', _arch, 'bin')
FFMPEG_PATH = os.path.join(FFMPEG_BIN_DIR, 'ffmpeg')
FFPROBE_PATH = os.path.join(FFMPEG_BIN_DIR, 'ffprobe')
