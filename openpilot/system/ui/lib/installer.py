import re
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from openpilot.common.hardware import HARDWARE
from openpilot.common.utils import atomic_write

OPENPILOT_URL = "https://openpilot.comma.ai"
INSTALLER_PATH = "/tmp/installer"
INSTALLER_URL_PATH = Path("/tmp/installer_url")


class InvalidInstaller(Exception):
  pass


def installer_url(url: str) -> str:
  if re.match(r"^([^/.]+)/([^/]+)$", url):
    url = f"https://installer.comma.ai/{url}"

  parsed = urlparse(url, scheme="https")
  return (parsed if parsed.netloc else urlparse(f"https://{url}")).geturl()


def download_installer(url: str) -> Iterator[int]:
  headers = {"User-Agent": f"AGNOSSetup-{HARDWARE.get_os_version()}",
             "X-openpilot-serial": HARDWARE.get_serial(),
             "X-openpilot-device-type": HARDWARE.get_device_type()}
  req = urllib.request.Request(url, headers=headers)
  with urllib.request.urlopen(req, timeout=30) as response, atomic_write(INSTALLER_PATH, "w+b", overwrite=True) as f:
    size = int(response.headers.get("content-length", 0))
    while chunk := response.read(8192):
      f.write(chunk)
      if size:
        yield f.tell() * 100 // size

    f.seek(0)
    if f.read(4) != b"\x7fELF":
      raise InvalidInstaller
    INSTALLER_URL_PATH.write_text(url)
