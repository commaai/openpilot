import os
from openpilot.selfdrive.modeld.modeld import main
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.helpers import usbgpu_present, modeld_pkl_path
from openpilot.common.file_chunker import get_manifest_path

def main():
  _present = usbgpu_present()
  _compiled = os.path.isfile(get_manifest_path(modeld_pkl_path(usbgpu=True)))
  USBGPU = _present and _compiled
  params = Params()
  params.put_bool("UsbGpuPresent", _present)
  params.put_bool("UsbGpuCompiled", _compiled)
  if not USBGPU: exit(0)
  main(usbgpu=USBGPU)
