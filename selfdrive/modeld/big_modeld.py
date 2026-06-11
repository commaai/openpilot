import cereal.messaging as messaging
from openpilot.selfdrive.modeld.modeld import main as modeld_main
from openpilot.common.params import Params

def main():
  params = Params()
  state = messaging.recv_one_retry(messaging.sub_sock('usbgpuState', conflate=True)).usbgpuState
  if not (state.usbgpuPresent and params.get_bool("UsbgpuCompiled")):
    params.put_bool("UsbgpuFailed", True)
    exit(0)
  modeld_main(usbgpu=True)
