#!/usr/bin/env python3
import time
from openpilot.system.manager.process_config import managed_processes
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext, MonitoringModelFrame

cnt = 0
try:
  while True:
    """
    cl_context = CLContext()
    vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True, cl_context)
    while not vipc_client.connect(False):
      time.sleep(0.1)
    print("cnnctDD")
    del vipc_client
    time.sleep(0.5)
    continue
    """
    if (cnt % 5) == 0:
      managed_processes['modeld'].stop(block=True)
      managed_processes['modeld'].start()
    if (cnt % 3) == 0:
      managed_processes['dmonitoringmodeld'].stop(block=True)
      managed_processes['dmonitoringmodeld'].start()
    cnt += 1
    time.sleep(1)
except:
  managed_processes['modeld'].stop(block=True)
  managed_processes['dmonitoringmodeld'].stop(block=True)
