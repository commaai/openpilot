from common.params import Params
from cereal import log, messaging
  
def get_cached_torque_params():
  params = Params()
  torque_cache = params.get("LiveTorqueParameters")
  cache_ltp = log.Event.from_bytes(torque_cache).liveTorqueParameters
  print("Cached LiveTorqueParameters: ", cache_ltp)
    
def get_live_torque_params():
  retry = 0
  while retry<10:
    socket = messaging.sub_sock("liveParameters", timeout=1000)
    dat = messaging.recv_one(socket)
    if dat is None:
      retry+=1
      print("No liveParameters received, is the ignition on? Retrying... (%d/10)" % retry)
      continue
    ltp = dat.liveParameters.liveTorqueParameters
    print("Live Torque Parameters: ", ltp)

if __name__ == "__main__":
  get_cached_torque_params()
  get_live_torque_params()


