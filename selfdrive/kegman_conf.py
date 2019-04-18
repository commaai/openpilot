import json
import copy
import os
import threading
import time
import selfdrive.messaging as messaging
import zmq
from selfdrive.services import service_list
from selfdrive.swaglog import cloudlog

class kegman_conf():
  def __init__(self, read_only=False):  # start thread by default
    self.conf = self.read_config()
    context = zmq.Context()
    self.poller = zmq.Poller()
    self.kegman_Conf = messaging.sub_sock(context, service_list['kegmanConf'].port, conflate=True, poller=self.poller)
    # when you import kegman_conf and only use it to read data, you can specify read_only in your import as to not start the write_thread
    if not read_only:
      threading.Thread(target=self.zmq_thread()).start()

  def read_config(self):
    default_config = {"cameraOffset":"0.06", "lastTrMode":"1", "battChargeMin":"90", "battChargeMax":"95", "wheelTouchSeconds":"1800", "battPercOff":"25", "carVoltageMinEonShutdown":"11200", "brakeStoppingTarget":"0.25", "angle_steers_offset":"0" , "brake_distance_extra":"1" , "lastALCAMode":"1" , "brakefactor":"1.2", "lastGasMode":"0" , "lastSloMode":"1", "leadDistance":"5"}

    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        try:
          config = json.load(f)
        except:
          cloudlog.exception("reading kegman.json error")
          config = default_config
      self.last_conf = copy.deepcopy(config)
      if "battPercOff" not in config:
        config.update({"battPercOff":"25"})
      if "carVoltageMinEonShutdown" not in config:
        config.update({"carVoltageMinEonShutdown":"11200"})
      if "brakeStoppingTarget" not in config:
        config.update({"brakeStoppingTarget":"0.25"})
      if "angle_steers_offset" not in config:
        config.update({"angle_steers_offset":"0"})
      if "brake_distance_extra" not in config: # extra braking distance in m
        config.update({"brake_distance_extra":"1"})
      if "lastALCAMode" not in config:
        config.update({"lastALCAMode":"1"})
      if "brakefactor" not in config: # brake at 20% higher speeds than what I like
        config.update({"brakefactor":"1.2"})
      if "lastGasMode" not in config:
        config.update({"lastGasMode":"0"})
      if "lastSloMode" not in config:
        config.update({"lastSloMode":"1"})
      if "leadDistance" not in config: # leadDistance only works for Accord and Insight, have not tested other honda vehicles
        config.update({"leadDistance":"5.0"})

      # force update
      if config['carVoltageMinEonShutdown'] == "11800":
        config.update({"carVoltageMinEonShutdown":"11200"})
      if int(config['wheelTouchSeconds']) < 200:
        config.update({"wheelTouchSeconds":"1800"})
      if int(config['battChargeMin']) == 85:
        config.update({"battChargeMin":"90"})
      if int(config['battChargeMax']) == 90:
        config.update({"battChargeMax":"95"})
    else:
      config = default_config
    return config

  def zmq_thread(self):
    while True:
      time.sleep(5)
      kegmanConf = None
      for socket, event in self.poller.poll(0):
        if socket is self.kegman_Conf:
          kegmanConf = messaging.recv_one(socket).kegmanConf
      if kegmanConf:
        if kegmanConf.angle_steers_offset:
          self.conf["angle_steers_offset"] = float(kegmanConf.angle_steers_offset)
        if kegmanConf.battChargeMax:
          self.conf["battChargeMax"] = float(kegmanConf.battChargeMax)
        if kegmanConf.battChargeMin:
          self.conf["battChargeMin"] = float(kegmanConf.battChargeMin)
        if kegmanConf.battPercOffbattPercOff:
          self.conf["battPercOff"] = float(kegmanConf.battPercOff)
        if kegmanConf.brakeStoppingTarget:
          self.conf["brakeStoppingTarget"] = float(kegmanConf.brakeStoppingTarget)
        if kegmanConf.brake_distance_extra:
          self.conf["brake_distance_extra"] = float(kegmanConf.brake_distance_extra)
        if kegmanConf.brakefactor:
          self.conf["brakefactorbrakefactor"] = float(kegmanConf.brakefactor)
        if kegmanConf.cameraOffset:
          self.conf["cameraOffsetcameraOffset"] = float(kegmanConf.cameraOffset)
        if kegmanConf.carVoltageMinEonShutdown:
          self.conf["carVoltageMinEonShutdown"] = float(kegmanConf.carVoltageMinEonShutdown)
        if kegmanConf.lastALCAMode:
          self.conf["lastALCAMode"] = float(kegmanConf.lastALCAMode)
        if kegmanConf.lastGasMode:
          self.conf["lastGasMode"] = float(kegmanConf.lastGasMode)
        if kegmanConf.lastSloModelastSloMode:
          self.conf["lastSloMode"] = float(kegmanConf.lastSloModelastSloMode)
        if kegmanConf.lastTrMode:
          self.conf["lastTrMode"] = float(kegmanConf.lastTrMode)
        if kegmanConf.leadDistance:
          self.conf["leadDistance"] = float(kegmanConf.leadDistance)
        if kegmanConf.wheelTouchSeconds:
          self.conf["wheelTouchSeconds"] = float(kegmanConf.wheelTouchSeconds)
        self.write_config()

  def write_thread(self):
    last_conf = copy.deepcopy(self.conf)
    while True:
      time.sleep(5)  # every 5 seconds check for conf change
      if self.conf != last_conf:
        self.write_config()
        last_conf = copy.deepcopy(self.conf)  # cache the current config

  def write_config(self):  # never to be called outside kegman_conf
    try:
      start = time.time()
      with open('/data/kegman.json', 'w') as f:
        json.dump(self.conf, f, indent=2, sort_keys=True)
        os.chmod("/data/kegman.json", 0o764)
      with open("/data/kegman_times", "a") as f:
        f.write(str(time.time() - start)+"\n")
    except IOError:
      os.mkdir('/data')
      with open('/data/kegman.json', 'w') as f:
        json.dump(self.conf, f, indent=2, sort_keys=True)
        os.chmod("/data/kegman.json", 0o764)
