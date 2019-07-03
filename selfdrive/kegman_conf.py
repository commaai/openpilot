import json
import os

class kegman_conf():
  def __init__(self, CP=None):
    if CP is not None:
      self.type = CP.lateralTuning.which()
    self.conf = self.read_config(CP)
    if CP is not None:
      self.init_config(CP)

  def init_config(self, CP):
    write_conf = False
    if CP.lateralTuning.which() == 'pid':
      self.type = "pid"
      if self.conf['type'] == "-1":
        self.conf["type"] = "pid"
        write_conf = True
      if self.conf['Kp'] == "-1":
        self.conf['Kp'] = str(round(CP.lateralTuning.pid.kpV[0],3))
        write_conf = True
      if self.conf['Ki'] == "-1":
        self.conf['Ki'] = str(round(CP.lateralTuning.pid.kiV[0],3))
        write_conf = True
      if self.conf['Kf'] == "-1":
        self.conf['Kf'] = str(round(CP.lateralTuning.pid.kf,5))
        write_conf = True
      if self.conf['dampTime'] == "-1":
        self.conf['dampTime'] = str(round(CP.lateralTuning.pid.dampTime,3))
        write_conf = True
      if self.conf['reactMPC'] == "-1":
        self.conf['reactMPC'] = str(round(CP.lateralTuning.pid.reactMPC,3))
        write_conf = True
      if self.conf['rateFFGain'] == "-1":
        self.conf['rateFFGain'] = str(round(CP.lateralTuning.pid.rateFFGain,3))
        write_conf = True
    else:
      self.type = "indi"
      if self.conf['type'] == "-1":
        self.conf["type"] = "indi"
        write_conf = True
      if self.conf['timeConst'] == "-1":
        self.conf['type'] = "indi"
        self.conf['timeConst'] = str(round(CP.lateralTuning.indi.timeConstant,3))
        self.conf['actEffect'] = str(round(CP.lateralTuning.indi.actuatorEffectiveness,3))
        self.conf['outerGain'] = str(round(CP.lateralTuning.indi.outerLoopGain,3))
        self.conf['innerGain'] = str(round(CP.lateralTuning.indi.innerLoopGain,3))
        write_conf = True
      if self.conf['reactMPC'] == "-1":
        self.conf['reactMPC'] = str(round(CP.lateralTuning.indi.reactMPC,3))
        write_conf = True

    if write_conf:
      self.write_config(self.config)

  def read_config(self, CP=None):
    self.element_updated = False

    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        self.config = json.load(f)
        self.write_config(self.config)

      if ("type" not in self.config or self.config['type'] == "-1") and CP != None:
          self.config.update({"type":CP.lateralTuning.which()})
          print(CP.lateralTuning.which())
          self.element_updated = True

      if self.config["type"] == "pid":
        if "Kf" not in self.config:
          self.config.update({"Kf":"-1"})
          self.element_updated = True

        if "dampTime" not in self.config:
          self.config.update({"dampTime":"-1"})
          self.element_updated = True
        if "reactMPC" not in self.config:
          self.config.update({"reactMPC":"-1"})
          self.element_updated = True
        if "type" not in self.config:
          self.config.update({"type":"pid"})
          self.element_updated = True
        if "rateFFGain" not in self.config:
          self.config.update({"rateFFGain":"-1"})
          self.element_updated = True

      else:
        if "timeConst" not in self.config:
          self.config.update({"type":"indi", "timeConst":"-1", "actEffect":"-1", "outerGain":"-1", "innerGain":"-1", "reactMPC":"-1"})
          self.element_updated = True
        if "type" not in self.config:
          self.config.update({"type":"indi"})
          self.element_updated = True
        if "reactMPC" not in self.config:
          self.config.update({"reactMPC":"-1"})
          self.element_updated = True

      if self.element_updated:
        print("updated")
        self.write_config(self.config)


    else:
      if self.type == "pid" or CP.lateralTuning.which() == "pid":
        self.config = {"type":"pid","Kp":"-1", "Ki":"-1", "Kf":"-1", "dampTime":"-1", "reactMPC":"-1", "rateFFGain":"-1"}
      else:
        self.config = {"type":"indi","timeConst":"-1", "actEffect":"-1", "outerGain":"-1", "innerGain":"-1", "reactMPC":"-1"}

      self.write_config(self.config)
    return self.config

  def write_config(self, config):
    with open('/data/kegman.json', 'w') as f:
      json.dump(self.config, f, indent=2, sort_keys=True)
      os.chmod("/data/kegman.json", 0o764)
