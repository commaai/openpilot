import ConfigParser

config_path = '/data/honda_openpilot.cfg'
config_file_r = 'r'
config_file_w = 'wb'

def read_config_file(CS):
    file_changed = False
    configr = ConfigParser.ConfigParser()
    try:
      configr.read(config_path)
    except:
      file_changed = True
      print "no config file, creating with defaults..."
    config = ConfigParser.RawConfigParser()
    config.add_section('OP_CONFIG')
    
      
    #use_tesla_radar -> CS.useTeslaRadar
    try:
      CS.useTeslaRadar = configr.getboolean('OP_CONFIG','use_tesla_radar')
    except:
      CS.useTeslaRadar = False
      file_changed = True
    config.set('OP_CONFIG', 'use_tesla_radar', CS.useTeslaRadar)

    #radar_vin -> CS.radarVIN
    try:
      CS.radarVIN = configr.get('OP_CONFIG','radar_vin')
    except:
      CS.radarVIN = "                 "
      file_changed = True
    config.set('OP_CONFIG', 'radar_vin', CS.radarVIN)

    #radar_offset -> CS.radarOffset
    try:
      CS.radarOffset = configr.getfloat('OP_CONFIG','radar_offset')
    except:
      CS.radarOffset = 0.
      file_changed = True
    config.set('OP_CONFIG', 'radar_offset', CS.radarOffset)

    if file_changed:
      with open(config_path, config_file_w) as configfile:
        config.write(configfile)

class CarSettings(object):
  def __init__(self):
    ### START OF MAIN CONFIG OPTIONS ###
    ### Do NOT modify here, modify in /data/bb_openpilot.cfg and reboot
    self.useTeslaRadar = False
    self.radarVIN = "                 "
    self.radarOffset = 0.
    #read config file
    read_config_file(self)
    ### END OF MAIN CONFIG OPTIONS ###

  def get_value(self,name_of_variable):
    return_val = None
    exec("%s = self.%s" % ('return_val',name_of_variable))
    return return_val
