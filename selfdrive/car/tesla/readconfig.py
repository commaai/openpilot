import ConfigParser

config_path = '/data/bb_openpilot.cfg'
config_file_r = 'r'
config_file_w = 'wb'

def read_config_file(CS):
    configr = ConfigParser.ConfigParser()
    try:
      configr.read(config_path)
    except:
      print "no config file, creating with defaults..."
    config = ConfigParser.RawConfigParser()
    config.add_section('OP_CONFIG')
    
    #force_pedal_over_cc -> CS.forcePedalOverCC
    try:
      CS.forcePedalOverCC = configr.getboolean('OP_CONFIG','force_pedal_over_cc')
    except:
      CS.forcePedalOverCC = True
    config.set('OP_CONFIG', 'force_pedal_over_cc', CS.forcePedalOverCC)
    
    #enable_hso -> CS.enableHSO
    try:
      CS.enableHSO = configr.getboolean('OP_CONFIG','enable_hso')
    except:
      CS.enableHSO = True
    config.set('OP_CONFIG', 'enable_hso', CS.enableHSO)

    #enable_alca -> CS.enableALCA
    try:
      CS.enableALCA = configr.getboolean('OP_CONFIG','enable_alca')
    except:
      CS.enableALCA = True
    config.set('OP_CONFIG', 'enable_alca', CS.enableALCA)

    #enable_das_emulation -> CS.enableDasEmulation
    try:
      CS.enableDasEmulation = configr.getboolean('OP_CONFIG','enable_das_emulation')
    except:
      CS.enableDasEmulation = True
    config.set('OP_CONFIG', 'enable_das_emulation', CS.enableDasEmulation)

    #enable_radar_emulation -> CS.enableRadarEmulation
    try:
      CS.enableRadarEmulation = configr.getboolean('OP_CONFIG','enable_radar_emulation')
    except:
      CS.enableRadarEmulation = True
    config.set('OP_CONFIG', 'enable_radar_emulation', CS.enableRadarEmulation)

    #enable_speed_variable_angle -> CS.enableSpeedVariableDesAngle
    try:
      CS.enableSpeedVariableDesAngle = configr.getboolean('OP_CONFIG','enable_speed_variable_angle')
    except:
      CS.enableSpeedVariableDesAngle = True
    config.set('OP_CONFIG', 'enable_speed_variable_angle', CS.enableSpeedVariableDesAngle)

    #enable_roll_angle_correction -> CS.enableRollAngleCorrection
    try:
      CS.enableRollAngleCorrection = configr.getboolean('OP_CONFIG','enable_roll_angle_correction')
    except:
      CS.enableRollAngleCorrection = False
    config.set('OP_CONFIG', 'enable_roll_angle_correction', CS.enableRollAngleCorrection)

    #enable_feed_forward_angle_correction -> CS.enableFeedForwardAngleCorrection
    try:
      CS.enableFeedForwardAngleCorrection = configr.getboolean('OP_CONFIG','enable_feed_forward_angle_correction')
    except:
      CS.enableFeedForwardAngleCorrection = True

    config.set('OP_CONFIG', 'enable_feed_forward_angle_correction', CS.enableFeedForwardAngleCorrection)

    #enable_driver_monitor -> CS.enableDriverMonitor
    try:
      CS.enableDriverMonitor = configr.getboolean('OP_CONFIG','enable_driver_monitor')
    except:
      CS.enableDriverMonitor = True

    config.set('OP_CONFIG', 'enable_driver_monitor', CS.enableDriverMonitor)


    with open(config_path, config_file_w) as configfile:
      config.write(configfile)
    
