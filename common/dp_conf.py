#!/usr/bin/env python3.7
import os
import sys
import json
import time
from math import floor

'''
* type: Bool, Int8, UInt8, UInt16, Float32
* conf_type: param, struct
* dependencies needs to use struct and loaded prior so we don't have to read the param multiple times.
* update_once: True, False (the param will only load up once.)
'''
confs = [
  # thermald data
  {'name': 'dp_thermal_started', 'default': False, 'type': 'Bool', 'conf_type': ['struct']},
  {'name': 'dp_thermal_overheat', 'default': False, 'type': 'Bool', 'conf_type': ['struct']},

  {'name': 'dp_atl', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct'], 'update_once': True},
  # dashcam related
  {'name': 'dp_dashcamd', 'default': 0, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  # {'name': 'dp_dashcam_ui', 'default': 0, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  # auto shutdown
  {'name': 'dp_auto_shutdown', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_auto_shutdown_in', 'default': 90, 'type': 'UInt16', 'min': 1, 'max': 65535, 'depends': [{'name': 'dp_auto_shutdown', 'vals': [True]}], 'conf_type': ['param']},
  # service
  {'name': 'dp_updated', 'default': True, 'type': 'Bool', 'conf_type': ['param']},
  {'name': 'dp_logger', 'default': True, 'type': 'Bool', 'conf_type': ['param']},
  {'name': 'dp_athenad', 'default': True, 'type': 'Bool', 'depends': [{'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_uploader', 'default': True, 'type': 'Bool', 'depends': [{'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  # {'name': 'dp_gpxd', 'default': False, 'type': 'Bool', 'conf_type': ['param']},
  {'name': 'dp_hotspot_on_boot', 'default': False, 'type': 'Bool', 'conf_type': ['param']},
  # lat ctrl
  # {'name': 'dp_steering_limit_alert', 'default': True, 'type': 'Bool', 'depends': [{'name': 'dp_lat_ctrl', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_steering_on_signal', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_lat_ctrl', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_signal_off_delay', 'default': 0, 'type': 'UInt8', 'min': 0, 'max': 10, 'conf_type': ['param', 'struct']},
  # assist/auto lane change
  {'name': 'dp_lateral_mode', 'default': 1, 'type': 'UInt8', 'min': 0, 'max': 2, 'conf_type': ['param', 'struct']},
  {'name': 'dp_lc_min_mph', 'default': 45, 'type': 'Float32', 'min': 0, 'max': 255., 'depends': [{'name': 'dp_lateral_mode', 'vals': [1,2]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_lc_auto_cont', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_lateral_mode', 'vals': [2]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_lc_auto_min_mph', 'default': 60, 'type': 'Float32', 'min': 0, 'max': 255., 'depends': [{'name': 'dp_lateral_mode', 'vals': [2]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_lc_auto_delay', 'default': 3, 'type': 'Float32', 'min': 0, 'max': 10., 'depends': [{'name': 'dp_lateral_mode', 'vals': [2]}], 'conf_type': ['param', 'struct']},
  # long ctrl
  {'name': 'dp_allow_gas', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  # {'name': 'dp_max_ctrl_speed', 'default': 92., 'type': 'Float32', 'conf_type': ['param', 'struct']},
  # {'name': 'dp_lead_car_alert', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  # {'name': 'dp_lead_car_away_alert', 'default': True, 'type': 'Bool', 'conf_type': ['param']},

  {'name': 'dp_following_profile_ctrl', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_following_profile', 'default': 0, 'type': 'UInt8', 'min': 0, 'max': 2, 'depends': [{'name': 'dp_atl', 'vals': [False]}, {'name': 'dp_following_profile_ctrl', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_accel_profile_ctrl', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_accel_profile', 'default': 0, 'type': 'UInt8', 'min': 0, 'max': 2, 'depends': [{'name': 'dp_atl', 'vals': [False]}, {'name': 'dp_accel_profile_ctrl', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  # safety
  {'name': 'dp_gear_check', 'default': True, 'type': 'Bool', 'depends': [{'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_speed_check', 'default': True, 'type': 'Bool', 'depends': [{'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_temp_monitor', 'default': True, 'type': 'Bool', 'conf_type': ['param']},
  # UIs
  {'name': 'dp_ui_display_mode', 'default': 0, 'type': 'UInt8', 'min': 0, 'max': 2, 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_speed', 'default': True, 'type': 'Bool', 'depends': [], 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_event', 'default': True, 'type': 'Bool', 'depends': [], 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_max_speed', 'default': True, 'type': 'Bool', 'depends': [], 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_face', 'default': True, 'type': 'Bool', 'depends': [], 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_lane', 'default': True, 'type': 'Bool', 'depends': [], 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_lead', 'default': True, 'type': 'Bool', 'depends': [], 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_dev', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_dev_mini', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_blinker', 'default': False, 'type': 'Bool', 'depends': [], 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_brightness', 'default': 0, 'type': 'UInt8', 'min': 0, 'max': 100, 'conf_type': ['param', 'struct']},
  {'name': 'dp_ui_volume', 'default': -5, 'type': 'Int8', 'min': -5, 'max': 100, 'conf_type': ['param', 'struct']},
  # Apps
  # {'name': 'dp_app_auto_update', 'default': True, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_app_ext_gps', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_app_tomtom', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_app_tomtom_auto', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_app_tomtom', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_app_tomtom_manual', 'default': 0, 'type': 'Int8', 'min': -1, 'max': 1, 'depends': [{'name': 'dp_app_tomtom', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  # {'name': 'dp_app_autonavi', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  # {'name': 'dp_app_autonavi_auto', 'default':  False, 'type': 'Bool', 'depends': [{'name': 'dp_app_autonavi', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  # {'name': 'dp_app_autonavi_manual', 'default': 0, 'type': 'Int8', 'min': -1, 'max': 1, 'depends': [{'name': 'dp_app_autonavi', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  # {'name': 'dp_app_aegis', 'default':  False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  # {'name': 'dp_app_aegis_auto', 'default':  False, 'type': 'Bool', 'depends': [{'name': 'dp_app_aegis', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  # {'name': 'dp_app_aegis_manual', 'default': 0, 'type': 'Int8', 'min': -1, 'max': 1, 'depends': [{'name': 'dp_app_aegis', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_app_mixplorer', 'default':  False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_app_mixplorer_manual', 'default': 0, 'type': 'Int8', 'min': -1, 'max': 1, 'depends': [{'name': 'dp_app_mixplorer', 'vals': [True]}], 'conf_type': ['param', 'struct']},
  # custom car
  {'name': 'dp_car_selected', 'default': '', 'type': 'Text', 'conf_type': ['param']},
  {'name': 'dp_car_list', 'default': '', 'type': 'Text', 'conf_type': ['param']},
  {'name': 'dp_car_detected', 'default': '', 'type': 'Text', 'conf_type': ['param', 'struct']},
  # toyota
  {'name': 'dp_toyota_ldw', 'default': True, 'type': 'Bool', 'depends': [{'name': 'dp_car_detected', 'vals': ['toyota']}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_toyota_sng', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_car_detected', 'vals': ['toyota']}, {'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_toyota_zss', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_car_detected', 'vals': ['toyota']}], 'conf_type': ['param']},
  # {'name': 'dp_toyota_lowest_cruise_override', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_car_detected', 'vals': ['toyota']}, {'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  # {'name': 'dp_toyota_lowest_cruise_override_vego', 'default': False, 'type': 'Bool', 'depends': [{'name': 'dp_car_detected', 'vals': ['toyota']}, {'name': 'dp_atl', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  # {'name': 'dp_toyota_lowest_cruise_override_at', 'default': 44, 'type': 'Float32', 'depends': [{'name': 'dp_car_detected', 'vals': ['toyota']}, {'name': 'dp_toyota_lowest_cruise_override', 'vals': [True]}], 'min': 0, 'max': 255., 'conf_type': ['param', 'struct']},
  # {'name': 'dp_toyota_lowest_cruise_override_speed', 'default': 32, 'type': 'Float32', 'depends': [{'name': 'dp_car_detected', 'vals': ['toyota']}, {'name': 'dp_toyota_lowest_cruise_override_speed', 'vals': [True]}], 'min': 0, 'max': 255., 'conf_type': ['param', 'struct']},
  # hyundai
  {'name': 'dp_hkg_smart_mdps', 'default': False, 'type': 'Bool', 'conf_type': ['param']},
  # honda
  {'name': 'dp_honda_eps_mod', 'default': False, 'type': 'Bool', 'conf_type': ['param']},
  # volkswagen
  {'name': 'dp_vw_panda', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct'], 'update_once': True},
  {'name': 'dp_vw_timebomb_assist', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  #misc
  {'name': 'dp_ip_addr', 'default': '', 'type': 'Text', 'conf_type': ['struct']},
  {'name': 'dp_fan_mode', 'default': 0, 'type': 'UInt8', 'min': 0, 'max': 2, 'conf_type': ['param']},
  {'name': 'dp_last_modified', 'default': str(floor(time.time())), 'type': 'Text', 'conf_type': ['param']},
  {'name': 'dp_camera_offset', 'default': 6, 'type': 'Int8', 'min': -100, 'max': 100, 'conf_type': ['param', 'struct']},
  {'name': 'dp_path_offset', 'default': 0, 'type': 'Int8', 'min': -100, 'max': 100, 'conf_type': ['param', 'struct']},

  {'name': 'dp_locale', 'default': 'en-US', 'type': 'Text', 'conf_type': ['param', 'struct'], 'update_once': True},
  {'name': 'dp_disable_relay', 'default': False, 'type': 'Bool', 'conf_type': ['param']},
  {'name': 'dp_reg', 'default': False, 'type': 'Bool', 'conf_type': ['param']},

  {'name': 'dp_sr_learner', 'default': True, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_sr_custom', 'default': 9.99, 'min': 9.99, 'max': 30, 'type': 'Float32', 'depends': [{'name': 'dp_sr_learner', 'vals': [False]}], 'conf_type': ['param', 'struct']},
  {'name': 'dp_sr_stock', 'default': 9.99, 'min': 9.99, 'max': 100, 'type': 'Float32', 'conf_type': ['param', 'struct'], 'update_once': True},
  {'name': 'dp_lqr', 'default': False, 'type': 'Bool', 'conf_type': ['param']},
  {'name': 'dp_reset_live_param_on_start', 'default': False, 'type': 'Bool', 'conf_type': ['param']},
  {'name': 'dp_jetson', 'default': False, 'type': 'Bool', 'conf_type': ['param'], 'update_once': True},
  {'name': 'dp_debug', 'default': False, 'type': 'Bool', 'conf_type': ['param', 'struct']},
  {'name': 'dp_car_assigned', 'default': '', 'type': 'Text', 'conf_type': ['param'], 'update_once': True},
  {'name': 'dp_car_list', 'default': '', 'type': 'Text', 'conf_type': ['param'], 'update_once': True},
  {'name': 'dp_no_batt', 'default': False, 'type': 'Bool', 'conf_type': ['param'], 'update_once': True},
  {'name': 'dp_panda_fake_black', 'default': False, 'type': 'Bool', 'conf_type': ['param'], 'update_once': True},
  {'name': 'dp_panda_no_gps', 'default': False, 'type': 'Bool', 'conf_type': ['param'], 'update_once': True},
  {'name': 'dp_last_candidate', 'default': '', 'type': 'Text', 'conf_type': ['param'], 'update_once': True},
]

def get_definition(name):
  for conf in confs:
    if conf['name'] == name:
      return conf
  return None

def to_param_val(name, val):
  conf = get_definition(name)
  if conf is not None:
    type = conf['type'].lower()
    try:
      if 'bool' in type:
        val = '1' if val else '0'
      elif 'int' in type:
        val = int(val)
      elif 'float' in type:
        val = float(val)
      return str(val)
    except (ValueError, TypeError):
      return ''
  return ''

def to_struct_val(name, val):
  conf = get_definition(name)
  if conf is not None:
    try:
      type = conf['type'].lower()
      if 'bool' in type:
        val = True if val == '1' else False
      elif 'int' in type:
        val = int(val)
      elif 'float' in type:
        val = float(val)
      return val
    except (ValueError, TypeError):
      return None
  return None

'''
function to convert param name into struct name.
'''
def get_struct_name(snake_str):
  components = snake_str.split('_')
  # We capitalize the first letter of each component except the first one
  # with the 'title' method and join them together.
  return components[0] + ''.join(x.title() for x in components[1:])

'''
function to generate struct for log.capnp
'''
def gen_log_struct():
  count = 0
  str = "# dp\n"
  str += "struct DragonConf {\n"
  for conf in confs:
    name = get_struct_name(conf['name'])
    if 'struct' in conf['conf_type']:
      str += f"  {name} @{count} :{conf['type']};\n"
      count += 1
  str += "}"
  print(str)

'''
function to generate support car list
'''
def get_support_car_list():
  attrs = ['FINGERPRINTS', 'FW_VERSIONS']
  cars = dict({"cars": []})
  for car_folder in [x[0] for x in os.walk('/data/openpilot/selfdrive/car')]:
    try:
      car_name = car_folder.split('/')[-1]
      if car_name != "mock":
        names = []
        for attr in attrs:
          values = __import__('selfdrive.car.%s.values' % car_name, fromlist=[attr])
          if hasattr(values, attr):
            attr_values = getattr(values, attr)
          else:
            continue
          if isinstance(attr_values, dict):
            for f, v in attr_values.items():
              if f not in names:
                names.append(f)
          names.sort()
        brand_models = {"brand": car_name.upper(), "models": names }
        cars["cars"].append(brand_models)
    except (ImportError, IOError, ValueError):
      pass
  print(json.dumps(cars))
  return json.dumps(cars)

'''
function to init param value.
should add this into manager.py
'''
def init_params_vals(params):
  for conf in confs:
    if 'param' in conf['conf_type']:
      if conf['name'] == 'dp_car_list':
        params.put(conf['name'], get_support_car_list())
      elif params.get(conf['name']) is None:
        params.put(conf['name'], to_param_val(conf['name'], conf['default']))

def gen_params_cc_keys():
  for conf in confs:
    if 'param' in conf['conf_type']:
      print("{\"%s\", PERSISTENT}," % conf['name'])


if __name__ == "__main__":
  if (len(sys.argv) > 1) and sys.argv[1] == 'cc':
    gen_params_cc_keys()
  else:
    gen_log_struct()
