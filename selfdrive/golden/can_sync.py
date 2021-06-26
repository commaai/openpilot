#!/usr/bin/env python3
#pylint: skip-file
# flake8: noqa
#from cereal import car
#from common.params import Params
import cereal.messaging as messaging
import os
import cereal.messaging.messaging_pyx as messaging_pyx
import time
#from cereal import log
from opendbc.can.parser import CANParser
import json
from common.realtime import Ratekeeper

###########################################
#
#
# Sound Logic start
#
#
###########################################
import platform

BASE_PATH='/data/openpilot/selfdrive/'
SOUND_PATH=BASE_PATH + 'golden/sounds/'
OP_SOUND_PATH=BASE_PATH + 'assets/sounds/'
SOUND_PLAYER=BASE_PATH + 'sound_player/sound_player '

is_osx = platform.system() == 'Darwin'
if is_osx:
  SOUND_PATH='/Users/lixin/Project/op_sync/golden/sounds/'
  OP_SOUND_PATH='/Users/lixin/Project/openpilot/selfdrive/assets/sounds/'
  SOUND_PLAYER='afplay '

def play_sound(snd_file, play_time, time_gap):
    t_now = time.time()
    if (t_now - play_time) < time_gap:
      return play_time
    cmd = SOUND_PLAYER + snd_file
    print(cmd)
    os.system(cmd)
    return t_now

###########################################
#
#
# Sound Logic end
#
#
###########################################


def can_sync_thread():

    can_sock = messaging.sub_sock('can', conflate=True, timeout=100)
    can_pub = None

    os.environ["ZMQ"] = "1"
    pub_context = messaging_pyx.Context()
    can_pub = messaging_pyx.PubSocket()
    can_pub.connect(pub_context, 'testJoystick')
    del os.environ["ZMQ"]

    rk = Ratekeeper(100.0, print_delay_threshold=None)

    cc_main = 0
    cc_status = 0
    speed = 0.0

    dbc_file = "toyota_yaris"
    signals = [
        # sig_name, sig_address, default
        ("GEAR", "GEAR_PACKET", 0),
        ("BRAKE_PRESSED", "BRAKE_MODULE2", 0),
        ("GAS_PEDAL", "GAS_PEDAL", 0),
        ("SEATBELT_DRIVER_UNLATCHED", "SEATS_DOORS", 0),
        ("DOOR_OPEN_FL", "SEATS_DOORS", 0),
        ("DOOR_OPEN_FR", "SEATS_DOORS", 0),
        ("DOOR_OPEN_RL", "SEATS_DOORS", 0),
        ("DOOR_OPEN_RR", "SEATS_DOORS", 0),
        ("MAIN_ON", "PCM_CRUISE_SM", 0),
        ("CRUISE_CONTROL_STATE", "PCM_CRUISE_SM", 0),
        ("TURN_SIGNALS", "STEERING_LEVERS", 0),   # 3 is no blinkers
        ("ENGINE_RPM", "POWERTRAIN", 0),
        ("SPEED", "SPEED", 0),
        ("MAY_CONTAIN_LIGHTS", "BOOLS", 0),
        ("CHANGES_EACH_RIDE", "SLOW_VARIABLE_INFOS", 0),
        ("INCREASING_VALUE_FUEL", "SLOW_VARIABLE_INFOS", 0)
    ]
    checks = []
    parser = CANParser(dbc_file, signals, checks, 0)
    play_time = 0

    while True:
        try:
            can_strs = messaging.drain_sock_raw(can_sock, wait_for_one=True)
            parser.update_strings(can_strs)

            # print (parser.vl)

            cc_main = parser.vl['PCM_CRUISE_SM']['MAIN_ON']
            cc_status = parser.vl['PCM_CRUISE_SM']['CRUISE_CONTROL_STATE']
            speed = parser.vl['SPEED']['SPEED']

            doorOpen = any([parser.vl["SEATS_DOORS"]['DOOR_OPEN_FL'], parser.vl["SEATS_DOORS"]['DOOR_OPEN_FR'],
                        parser.vl["SEATS_DOORS"]['DOOR_OPEN_RL'], parser.vl["SEATS_DOORS"]['DOOR_OPEN_RR']])
            seatbeltUnlatched = parser.vl["SEATS_DOORS"]['SEATBELT_DRIVER_UNLATCHED'] != 0
            light = parser.vl["BOOLS"]["MAY_CONTAIN_LIGHTS"]
            a = parser.vl["SLOW_VARIABLE_INFOS"]["CHANGES_EACH_RIDE"]
            b = parser.vl["SLOW_VARIABLE_INFOS"]["INCREASING_VALUE_FUEL"]

            if doorOpen:
                play_time = play_sound(SOUND_PATH + 'door_open.wav', play_time, 3)

            if seatbeltUnlatched:
                play_time = play_sound(SOUND_PATH + 'seatbelt.wav', play_time, 3)

            if light != 0:
                play_time = play_sound(SOUND_PATH + 'turn_signal.wav', play_time, 3)

            cc_main_v = 0
            if cc_main:
                cc_main_v = 1

            can_dict = {
                'cc_main' : cc_main_v,
                'cc_status' : cc_status,
                'speed' : speed,
                'light' : light,
                'a': a,
                'b': b
            }

            json_str = json.dumps(can_dict)
            json_str = json_str.replace('\'', '"')
            can_pub.send(json_str)

        except messaging_pyx.MessagingError:
            print('MessagingError error happens')

    rk.keep_time()

def main():
  can_sync_thread()

if __name__ == "__main__":
  main()
