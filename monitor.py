#!/usr/bin/env python
import sys
import zmq
from cereal import log
from selfdrive.services import service_list
import yaml
#import netifaces as ni
addrMap = {
    228: "STEERING_CONTROL",
    232: "BRAKE_HOLD",
    330: "STEERING_SENSORS",
    344: "ENGINE_DATA",
    380: "POWERTRAIN_DATA",
    399: "STEER_STATUS",
    420: "VSA_STATUS",
    450: "EPB_STATUS",
    464: "WHEEL_SPEEDS",
    479: "ACC_CONTROL",
    495: "ACC_CONTROL_ON",
    597: "ROUGH_WHEEL_SPEED",
    662: "SCM_BUTTONS",
    773: "SEATBELT_STATUS",
    777: "CAR_SPEED",
    780: "ACC_HUD",
    804: "CRUISE",
    806: "SCM_FEEDBACK",
    829: "LKAS_HUD",
    862: "CAMERA_MESSAGES",
    884: "STALK_STATUS",
    419: "GEARBOX",
    432: "STANDSTILL",
    446: "BRAKE_MODULE",
    927: "RADAR_HUD",
    1302: "ODOMETER"
}

# def has_address(address):
#     ips = []
#     for nfi in ni.interfaces():
#         addr = ni.ifaddresses(nfi)
#         if ni.AF_INET in addr:
#             ips.append(ni.ifaddresses(nfi)[ni.AF_INET][0]['addr'].encode("ascii"))
#     return len([i for i in ips if address in i])>0

#ip="192.168.8.244" if has_address("192.168.8.") else "192.168.43.1"
ip = "127.0.0.1"

class Monitor(object):
    def __init__(self, port):

        context = zmq.Context()
        self.logcan = context.socket(zmq.SUB)
        print "tcp://%s:%s" % (ip, port)
        self.logcan.connect("tcp://%s:%s" % (ip, port))
        self.logcan.setsockopt(zmq.SUBSCRIBE, b"")

    def update(self):
        return log.Event.from_bytes(self.logcan.recv())


if __name__ == "__main__":
    port = "8006"
    service = None
    if len(sys.argv) > 1:
        service = sys.argv[1]
        port = service_list[service].port

    M = Monitor(port)
    while 1:
        ret = M.update()
        if service == "logMessage":
            msg = yaml.safe_load(ret.logMessage)
            if "running" not in msg['msg']:
                print msg['msg']
            if "crash" in msg['msg']:
                print msg['exc_info']
        else:
            print ret
