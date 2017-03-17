import zmq
import selfdrive.messaging as messaging
from common.services import service_list
import numpy as np
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper

import time

def wavevisiond_thread(gctx,rate=100):
    set_realtime_priority(1)
    
    context = zmq.Context()

    waveModel = messaging.pub_sock(context, service_list['waveModel'].port)
    
    while True:
        msg = messaging.new_message()
        msg.init('model')
        msg.path.points=[1.0,5.0,4.0,4.0,5.0]
        msg.path.prob = 0.743
        msg.path.std = 0.2

        msg.leftLane.points = [5.0, 4.0, 3.0, 2.0, 1.0 ]
        msg.leftLane.prob = 0.987
        msg.leftLane.std = 0.1

        waveModel.send(msg.to_bytes())
        time.sleep(6)
    
    
def main(gctx=None):
    wavevisiond_thread(gctx, 100)
