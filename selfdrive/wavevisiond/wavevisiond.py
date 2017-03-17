import zmq
import sys
import selfdrive.messaging as messaging
from common.services import service_list
import numpy as np
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper

import time

def wavevisiond_thread(gctx,rate=100):
    print('line 10')
    set_realtime_priority(1)
    print('line 12')
    context = zmq.Context()
    print('line 14')
    waveModel = messaging.pub_sock(context, service_list['waveModel'].port)
    print('line 16')

    while True:
        print('line 19')
        msg = messaging.new_message()
        print('line 21')
        msg.init('model')
        print('line 23')
        msg.path.points=[1.0,5.0,4.0,4.0,5.0]
        msg.path.prob = 0.743
        msg.path.std = 0.2
        print('line 27')

        msg.leftLane.points = [5.0, 4.0, 3.0, 2.0, 1.0 ]
        msg.leftLane.prob = 0.987
        msg.leftLane.std = 0.1
        print('line 32')

        waveModel.send(msg.to_bytes())
        print('line 35')
        time.sleep(6)
        print('line 37')
    
    
def main(gctx=None):
    print('line 41')
    wavevisiond_thread(gctx, 100)

if __name__ == "__main__":
    print('sys version', sys.version)
    main()