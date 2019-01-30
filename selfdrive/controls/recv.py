import zmq
import time
import selfdrive.messaging as messaging

context = zmq.Context()
uievent = messaging.sub_sock(context, 8064)

while True:
    uie = None
    try:
        uie = uievent.recv(zmq.NOBLOCK)
    except zmq.error.Again:
        uie = None
    if uie is not None:
        print uie
    if uie=="2":
        print "event 2"
    time.sleep(.2)

