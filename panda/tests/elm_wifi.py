
import os
import sys
import time
import socket
import select
import pytest
import struct

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from . import elm_car_simulator
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
from panda import Panda

def elm_connect():
    s = socket.create_connection(("192.168.0.10", 35000))
    s.setblocking(0)
    return s

def read_or_fail(s):
    ready = select.select([s], [], [], 4)
    assert ready[0], "Socket did not receive data within the timeout duration."
    return s.recv(1000)

def sendrecv(s, dat):
    s.send(dat)
    return read_or_fail(s)

def send_compare(s, dat, ret, timeout=4):
    s.send(dat)
    res = b''
    while ret.startswith(res) and ret != res:
        print("Waiting")
        ready = select.select([s], [], [], timeout)
        if not ready[0]:
            print("current recv data:", repr(res))
            break;
        res += s.recv(1000)
    #print("final recv data: '%s'" % repr(res))
    assert ret == res#, "Data does not agree (%s) (%s)"%(repr(ret), repr(res))

def sync_reset(s):
    s.send("ATZ\r")
    res = b''
    while not res.endswith("ELM327 v1.5\r\r>"):
        res += read_or_fail(s)
        print("Reset response is '%s'" % repr(res))

def test_reset():
    s = socket.create_connection(("192.168.0.10", 35000))
    s.setblocking(0)

    try:
        sync_reset(s)
    finally:
        s.close()

def test_elm_cli():
    s = elm_connect()

    try:
        sync_reset(s)

        send_compare(s, b'ATI\r', b'ATI\rELM327 v1.5\r\r>')

        #Test Echo Off
        #Expected to be misimplimentation, but this is how the reference device behaved.
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') #Here is the odd part
        send_compare(s, b'ATE0\r', b'OK\r\r>')       #Should prob show this immediately
        send_compare(s, b'ATI\r', b'ELM327 v1.5\r\r>')

        #Test Newline On
        send_compare(s, b'ATL1\r', b'OK\r\n\r\n>')
        send_compare(s, b'ATI\r', b'ELM327 v1.5\r\n\r\n>')
        send_compare(s, b'ATL0\r', b'OK\r\r>')
        send_compare(s, b'ATI\r', b'ELM327 v1.5\r\r>')

        send_compare(s, b'ATI\r', b'ELM327 v1.5\r\r>') #Test repeat command no echo
        send_compare(s, b'\r', b'ELM327 v1.5\r\r>')

        send_compare(s, b'aTi\r', b'ELM327 v1.5\r\r>') #Test different case

        send_compare(s, b'  a     T i\r', b'ELM327 v1.5\r\r>') #Test with white space

        send_compare(s, b'ATCATHAT\r', b'?\r\r>') #Test Invalid AT command

        send_compare(s, b'01 00 00 00 00 00 00 00\r', b'?\r\r>') #Test Invalid (too long) OBD command
        send_compare(s, b'01 GZ\r', b'?\r\r>') #Test Invalid (Non hex chars) OBD command
    finally:
        s.close()

def test_elm_setget_protocol():
    s = elm_connect()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF

        send_compare(s, b'ATSP0\r', b"OK\r\r>") # Set auto
        send_compare(s, b'ATDP\r', b"AUTO\r\r>")
        send_compare(s, b'ATDPN\r', b"A0\r\r>")

        send_compare(s, b'ATSP6\r', b"OK\r\r>") # Set protocol
        send_compare(s, b'ATDP\r', b"ISO 15765-4 (CAN 11/500)\r\r>")
        send_compare(s, b'ATDPN\r', b"6\r\r>")

        send_compare(s, b'ATSPA6\r', b"OK\r\r>") # Set auto with protocol default
        send_compare(s, b'ATDP\r', b"AUTO, ISO 15765-4 (CAN 11/500)\r\r>")
        send_compare(s, b'ATDPN\r', b"A6\r\r>")

        send_compare(s, b'ATSP7\r', b"OK\r\r>")
        send_compare(s, b'ATDP\r', b"ISO 15765-4 (CAN 29/500)\r\r>")
        send_compare(s, b'ATDPN\r', b"7\r\r>") #Test Does not accept invalid protocols
        send_compare(s, b'ATSPD\r', b"?\r\r>")
        send_compare(s, b'ATDP\r', b"ISO 15765-4 (CAN 29/500)\r\r>")
        send_compare(s, b'ATDPN\r', b"7\r\r>")
    finally:
        s.close()

def test_elm_protocol_failure():
    s = elm_connect()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF

        send_compare(s, b'ATSP0\r', b"OK\r\r>")
        send_compare(s, b'0100\r', b"SEARCHING...\rUNABLE TO CONNECT\r\r>", timeout=10)

        send_compare(s, b'ATSP1\r', b"OK\r\r>")
        send_compare(s, b'0100\r', b"NO DATA\r\r>")

        send_compare(s, b'ATSP2\r', b"OK\r\r>")
        send_compare(s, b'0100\r', b"NO DATA\r\r>")

        send_compare(s, b'ATSP3\r', b"OK\r\r>")
        send_compare(s, b'0100\r', b"BUS INIT: ...ERROR\r\r>")

        send_compare(s, b'ATSP4\r', b"OK\r\r>")
        send_compare(s, b'0100\r', b"BUS INIT: ...ERROR\r\r>")

        send_compare(s, b'ATSP5\r', b"OK\r\r>")
        send_compare(s, b'0100\r', b"BUS INIT: ERROR\r\r>")

        #send_compare(s, b'ATSP6\r', b"OK\r\r>")
        #send_compare(s, b'0100\r', b"NO DATA\r\r>")
        #
        #send_compare(s, b'ATSP7\r', b"OK\r\r>")
        #send_compare(s, b'0100\r', b"NO DATA\r\r>")
        #
        #send_compare(s, b'ATSP8\r', b"OK\r\r>")
        #send_compare(s, b'0100\r', b"NO DATA\r\r>")
        #
        #send_compare(s, b'ATSP9\r', b"OK\r\r>")
        #send_compare(s, b'0100\r', b"NO DATA\r\r>")
        #
        #send_compare(s, b'ATSPA\r', b"OK\r\r>")
        #send_compare(s, b'0100\r', b"NO DATA\r\r>")
        #
        #send_compare(s, b'ATSPB\r', b"OK\r\r>")
        #send_compare(s, b'0100\r', b"NO DATA\r\r>")
        #
        #send_compare(s, b'ATSPC\r', b"OK\r\r>")
        #send_compare(s, b'0100\r', b"NO DATA\r\r>")
    finally:
        s.close()

def test_elm_protocol_autodetect_ISO14230_KWP_FAST():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, can=False)#, silent=True)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATH0\r', b'OK\r\r>') # Headers ON
        send_compare(s, b'ATS0\r', b"OK\r\r>")

        send_compare(s, b'ATSP0\r', b"OK\r\r>")
        send_compare(s, b'010D\r', b"SEARCHING...\r410D53\r\r>", timeout=10)
        send_compare(s, b'ATDPN\r', b"A5\r\r>")
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_basic_send_lin():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, can=False)#, silent=True)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATSP5\r', b"ATSP5\rOK\r\r>") # Set Proto

        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'0100\r', b"BUS INIT: OK\r41 00 FF FF FF FE \r\r>")
        send_compare(s, b'010D\r', b"41 0D 53 \r\r>")

        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces Off
        send_compare(s, b'0100\r', b"4100FFFFFFFE\r\r>")
        send_compare(s, b'010D\r', b"410D53\r\r>")

        send_compare(s, b'ATH1\r', b'OK\r\r>') # Spaces Off Headers On
        send_compare(s, b'0100\r', b"86F1104100FFFFFFFEC3\r\r>")
        send_compare(s, b'010D\r', b"83F110410D5325\r\r>")

        send_compare(s, b'ATS1\r', b'OK\r\r>') # Spaces On Headers On
        send_compare(s, b'0100\r', b"86 F1 10 41 00 FF FF FF FE C3 \r\r>")
        send_compare(s, b'010D\r', b"83 F1 10 41 0D 53 25 \r\r>")

        send_compare(s, b'1F00\r', b"NO DATA\r\r>") # Unhandled msg, no response.

        # Repeat last check to see if it still works after NO DATA was received
        send_compare(s, b'0100\r', b"86 F1 10 41 00 FF FF FF FE C3 \r\r>")
        send_compare(s, b'010D\r', b"83 F1 10 41 0D 53 25 \r\r>")
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_send_lin_multiline_msg():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, can=False)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATSP5\r', b"OK\r\r>") # Set Proto

        send_compare(s, b'0902\r', # headers OFF, Spaces ON
                     b"BUS INIT: OK\r"
                     "49 02 01 00 00 00 31 \r"
                     "49 02 02 44 34 47 50 \r"
                     "49 02 03 30 30 52 35 \r"
                     "49 02 04 35 42 31 32 \r"
                     "49 02 05 33 34 35 36 \r\r>")

        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces OFF
        send_compare(s, b'0902\r', # Headers OFF, Spaces OFF
                     b"49020100000031\r"
                     "49020244344750\r"
                     "49020330305235\r"
                     "49020435423132\r"
                     "49020533343536\r\r>")

        send_compare(s, b'ATH1\r', b'OK\r\r>') # Headers ON
        send_compare(s, b'0902\r', # Headers ON, Spaces OFF
                     b"87F1104902010000003105\r"
                     "87F11049020244344750E4\r"
                     "87F11049020330305235BD\r"
                     "87F11049020435423132B1\r"
                     "87F11049020533343536AA\r\r>")

        send_compare(s, b'ATS1\r', b'OK\r\r>') # Spaces ON
        send_compare(s, b'0902\r', # Headers ON, Spaces ON
                     b"87 F1 10 49 02 01 00 00 00 31 05 \r"
                     "87 F1 10 49 02 02 44 34 47 50 E4 \r"
                     "87 F1 10 49 02 03 30 30 52 35 BD \r"
                     "87 F1 10 49 02 04 35 42 31 32 B1 \r"
                     "87 F1 10 49 02 05 33 34 35 36 AA \r\r>")
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_send_lin_multiline_msg_throughput():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, can=False, silent=True)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATSP5\r', b"ATSP5\rOK\r\r>") # Set Proto
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces OFF
        send_compare(s, b'ATH0\r', b'OK\r\r>') # Headers OFF

        send_compare(s, b'09fc\r', # headers OFF, Spaces OFF
                     b"BUS INIT: OK\r" +
                     b''.join((b'49FC' + hex(num+1)[2:].upper().zfill(2) +
                               b'AAAA' + hex(num+1)[2:].upper().zfill(4) + b'\r'
                               for num in range(80))) +
                     b"\r>",
                     timeout=10
        )
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_panda_safety_mode_KWPFast():
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    p_car = Panda(serial) # Configure this so the messages will send
    p_car.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
    p_car.kline_drain()

    p_elm = Panda("WIFI")
    p_elm.set_safety_mode(Panda.SAFETY_ELM327);

    def get_checksum(dat):
        result = 0
        result += sum(map(ord, dat)) if isinstance(b'dat', str) else sum(dat)
        return struct.pack("B", result % 0x100)

    def timed_recv_check(p, bus, goodmsg):
        t = time.time()
        msg = bytearray()

        while time.time()-t < 0.5 and len(msg) != len(goodmsg):
            msg += p._handle.controlRead(Panda.REQUEST_OUT, 0xe0, bus, 0, len(goodmsg)-len(msg))
            #print("Received", repr(msg))
            if msg == goodmsg:
                return True
            time.sleep(0.01)
        return False

    def kline_send(p, x, bus=2):
        p.kline_drain(bus=bus)
        p._handle.bulkWrite(2, chr(bus).encode()+x)
        return timed_recv_check(p, bus, x)

    def did_send(priority, toaddr, fromaddr, dat, bus=2, checkbyte=None):
        msgout = struct.pack("BBB", priority | len(dat), toaddr, fromaddr) + dat
        msgout += get_checksum(msgout) if checkbyte is None else checkbyte
        print("Sending", hex(priority), hex(toaddr), hex(fromaddr), repr(msgout))

        if not kline_send(p_elm, msgout, bus=bus):
            return False
        return timed_recv_check(p_car, bus, msgout)

    assert not did_send(0xC0, 0x33, 0xF1, b'\x01\x0F', bus=3) #wrong bus
    assert not did_send(0xC0, 0x33, 0xF1, b'') #wrong length
    assert not did_send(0xB0, 0x33, 0xF1, b'\x01\x0E') #bad priority
    assert not did_send(0xC0, 0x00, 0xF1, b'\x01\x0D') #bad addr
    assert not did_send(0xC0, 0x33, 0x00, b'\x01\x0C') #bad addr

    assert did_send(0xC0, 0x33, 0xF1, b'\x01\x0B') #good! (obd func req)

def test_elm_lin_keepalive():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, can=False, silent=True)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATSP5\r', b"ATSP5\rOK\r\r>") # Set Proto
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces OFF
        send_compare(s, b'ATH0\r', b'OK\r\r>') # Headers OFF

        send_compare(s, b'0100\r', b"BUS INIT: OK\r4100FFFFFFFE\r\r>")
        assert sim.lin_active
        time.sleep(6)
        assert sim.lin_active

        send_compare(s, b'ATPC\r', b"OK\r\r>") #STOP KEEPALIVE
        assert sim.lin_active
        time.sleep(6)
        assert not sim.lin_active

    finally:
        sim.stop()
        sim.join()
        s.close()

#////////////
def test_elm_protocol_autodetect_ISO15765():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, lin=False, silent=True)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATH1\r', b'OK\r\r>') # Headers ON
        send_compare(s, b'ATS0\r', b"OK\r\r>")

        sim.can_mode_11b()
        send_compare(s, b'ATSP0\r', b"OK\r\r>")
        send_compare(s, b'010D\r', b"SEARCHING...\r7E803410D53\r\r>", timeout=10)
        send_compare(s, b'ATDPN\r', b"A6\r\r>")

        sim.can_mode_29b()
        send_compare(s, b'ATSP0\r', b"OK\r\r>")
        send_compare(s, b'010D\r', b"SEARCHING...\r18DAF11003410D53\r\r>", timeout=10)
        send_compare(s, b'ATDPN\r', b"A7\r\r>")

        sim.change_can_baud(250)

        sim.can_mode_11b()
        send_compare(s, b'ATSP0\r', b"OK\r\r>")
        send_compare(s, b'010D\r', b"SEARCHING...\r7E803410D53\r\r>", timeout=10)
        send_compare(s, b'ATDPN\r', b"A8\r\r>")

        sim.can_mode_29b()
        send_compare(s, b'ATSP0\r', b"OK\r\r>")
        send_compare(s, b'010D\r', b"SEARCHING...\r18DAF11003410D53\r\r>", timeout=10)
        send_compare(s, b'ATDPN\r', b"A9\r\r>")
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_basic_send_can():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, lin=False, silent=True)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATSP6\r', b"ATSP6\rOK\r\r>") # Set Proto

        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'0100\r', b"41 00 FF FF FF FE \r\r>")
        send_compare(s, b'010D\r', b"41 0D 53 \r\r>")

        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces Off
        send_compare(s, b'0100\r', b"4100FFFFFFFE\r\r>")
        send_compare(s, b'010D\r', b"410D53\r\r>")

        send_compare(s, b'ATH1\r', b'OK\r\r>') # Spaces Off Headers On
        send_compare(s, b'0100\r', b"7E8064100FFFFFFFE\r\r>")
        send_compare(s, b'010D\r', b"7E803410D53\r\r>")

        send_compare(s, b'ATS1\r', b'OK\r\r>') # Spaces On Headers On
        send_compare(s, b'0100\r', b"7E8 06 41 00 FF FF FF FE \r\r>")
        send_compare(s, b'010D\r', b"7E8 03 41 0D 53 \r\r>")

        send_compare(s, b'1F00\r', b"NO DATA\r\r>") # Unhandled msg, no response.

        # Repeat last check to see if it still works after NO DATA was received
        send_compare(s, b'0100\r', b"7E8 06 41 00 FF FF FF FE \r\r>")
        send_compare(s, b'010D\r', b"7E8 03 41 0D 53 \r\r>")
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_send_can_multimsg():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, lin=False)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATS1\r', b'OK\r\r>') # Spaces OFF
        send_compare(s, b'ATH1\r', b'OK\r\r>') # Headers ON

        send_compare(s, b'ATSP6\r', b"OK\r\r>") # Set Proto ISO 15765-4 (CAN 11/500)
        sim.can_add_extra_noise(b'\x03\x41\x0D\xFA', addr=0x7E9)# Inject message into the stream
        send_compare(s, b'010D\r',
                     b"7E8 03 41 0D 53 \r"
                     "7E9 03 41 0D FA \r\r>") # Check it was ignored.
    finally:
        sim.stop()
        sim.join()
        s.close()

"""The ability to correctly filter out messages with the wrong PID is not
implemented correctly in the reference device."""
def test_elm_can_check_mode_pid():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, lin=False)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces OFF
        send_compare(s, b'ATH0\r', b'OK\r\r>') # Headers OFF

        send_compare(s, b'ATSP6\r', b"OK\r\r>") # Set Proto ISO 15765-4 (CAN 11/500)
        sim.can_add_extra_noise(b'\x03\x41\x0E\xFA')# Inject message into the stream
        send_compare(s, b'010D\r', b"410D53\r\r>") # Check it was ignored.
        send_compare(s, b'0100\r', b"4100FFFFFFFE\r\r>") # Check it was ignored again.
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_send_can_multiline_msg():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, lin=False)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATSP6\r', b"ATSP6\rOK\r\r>") # Set Proto
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF

        send_compare(s, b'0902\r', # headers OFF, Spaces ON
                     b"014 \r"
                     "0: 49 02 01 31 44 34 \r"
                     "1: 47 50 30 30 52 35 35 \r"
                     "2: 42 31 32 33 34 35 36 \r\r>")

        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces OFF
        send_compare(s, b'0902\r', # Headers OFF, Spaces OFF
                     b"014\r"
                     "0:490201314434\r"
                     "1:47503030523535\r"
                     "2:42313233343536\r\r>")

        send_compare(s, b'ATH1\r', b'OK\r\r>') # Headers ON
        send_compare(s, b'0902\r', # Headers ON, Spaces OFF
                     b"7E81014490201314434\r"
                     "7E82147503030523535\r"
                     "7E82242313233343536\r\r>")

        send_compare(s, b'ATS1\r', b'OK\r\r>') # Spaces ON
        send_compare(s, b'0902\r', # Headers ON, Spaces ON
                     b"7E8 10 14 49 02 01 31 44 34 \r"
                     "7E8 21 47 50 30 30 52 35 35 \r"
                     "7E8 22 42 31 32 33 34 35 36 \r\r>")
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_send_can_multiline_msg_throughput():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, lin=False, silent=True)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATSP6\r', b"ATSP6\rOK\r\r>") # Set Proto
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces OFF
        send_compare(s, b'ATH1\r', b'OK\r\r>') # Headers ON

        rows = 584
        send_compare(s, b'09ff\r', # headers ON, Spaces OFF
                     ("7E8" + "1" + hex((rows*7)+6)[2:].upper().zfill(3) + "49FF01"+"AA0000\r" +
                      "".join(
                          ("7E82"+hex((num+1)%0x10)[2:].upper()+("AA"*5) +
                           hex(num+1)[2:].upper().zfill(4) + "\r" for num in range(rows))
                      ) + "\r>").encode(),
                     timeout=10
        )
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_interrupted_obd_cmd_resets_state():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, lin=False, silent=True)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces OFF
        s.send(b"09fd\r")
        ready = select.select([s], [], [], 4)
        assert ready[0], "Socket did not receive data within the timeout duration."
        s.send(b"ATI\r")

        assert b"236\r0:49FD01AAAAAA\r" in s.recv(10000)

        #Will likely have to be improved to scan for STOPPED if the FW gets more responsive.
        ready = select.select([s], [], [], 4)
        assert ready[0], "Socket did not receive data within the timeout duration."

        assert b"STOPPED" in s.recv(10000)

        sim.set_enable(False)
        send_compare(s, b'09fd\r', b"NO DATA\r\r>")
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_can_baud():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    sim = elm_car_simulator.ELMCarSimulator(serial, lin=False)
    sim.start()

    try:
        sync_reset(s)
        send_compare(s, b'ATE0\r', b'ATE0\rOK\r\r>') # Echo OFF
        send_compare(s, b'ATS0\r', b'OK\r\r>') # Spaces OFF
        send_compare(s, b'ATH1\r', b'OK\r\r>') # Headers ON

        send_compare(s, b'ATSP6\r', b"OK\r\r>") # Set Proto ISO 15765-4 (CAN 11/500)
        send_compare(s, b'0100\r', b"7E8064100FFFFFFFE\r\r>")

        send_compare(s, b'ATSP8\r', b"OK\r\r>") # Set Proto ISO 15765-4 (CAN 11/250)
        send_compare(s, b'0100\r', b"CAN ERROR\r\r>")

        sim.change_can_baud(250)

        send_compare(s, b'ATSP6\r', b"OK\r\r>") # Set Proto ISO 15765-4 (CAN 11/500)
        send_compare(s, b'0100\r', b"CAN ERROR\r\r>")

        send_compare(s, b'ATSP8\r', b"OK\r\r>") # Set Proto ISO 15765-4 (CAN 11/250)
        send_compare(s, b'0100\r', b"7E8064100FFFFFFFE\r\r>")
    finally:
        sim.stop()
        sim.join()
        s.close()

def test_elm_panda_safety_mode_ISO15765():
    s = elm_connect()
    serial = os.getenv("CANSIMSERIAL") if os.getenv("CANSIMSERIAL") else None
    p_car = Panda(serial) # Configure this so the messages will send
    p_car.set_can_speed_kbps(0, 500)
    p_car.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

    p_elm = Panda("WIFI")
    p_elm.set_safety_mode(Panda.SAFETY_ELM327);

    #sim = elm_car_simulator.ELMCarSimulator(serial, lin=False)
    #sim.start()

    def did_send(p, addr, dat, bus):
        p.can_send(addr, dat, bus)
        t = time.time()
        while time.time()-t < 0.5:
            msg = p.can_recv()
            for addrin, _, datin, busin in msg:
                if (0x80 | bus) == busin and addr == addrin and datin == dat:
                    return True
            time.sleep(0.01)
        return False

    try:
        sync_reset(s) # Reset elm (which requests the ELM327 safety mode)

        #29 bit
        assert not did_send(p_elm, 0x18DB33F1, b'\x02\x01\x00\x00\x00\x00\x00\x00', 1) #wrong canid
        assert not did_send(p_elm, 0x18DB33F1, b'\x02\x01\x00', 0) #wrong length
        assert not did_send(p_elm, 0x10000000, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #bad addr
        assert not did_send(p_elm, 0x18DAF133, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #bad addr (phy addr)
        assert not did_send(p_elm, 0x18DAF000, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #bad addr
        assert not did_send(p_elm, 0x18DAF1F3, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #bad addr! (phys rsp to elm)

        assert did_send(p_elm, 0x18DB33F1, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #good! (obd func req)
        assert did_send(p_elm, 0x18DA10F1, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #good! (phys response)

        #11 bit
        assert not did_send(p_elm, 0X7DF, b'\x02\x01\x00\x00\x00\x00\x00\x00', 1) #wrong canid
        assert not did_send(p_elm, 0X7DF, b'\x02\x01\x00', 0) #wrong length
        assert not did_send(p_elm, 0xAA, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #bad addr
        assert not did_send(p_elm, 0x7DA, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #bad addr (phy addr)
        assert not did_send(p_elm, 0x7E8, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #bad addr (sending 'response')

        assert did_send(p_elm, 0x7DF, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #good! (obd func req)
        assert did_send(p_elm, 0x7E1, b'\x02\x01\x00\x00\x00\x00\x00\x00', 0) #good! (phys response)

    finally:
        s.close()
