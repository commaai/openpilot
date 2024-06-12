#!/usr/bin/env python3

"""Used to Reverse/Test ELM protocol auto detect and OBD message response without a car."""

import os
import sys
import struct
import binascii
import time
import threading
from collections import deque

from panda import Panda

def lin_checksum(dat):
    return sum(dat) % 0x100

class ELMCarSimulator():
    def __init__(self, sn, silent=False, can_kbaud=500,
                 can=True, can11b=True, can29b=True,
                 lin=True):
        self.__p = Panda(sn if sn else Panda.list()[0])
        self.__on = True
        self.__stop = False
        self.__silent = silent

        self.__lin_timer = None
        self.__lin_active = False
        self.__lin_enable = lin
        self.__lin_monitor_thread = threading.Thread(target=self.__lin_monitor)

        self.__can_multipart_data = None
        self.__can_kbaud = can_kbaud
        self.__can_extra_noise_msgs = deque()
        self.__can_enable = can
        self.__can11b = can11b
        self.__can29b = can29b
        self.__can_monitor_thread = threading.Thread(target=self.__can_monitor)

    @property
    def panda(self):
        return self.__p

    def stop(self):
        if self.__lin_timer:
            self.__lin_timer.cancel()
            self.__lin_timeout_handler()

        self.__stop = True

    def join(self):
        if self.__lin_monitor_thread.is_alive():
            self.__lin_monitor_thread.join()
        if self.__can_monitor_thread.is_alive():
            self.__can_monitor_thread.join()
        if self.__p:
            print("closing handle")
            self.__p.close()

    def set_enable(self, on):
        self.__on = on

    def start(self):
        self.panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
        if self.__lin_enable:
            self.__lin_monitor_thread.start()
        if self.__can_enable:
            self.__can_monitor_thread.start()

    #########################
    # CAN related functions #
    #########################

    def __can_monitor(self):
        print("STARTING CAN THREAD")
        self.panda.set_can_speed_kbps(0, self.__can_kbaud)
        self.panda.can_recv()  # Toss whatever was already there

        while not self.__stop:
            for address, ts, data, src in self.panda.can_recv():
                if self.__on and src == 0 and len(data) == 8 and data[0] >= 2:
                    if not self.__silent:
                        print("Processing CAN message", src, hex(address), binascii.hexlify(data))
                    self.__can_process_msg(data[1], data[2], address, ts, data, src)
                elif not self.__silent:
                    print("Rejecting CAN message", src, hex(address), binascii.hexlify(data))

    def can_mode_11b(self):
        self.__can11b = True
        self.__can29b = False

    def can_mode_29b(self):
        self.__can11b = False
        self.__can29b = True

    def can_mode_11b_29b(self):
        self.__can11b = True
        self.__can29b = True

    def change_can_baud(self, kbaud):
        self.__can_kbaud = kbaud
        self.panda.set_can_speed_kbps(0, self.__can_kbaud)

    def can_add_extra_noise(self, noise_msg, addr=None):
        self.__can_extra_noise_msgs.append((addr, noise_msg))

    def _can_send(self, addr, msg):
        if not self.__silent:
            print("    CAN Reply (%x)" % addr, binascii.hexlify(msg))
        self.panda.can_send(addr, msg + b'\x00' * (8 - len(msg)), 0)
        if self.__can_extra_noise_msgs:
            noise = self.__can_extra_noise_msgs.popleft()
            self.panda.can_send(noise[0] if noise[0] is not None else addr,
                             noise[1] + b'\x00' * (8 - len(noise[1])), 0)

    def _can_addr_matches(self, addr):
        if self.__can11b and (addr == 0x7DF or (addr & 0x7F8) == 0x7E0):
            return True
        if self.__can29b and (addr == 0x18db33f1 or (addr & 0x1FFF00FF) == 0x18da00f1):
            return True
        return False

    def __can_process_msg(self, mode, pid, address, ts, data, src):
        if not self.__silent:
            print("CAN MSG", binascii.hexlify(data[1:1 + data[0]]),
                  "Addr:", hex(address), "Mode:", hex(mode)[2:].zfill(2),
                  "PID:", hex(pid)[2:].zfill(2), "canLen:", len(data),
                  binascii.hexlify(data))

        if self._can_addr_matches(address) and len(data) == 8:
            outmsg = None
            if data[:3] == b'\x30\x00\x00' and len(self.__can_multipart_data):
                if not self.__silent:
                    print("Request for more data")
                outaddr = 0x7E8 if address == 0x7DF or address == 0x7E0 else 0x18DAF110
                msgnum = 1
                while(self.__can_multipart_data):
                    datalen = min(7, len(self.__can_multipart_data))
                    msgpiece = struct.pack("B", 0x20 | msgnum) + self.__can_multipart_data[:datalen]
                    self._can_send(outaddr, msgpiece)
                    self.__can_multipart_data = self.__can_multipart_data[7:]
                    msgnum = (msgnum + 1) % 0x10
                    time.sleep(0.01)

            else:
                outmsg = self._process_obd(mode, pid)

            if outmsg:
                outaddr = 0x7E8 if address == 0x7DF or address == 0x7E0 else 0x18DAF110

                if len(outmsg) <= 5:
                    self._can_send(outaddr,
                                   struct.pack("BBB", len(outmsg) + 2, 0x40 | data[1], pid) + outmsg)
                else:
                    first_msg_len = min(3, len(outmsg) % 7)
                    payload_len = len(outmsg) + 3
                    msgpiece = struct.pack("BBBBB", 0x10 | ((payload_len >> 8) & 0xF),
                                           payload_len & 0xFF,
                                           0x40 | data[1], pid, 1) + outmsg[:first_msg_len]
                    self._can_send(outaddr, msgpiece)
                    self.__can_multipart_data = outmsg[first_msg_len:]

    #########################
    # General OBD functions #
    #########################

    def _process_obd(self, mode, pid):
        if mode == 0x01:  # Mode: Show current data
            if pid == 0x00:  # List supported things
                return b"\xff\xff\xff\xfe"  # b"\xBE\x1F\xB8\x10" #Bitfield, random features
            elif pid == 0x01:  # Monitor Status since DTC cleared
                return b"\x00\x00\x00\x00"  # Bitfield, random features
            elif pid == 0x04:  # Calculated engine load
                return b"\x2f"
            elif pid == 0x05:  # Engine coolant temperature
                return b"\x3c"
            elif pid == 0x0B:  # Intake manifold absolute pressure
                return b"\x90"
            elif pid == 0x0C:  # Engine RPM
                return b"\x1A\xF8"
            elif pid == 0x0D:  # Vehicle Speed
                return b"\x53"
            elif pid == 0x10:  # MAF air flow rate
                return b"\x01\xA0"
            elif pid == 0x11:  # Throttle Position
                return b"\x90"
            elif pid == 0x33:  # Absolute Barometric Pressure
                return b"\x90"
        elif mode == 0x09:  # Mode: Request vehicle information
            if pid == 0x02:   # Show VIN
                return b"1D4GP00R55B123456"
            if pid == 0xFC:   # test long multi message. Ligned up for LIN responses
                return b''.join(struct.pack(">BBH", 0xAA, 0xAA, num + 1) for num in range(80))
            if pid == 0xFD:   # test long multi message
                parts = (b'\xAA\xAA\xAA' + struct.pack(">I", num) for num in range(80))
                return b'\xAA\xAA\xAA' + b''.join(parts)
            if pid == 0xFE:   # test very long multi message
                parts = (b'\xAA\xAA\xAA' + struct.pack(">I", num) for num in range(584))
                return b'\xAA\xAA\xAA' + b''.join(parts) + b'\xAA'
            if pid == 0xFF:
                return b'\xAA\x00\x00' + \
                       b"".join((b'\xAA' * 5) + struct.pack(">H", num + 1) for num in range(584))
                #return b"\xAA"*100#(0xFFF-3)


if __name__ == "__main__":
    serial = os.getenv("SERIAL") if os.getenv("SERIAL") else None
    kbaud = int(os.getenv("CANKBAUD")) if os.getenv("CANKBAUD") else 500  # type: ignore
    bitwidth = int(os.getenv("CANBITWIDTH")) if os.getenv("CANBITWIDTH") else 0  # type: ignore
    canenable = bool(int(os.getenv("CANENABLE"))) if os.getenv("CANENABLE") else True  # type: ignore
    linenable = bool(int(os.getenv("LINENABLE"))) if os.getenv("LINENABLE") else True  # type: ignore
    sim = ELMCarSimulator(serial, can_kbaud=kbaud, can=canenable, lin=linenable)
    if(bitwidth == 0):
        sim.can_mode_11b_29b()
    if(bitwidth == 11):
        sim.can_mode_11b()
    if(bitwidth == 29):
        sim.can_mode_29b()

    import signal

    def signal_handler(signal, frame):
        print('\nShutting down simulator')
        sim.stop()
        sim.join()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    sim.start()

    signal.pause()
