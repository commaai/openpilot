#!/usr/bin/env python3

import socket
import threading
import select

class Reader(threading.Thread):
    def __init__(self, s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._s = s
        self.__stop = False

    def stop(self):
        self.__stop = True

    def run(self):
        while not self.__stop:
            s.recv(1000)

def read_or_fail(s):
    ready = select.select([s], [], [], 4)
    assert ready[0], "Socket did not receive data within the timeout duration."
    return s.recv(1000)

def send_msg(s, msg):
    s.send(msg)
    res = b''
    while not res.endswith(">"):
        res += read_or_fail(s)
    return res

if __name__ == "__main__":
    s = socket.create_connection(("192.168.0.10", 35000))
    send_msg(s, b"ATZ\r")
    send_msg(s, b"ATL1\r")
    print(send_msg(s, b"ATE0\r"))
    print(send_msg(s, b"ATS0\r"))
    print(send_msg(s, b"ATSP6\r"))

    print("\nLOOP\n")

    while True:
        print(send_msg(s, b"0100\r"))
        print(send_msg(s, b"010d\r"))
