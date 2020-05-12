# coding: utf-8
# vim: set ts=4 sw=4 et:
""" This file contains some utils for connecting to Logentries
    as well as storing logs in a queue and sending them."""

VERSION = '2.0.7'

from logentries import helpers as le_helpers

import logging
import threading
import socket
import random
import time
import sys

import certifi


# Size of the internal event queue
QUEUE_SIZE = 32768
# Logentries API server address
LE_API_DEFAULT = "data.logentries.com"
# Port number for token logging to Logentries API server
LE_PORT_DEFAULT = 80
LE_TLS_PORT_DEFAULT = 443
# Minimal delay between attempts to reconnect in seconds
MIN_DELAY = 0.1
# Maximal delay between attempts to recconect in seconds
MAX_DELAY = 10
# Unicode Line separator character   \u2028
LINE_SEP = le_helpers.to_unicode('\u2028')


# LE appender signature - used for debugging messages
LE = "LE: "
# Error message displayed when an incorrect Token has been detected
INVALID_TOKEN = ("\n\nIt appears the LOGENTRIES_TOKEN "
                 "parameter you entered is incorrect!\n\n")


def dbg(msg):
    print(LE + msg)


class PlainTextSocketAppender(threading.Thread):
    def __init__(self, verbose=True, le_api=LE_API_DEFAULT, le_port=LE_PORT_DEFAULT, le_tls_port=LE_TLS_PORT_DEFAULT):
        threading.Thread.__init__(self)

        # Logentries API server address
        self.le_api = le_api

        # Port number for token logging to Logentries API server
        self.le_port = le_port
        self.le_tls_port = le_tls_port

        self.daemon = True
        self.verbose = verbose
        self._conn = None
        self._queue = le_helpers.create_queue(QUEUE_SIZE)

    def empty(self):
        return self._queue.empty()

    def open_connection(self):
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn.connect((self.le_api, self.le_port))

    def reopen_connection(self):
        self.close_connection()

        root_delay = MIN_DELAY
        while True:
            try:
                self.open_connection()
                return
            except Exception:
                if self.verbose:
                    dbg("Unable to connect to Logentries")

            root_delay *= 2
            if(root_delay > MAX_DELAY):
                root_delay = MAX_DELAY

            wait_for = root_delay + random.uniform(0, root_delay)

            try:
                time.sleep(wait_for)
            except KeyboardInterrupt:
                raise

    def close_connection(self):
        if self._conn is not None:
            self._conn.close()

    def run(self):
        try:
            # Open connection
            self.reopen_connection()

            # Send data in queue
            while True:
                # Take data from queue
                data = self._queue.get(block=True)

                # Replace newlines with Unicode line separator
                # for multi-line events
                if not le_helpers.is_unicode(data):
                    multiline = le_helpers.create_unicode(data).replace(
                        '\n', LINE_SEP)
                else:
                    multiline = data.replace('\n', LINE_SEP)
                multiline += "\n"
                # Send data, reconnect if needed
                while True:
                    try:
                        self._conn.send(multiline.encode('utf-8'))
                    except socket.error:
                        self.reopen_connection()
                        continue
                    break
        except KeyboardInterrupt:
            if self.verbose:
                dbg("Logentries asynchronous socket client interrupted")

        self.close_connection()

SocketAppender = PlainTextSocketAppender

try:
    import ssl
    ssl_enabled = True
except ImportError:  # for systems without TLS support.
    ssl_enabled = False
    dbg("Unable to import ssl module. Will send over port 80.")
else:
    class TLSSocketAppender(PlainTextSocketAppender):

        def open_connection(self):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock = ssl.wrap_socket(
                sock=sock,
                keyfile=None,
                certfile=None,
                server_side=False,
                cert_reqs=ssl.CERT_REQUIRED,
                ssl_version=getattr(
                    ssl,
                    'PROTOCOL_TLSv1_2',
                    ssl.PROTOCOL_TLSv1
                ),
                ca_certs=certifi.where(),
                do_handshake_on_connect=True,
                suppress_ragged_eofs=True,
            )

            sock.connect((self.le_api, self.le_tls_port))
            self._conn = sock


class LogentriesHandler(logging.Handler):
    def __init__(self, token, use_tls=True, verbose=True, format=None, le_api=LE_API_DEFAULT, le_port=LE_PORT_DEFAULT, le_tls_port=LE_TLS_PORT_DEFAULT):
        logging.Handler.__init__(self)
        self.token = token
        self.good_config = True
        self.verbose = verbose
        # give the socket 10 seconds to flush,
        # otherwise drop logs
        self.timeout = 10
        if not le_helpers.check_token(token):
            if self.verbose:
                dbg(INVALID_TOKEN)
            self.good_config = False
        if format is None:
            format = logging.Formatter('%(asctime)s : %(levelname)s, %(message)s',
                                       '%a %b %d %H:%M:%S %Z %Y')
        self.setFormatter(format)
        self.setLevel(logging.DEBUG)
        if use_tls and ssl_enabled:
            self._thread = TLSSocketAppender(verbose=verbose, le_api=le_api, le_port=le_port, le_tls_port=le_tls_port)
        else:
            self._thread = SocketAppender(verbose=verbose, le_api=le_api, le_port=le_port, le_tls_port=le_tls_port)

    def flush(self):
        # wait for all queued logs to be send
        now = time.time()
        while not self._thread.empty():
            time.sleep(0.2)
            if time.time() - now > self.timeout:
                break

    def emit_raw(self, msg):
        if self.good_config and not self._thread.is_alive():
            try:
                self._thread.start()
                if self.verbose:
                    dbg("Starting Logentries Asynchronous Socket Appender")
            except RuntimeError: # It's already started.
                pass

        msg = self.token + msg
        try:
            self._thread._queue.put_nowait(msg)
        except Exception:
            # Queue is full, try to remove the oldest message and put again
            try:
                self._thread._queue.get_nowait()
                self._thread._queue.put_nowait(msg)
            except Exception:
                # Race condition, no need for any action here
                pass

    def emit(self, record):
        msg = self.format(record).rstrip('\n')
        self.emit_raw(msg)

    def close(self):
        logging.Handler.close(self)
