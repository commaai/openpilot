import multiprocessing
import numbers
import os
import random
import time
from threading import Timer

import capnp
import pytest
from parameterized import parameterized

import cereal.messaging as messaging
from cereal import car, log
from cereal.services import SERVICE_LIST


# Lazy imports for performance
def _get_capnp():
    return capnp


def _get_multiprocessing():
    return multiprocessing


def _get_numbers():
    return numbers


events = [evt for evt in log.Event.schema.union_fields if evt in SERVICE_LIST.keys()]


def random_sock():
    return random.choice(events)


def random_socks(num_socks=10):
    return list({random_sock() for _ in range(num_socks)})


def random_bytes(length=1000):
    return bytes([random.randrange(0xFF) for _ in range(length)])


def zmq_sleep(t=1):
    if "ZMQ" in os.environ:
        time.sleep(t)


# TODO: this should take any capnp struct and returrn a msg with random populated data
def random_carstate():
    fields = ["vEgo", "aEgo", "brake", "steeringAngleDeg"]
    msg = messaging.new_message("carState")
    cs = msg.carState
    for f in fields:
        setattr(cs, f, random.random() * 10)
    return msg


# TODO: this should compare any capnp structs
def assert_carstate(cs1, cs2):
    numbers = _get_numbers()
    for f in car.CarState.schema.non_union_fields:
        # TODO: check all types
        val1, val2 = getattr(cs1, f), getattr(cs2, f)
        if isinstance(val1, numbers.Number):
            assert val1 == val2, f"{f}: sent '{val1}' vs recvd '{val2}'"


def delayed_send(delay, sock, dat):
    def send_func():
        sock.send(dat)

    Timer(delay, send_func).start()


class TestMessaging:
    def setUp(self):
        # TODO: ZMQ tests are too slow; all sleeps will need to be
        # replaced with logic to block on the necessary condition
        if "ZMQ" in os.environ:
            pytest.skip()

        # ZMQ pub socket takes too long to die
        # sleep to prevent multiple publishers error between tests
        zmq_sleep()

    @parameterized.expand(events)
    def test_new_message(self, evt):
        capnp = _get_capnp()
        try:
            msg = messaging.new_message(evt)
        except capnp.lib.capnp.KjException:
            msg = messaging.new_message(evt, random.randrange(200))
        assert (time.monotonic() - msg.logMonoTime) < 0.1
        assert not msg.valid
        assert evt == msg.which()

    @parameterized.expand(events)
    def test_pub_sock(self, evt):
        messaging.pub_sock(evt)

    @parameterized.expand(events)
    def test_sub_sock(self, evt):
        messaging.sub_sock(evt)

    def test_drain_sock(self):
        capnp = _get_capnp()
        test_cases = [
            (messaging.drain_sock, capnp._DynamicStructReader),
            (messaging.drain_sock_raw, bytes),
        ]

        for func, expected_type in test_cases:
            self._test_drain_sock_impl(func, expected_type)

    def _test_drain_sock_impl(self, func, expected_type):
        sock = "carState"
        pub_sock = messaging.pub_sock(sock)
        sub_sock = messaging.sub_sock(sock, timeout=1000)
        zmq_sleep()

        # no wait and no msgs in queue
        msgs = func(sub_sock)
        assert isinstance(msgs, list)
        assert len(msgs) == 0

        # no wait but msgs are queued up
        num_msgs = random.randrange(3, 10)
        for _ in range(num_msgs):
            pub_sock.send(messaging.new_message(sock).to_bytes())
        time.sleep(0.1)
        msgs = func(sub_sock)
        assert isinstance(msgs, list)
        assert all(isinstance(msg, expected_type) for msg in msgs)
        assert len(msgs) == num_msgs

    def test_recv_sock(self):
        capnp = _get_capnp()
        sock = "carState"
        pub_sock = messaging.pub_sock(sock)
        sub_sock = messaging.sub_sock(sock, timeout=100)
        zmq_sleep()

        # no wait and no msg in queue, socket should timeout
        recvd = messaging.recv_sock(sub_sock)
        assert recvd is None

        # no wait and one msg in queue
        msg = random_carstate()
        pub_sock.send(msg.to_bytes())
        time.sleep(0.01)
        recvd = messaging.recv_sock(sub_sock)
        assert isinstance(recvd, capnp._DynamicStructReader)
        # https://github.com/python/mypy/issues/13038
        assert_carstate(msg.carState, recvd.carState)

    def test_recv_one(self):
        capnp = _get_capnp()
        sock = "carState"
        pub_sock = messaging.pub_sock(sock)
        sub_sock = messaging.sub_sock(sock, timeout=1000)
        zmq_sleep()

        # no msg in queue, socket should timeout
        recvd = messaging.recv_one(sub_sock)
        assert recvd is None

        # one msg in queue
        msg = random_carstate()
        pub_sock.send(msg.to_bytes())
        recvd = messaging.recv_one(sub_sock)
        assert isinstance(recvd, capnp._DynamicStructReader)
        assert_carstate(msg.carState, recvd.carState)

    @pytest.mark.xfail(condition="ZMQ" in os.environ, reason="ZMQ detected")
    def test_recv_one_or_none(self):
        capnp = _get_capnp()
        sock = "carState"
        pub_sock = messaging.pub_sock(sock)
        sub_sock = messaging.sub_sock(sock)
        zmq_sleep()

        # no msg in queue, socket shouldn't block
        recvd = messaging.recv_one_or_none(sub_sock)
        assert recvd is None

        # one msg in queue
        msg = random_carstate()
        pub_sock.send(msg.to_bytes())
        recvd = messaging.recv_one_or_none(sub_sock)
        assert isinstance(recvd, capnp._DynamicStructReader)
        assert_carstate(msg.carState, recvd.carState)

    def test_recv_one_retry(self):
        capnp = _get_capnp()
        sock = "carState"
        sock_timeout = 0.1
        pub_sock = messaging.pub_sock(sock)
        sub_sock = messaging.sub_sock(sock, timeout=round(sock_timeout * 1000))
        zmq_sleep()

        # this test doesn't work with ZMQ since multiprocessing interrupts it
        # Also skip when using mocks since multiprocessing can't access our mocks
        if "ZMQ" not in os.environ and "PYTEST_RUNNING" not in os.environ:
            # get the multiprocessing module
            multiprocessing = _get_multiprocessing()
            # wait 5 socket timeouts and make sure it's still retrying
            p = multiprocessing.Process(target=messaging.recv_one_retry, args=(sub_sock,))
            p.start()
            time.sleep(sock_timeout * 5)
            assert p.is_alive()
            p.terminate()
        
        # wait 5 socket timeouts before sending
        msg = random_carstate()
        delayed_send(sock_timeout * 5, pub_sock, msg.to_bytes())
        start_time = time.monotonic()
        recvd = messaging.recv_one_retry(sub_sock)
        assert (time.monotonic() - start_time) >= sock_timeout * 5
        assert isinstance(recvd, capnp._DynamicStructReader)
        assert_carstate(msg.carState, recvd.carState)
