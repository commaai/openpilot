import pytest
import numpy as np
from opendbc.car.common.pid import PIDController

@pytest.fixture
def pid():
    return PIDController(k_p=1.0, k_i=0.5, k_d=0.1, k_f=0.0, pos_limit=10.0, neg_limit=-10.0, rate=100)

def test_initialization_scalar():
    pid = PIDController(k_p=2.0, k_i=1.0, k_d=0.5)
    assert isinstance(pid._k_p, list)
    assert isinstance(pid._k_i, list)
    assert isinstance(pid._k_d, list)
    assert pid.k_p == 2.0
    assert pid.k_i == 1.0
    assert pid.k_d == 0.5

def test_initialization_list():
    pid = PIDController(k_p=([0], [2.0]), k_i=([0], [1.0]), k_d=([0], [0.5]))
    assert isinstance(pid._k_p, tuple) or isinstance(pid._k_p, list)

def test_reset(pid):
    pid.update(5)
    pid.reset()
    assert pid.p == 0.0
    assert pid.i == 0.0
    assert pid.d == 0.0
    assert pid.f == 0.0
    assert pid.control == 0

def test_update_basic(pid):
    output = pid.update(error=2.0, error_rate=0.5, speed=0.0)
    assert isinstance(output, float)
    assert -10.0 <= output <= 10.0

def test_update_override(pid):
    pid.i = 5.0
    _ = pid.update(error=0.0, error_rate=0.0, speed=0.0, override=True)
    assert abs(pid.i) < 5.0

def test_update_freeze_integrator(pid):
    pid.i = 5.0
    _ = pid.update(error=2.0, error_rate=0.5, speed=0.0, freeze_integrator=True)
    assert pid.i == 5.0

def test_control_clipping(pid):
    output = pid.update(error=1e6)
    assert output <= pid.pos_limit
    output = pid.update(error=-1e6)
    assert output >= pid.neg_limit

def test_error_integral_property(pid):
    pid.i = 1.0
    result = pid.error_integral
    assert np.isclose(result, pid.i / pid.k_i)

def test_feedforward_gain():
    pid = PIDController(k_p=0.0, k_i=0.0, k_d=0.0, k_f=2.0)
    output = pid.update(error=0.0, feedforward=3.0)
    assert output == 6.0

def test_speed_interpolation():
    pid = PIDController(k_p=([0, 10], [1.0, 5.0]), k_i=1.0, k_d=0.1)
    output_low_speed = pid.update(error=1.0, speed=0)
    output_high_speed = pid.update(error=1.0, speed=10)
    assert output_high_speed > output_low_speed
