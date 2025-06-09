import numpy as np
from types import SimpleNamespace

from openpilot.selfdrive.locationd.torqued import TorqueEstimator

class CPStub:
    def __init__(self):
        self.carFingerprint = "stub"
        self.brand = "toyota"
        self.lateralTuning = SimpleNamespace()
        self.lateralTuning.torque = SimpleNamespace(friction=0.0, latAccelFactor=1.0)
        self.lateralTuning.which = lambda: "torque"

def test_calPerc_progress():
    est = TorqueEstimator(CPStub(), decimated=True)
    msg = est.get_msg()
    assert msg.liveTorqueParameters.calPerc == 0

    for (low, high), req in zip(est.filtered_points.buckets.keys(), est.filtered_points.buckets_min_points.values()):
        for _ in range(int(req)):
            est.filtered_points.add_point((low + high) / 2.0, 0.0)

    msg = est.get_msg()
    assert msg.liveTorqueParameters.calPerc == 100
