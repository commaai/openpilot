import os
import numpy as np
import unittest

from kinematic_kf import KinematicKalman, ObservationKind, States  # pylint: disable=import-error

GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'generated'))

class TestKinematic(unittest.TestCase):
  def test_kinematic_kf(self):
    np.random.seed(0)

    kf = KinematicKalman(GENERATED_DIR)

    # Simple simulation
    dt = 0.01
    ts = np.arange(0, 5, step=dt)
    vs = np.sin(ts * 5)

    x = 0.0
    xs = []

    xs_meas = []

    xs_kf = []
    vs_kf = []

    xs_kf_std = []
    vs_kf_std = []

    for t, v in zip(ts, vs):
      xs.append(x)

      # Update kf
      meas = np.random.normal(x, 0.1)
      xs_meas.append(meas)
      kf.predict_and_observe(t, ObservationKind.POSITION, [meas])

      # Retrieve kf values
      state = kf.x
      xs_kf.append(float(state[States.POSITION].item()))
      vs_kf.append(float(state[States.VELOCITY].item()))
      std = np.sqrt(kf.P)
      xs_kf_std.append(float(std[States.POSITION, States.POSITION].item()))
      vs_kf_std.append(float(std[States.VELOCITY, States.VELOCITY].item()))

      # Update simulation
      x += v * dt

    xs, xs_meas, xs_kf, vs_kf, xs_kf_std, vs_kf_std = (np.asarray(a) for a in (xs, xs_meas, xs_kf, vs_kf, xs_kf_std, vs_kf_std))

    self.assertAlmostEqual(xs_kf[-1], -0.010866289677966417)
    self.assertAlmostEqual(xs_kf_std[-1], 0.04477103863330089)
    self.assertAlmostEqual(vs_kf[-1], -0.8553720537261753)
    self.assertAlmostEqual(vs_kf_std[-1], 0.6695762270974388)

    if "PLOT" in os.environ:
      import matplotlib.pyplot as plt  # pylint: disable=import-error
      plt.figure()
      plt.subplot(2, 1, 1)
      plt.plot(ts, xs, 'k', label='Simulation')
      plt.plot(ts, xs_meas, 'k.', label='Measurements')
      plt.plot(ts, xs_kf, label='KF')
      ax = plt.gca()
      ax.fill_between(ts, xs_kf - xs_kf_std, xs_kf + xs_kf_std, alpha=.2, color='C0')

      plt.xlabel("Time [s]")
      plt.ylabel("Position [m]")
      plt.legend()

      plt.subplot(2, 1, 2)
      plt.plot(ts, vs, 'k', label='Simulation')
      plt.plot(ts, vs_kf, label='KF')

      ax = plt.gca()
      ax.fill_between(ts, vs_kf - vs_kf_std, vs_kf + vs_kf_std, alpha=.2, color='C0')

      plt.xlabel("Time [s]")
      plt.ylabel("Velocity [m/s]")
      plt.legend()

      plt.show()


if __name__ == "__main__":
  unittest.main()
