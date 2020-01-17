import math
from common.numpy_fast import clip

MAX_ANGLE_OFFSET = math.radians(10.)
MAX_ANGLE_OFFSET_TH = math.radians(9.)
MIN_STIFFNESS = 0.5
MAX_STIFFNESS = 2.0
MIN_SR = 0.5
MAX_SR = 2.0
MIN_SR_TH = 0.55
MAX_SR_TH = 1.9

DEBUG = False


class ParamsLearner():
  def __init__(self, VM, angle_offset=0., stiffness_factor=1.0, steer_ratio=None, learning_rate=1.0):
    self.VM = VM

    self.ao = math.radians(angle_offset)
    self.slow_ao = math.radians(angle_offset)
    self.x = stiffness_factor
    self.sR = VM.sR if steer_ratio is None else steer_ratio
    self.MIN_SR = MIN_SR * self.VM.sR
    self.MAX_SR = MAX_SR * self.VM.sR
    self.MIN_SR_TH = MIN_SR_TH * self.VM.sR
    self.MAX_SR_TH = MAX_SR_TH * self.VM.sR

    self.alpha1 = 0.01 * learning_rate
    self.alpha2 = 0.0005 * learning_rate
    self.alpha3 = 0.1 * learning_rate
    self.alpha4 = 1.0 * learning_rate

  def get_values(self):
    return {
      'angleOffsetAverage': math.degrees(self.slow_ao),
      'stiffnessFactor': self.x,
      'steerRatio': self.sR,
    }

  def update(self, psi, u, sa):
    cF0 = self.VM.cF
    cR0 = self.VM.cR
    aR = self.VM.aR
    aF = self.VM.aF
    l = self.VM.l
    m = self.VM.m

    x = self.x
    ao = self.ao
    sR = self.sR

    # Gradient descent:  learn angle offset, tire stiffness and steer ratio.
    if u > 10.0 and abs(math.degrees(sa)) < 15.:
      self.ao -= self.alpha1 * 2.0*cF0*cR0*l*u*x*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0)))/(sR**2*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0))**2)

      ao = self.slow_ao
      self.slow_ao -= self.alpha2 * 2.0*cF0*cR0*l*u*x*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0)))/(sR**2*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0))**2)

      self.x -= self.alpha3 * -2.0*cF0*cR0*l*m*u**3*(ao - sa)*(aF*cF0 - aR*cR0)*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0)))/(sR**2*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0))**3)

      self.sR -= self.alpha4 * -2.0*cF0*cR0*l*u*x*(ao - sa)*(1.0*cF0*cR0*l*u*x*(ao - sa) + psi*sR*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0)))/(sR**3*(cF0*cR0*l**2*x - m*u**2*(aF*cF0 - aR*cR0))**2)

    if DEBUG:
      # s1 = "Measured yaw rate % .6f" % psi
      # ao = 0.
      # s2 = "Uncompensated yaw % .6f" % (1.0*u*(-ao + sa)/(l*sR*(1 - m*u**2*(aF*cF0*x - aR*cR0*x)/(cF0*cR0*l**2*x**2))))
      # instant_ao = aF*m*psi*sR*u/(cR0*l*x) - aR*m*psi*sR*u/(cF0*l*x) - l*psi*sR/u + sa
      s4 = "Instant AO: % .6f Avg. AO % .6f" % (math.degrees(self.ao), math.degrees(self.slow_ao))
      s5 = "Stiffnes: % .6f x" % self.x
      s6 = "sR: %.4f" % self.sR
      print("{0} {1} {2}".format(s4, s5, s6))


    self.ao = clip(self.ao, -MAX_ANGLE_OFFSET, MAX_ANGLE_OFFSET)
    self.slow_ao = clip(self.slow_ao, -MAX_ANGLE_OFFSET, MAX_ANGLE_OFFSET)
    self.x = clip(self.x, MIN_STIFFNESS, MAX_STIFFNESS)
    self.sR = clip(self.sR, self.MIN_SR, self.MAX_SR)

    # don't check stiffness for validity, as it can change quickly if sR is off
    valid = abs(self.slow_ao) < MAX_ANGLE_OFFSET_TH and \
      self.sR > self.MIN_SR_TH and self.sR < self.MAX_SR_TH

    return valid
