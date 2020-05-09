class FakeSteeringWheel():
  def __init__(self, dt=0.01):
    # physical params
    self.DAC = 4. / 0.625 # convert torque value from can to Nm
    self.k = 0.035
    self.centering_k = 4.1 * 0.9
    self.b = 0.1 * 0.8
    self.I = 1 * 1.36 * (0.175**2)
    self.dt = dt
    # ...

    self.angle = 0. # start centered
    # self.omega = 0.

  def response(self, torque, vego=0):
    exerted_torque = torque * self.DAC
    # centering_torque = np.clip(-(self.centering_k * self.angle), -1.1, 1.1)
    # damping_torque = -(self.b * self.omega)
    # self.omega += self.dt * (exerted_torque + centering_torque + damping_torque) / self.I
    # self.omega = np.clip(self.omega, -1.1, 1.1)
    # self.angle += self.dt * self.omega
    self.angle += self.dt * self.k * exerted_torque # aristotle

    # print(" ========== ")
    # print("angle,", self.angle)
    # print("omega,", self.omega)
    # print("torque,", exerted_torque)
    # print("centering torque,", centering_torque)
    # print("damping torque,", damping_torque)
    # print(" ========== ")

  def set_angle(self, target):
    self.angle = target
    # self.omega = 0.
