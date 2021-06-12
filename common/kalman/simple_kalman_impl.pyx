# distutils: language = c++
# cython: language_level=3

cdef class KF1D:
  def __init__(self, x0, A, C, K):
    self.x0_0 = x0[0][0]
    self.x1_0 = x0[1][0]
    self.A0_0 = A[0][0]
    self.A0_1 = A[0][1]
    self.A1_0 = A[1][0]
    self.A1_1 = A[1][1]
    self.C0_0 = C[0]
    self.C0_1 = C[1]
    self.K0_0 = K[0][0]
    self.K1_0 = K[1][0]

    self.A_K_0 = self.A0_0 - self.K0_0 * self.C0_0
    self.A_K_1 = self.A0_1 - self.K0_0 * self.C0_1
    self.A_K_2 = self.A1_0 - self.K1_0 * self.C0_0
    self.A_K_3 = self.A1_1 - self.K1_0 * self.C0_1

  def update(self, meas):
    cdef double x0_0 = self.A_K_0 * self.x0_0 + self.A_K_1 * self.x1_0 + self.K0_0 * meas
    cdef double x1_0 = self.A_K_2 * self.x0_0 + self.A_K_3 * self.x1_0 + self.K1_0 * meas
    self.x0_0 = x0_0
    self.x1_0 = x1_0

    return [self.x0_0, self.x1_0]

  @property
  def x(self):
    return [[self.x0_0], [self.x1_0]]

  @x.setter
  def x(self, x):
    self.x0_0 = x[0][0]
    self.x1_0 = x[1][0]
