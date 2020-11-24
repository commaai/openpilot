# cython: language_level = 3

cdef class KF1D:
  cdef public:
    double x0_0
    double x1_0
    double K0_0
    double K1_0
    double A0_0
    double A0_1
    double A1_0
    double A1_1
    double C0_0
    double C0_1
    double A_K_0
    double A_K_1
    double A_K_2
    double A_K_3
