# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class Glonass(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.idle_chip = self._io.read_bits_int_be(1) != 0
        self.string_number = self._io.read_bits_int_be(4)
        # workaround for kaitai bit alignment issue (see glonass_fix.patch for C++)
        # self._io.align_to_byte()
        _on = self.string_number
        if _on == 4:
            self.data = Glonass.String4(self._io, self, self._root)
        elif _on == 1:
            self.data = Glonass.String1(self._io, self, self._root)
        elif _on == 3:
            self.data = Glonass.String3(self._io, self, self._root)
        elif _on == 5:
            self.data = Glonass.String5(self._io, self, self._root)
        elif _on == 2:
            self.data = Glonass.String2(self._io, self, self._root)
        else:
            self.data = Glonass.StringNonImmediate(self._io, self, self._root)
        self.hamming_code = self._io.read_bits_int_be(8)
        self.pad_1 = self._io.read_bits_int_be(11)
        self.superframe_number = self._io.read_bits_int_be(16)
        self.pad_2 = self._io.read_bits_int_be(8)
        self.frame_number = self._io.read_bits_int_be(8)

    class String4(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.tau_n_sign = self._io.read_bits_int_be(1) != 0
            self.tau_n_value = self._io.read_bits_int_be(21)
            self.delta_tau_n_sign = self._io.read_bits_int_be(1) != 0
            self.delta_tau_n_value = self._io.read_bits_int_be(4)
            self.e_n = self._io.read_bits_int_be(5)
            self.not_used_1 = self._io.read_bits_int_be(14)
            self.p4 = self._io.read_bits_int_be(1) != 0
            self.f_t = self._io.read_bits_int_be(4)
            self.not_used_2 = self._io.read_bits_int_be(3)
            self.n_t = self._io.read_bits_int_be(11)
            self.n = self._io.read_bits_int_be(5)
            self.m = self._io.read_bits_int_be(2)

        @property
        def tau_n(self):
            if hasattr(self, '_m_tau_n'):
                return self._m_tau_n

            self._m_tau_n = ((self.tau_n_value * -1) if self.tau_n_sign else self.tau_n_value)
            return getattr(self, '_m_tau_n', None)

        @property
        def delta_tau_n(self):
            if hasattr(self, '_m_delta_tau_n'):
                return self._m_delta_tau_n

            self._m_delta_tau_n = ((self.delta_tau_n_value * -1) if self.delta_tau_n_sign else self.delta_tau_n_value)
            return getattr(self, '_m_delta_tau_n', None)


    class StringNonImmediate(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.data_1 = self._io.read_bits_int_be(64)
            self.data_2 = self._io.read_bits_int_be(8)


    class String5(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.n_a = self._io.read_bits_int_be(11)
            self.tau_c = self._io.read_bits_int_be(32)
            self.not_used = self._io.read_bits_int_be(1) != 0
            self.n_4 = self._io.read_bits_int_be(5)
            self.tau_gps = self._io.read_bits_int_be(22)
            self.l_n = self._io.read_bits_int_be(1) != 0


    class String1(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.not_used = self._io.read_bits_int_be(2)
            self.p1 = self._io.read_bits_int_be(2)
            self.t_k = self._io.read_bits_int_be(12)
            self.x_vel_sign = self._io.read_bits_int_be(1) != 0
            self.x_vel_value = self._io.read_bits_int_be(23)
            self.x_accel_sign = self._io.read_bits_int_be(1) != 0
            self.x_accel_value = self._io.read_bits_int_be(4)
            self.x_sign = self._io.read_bits_int_be(1) != 0
            self.x_value = self._io.read_bits_int_be(26)

        @property
        def x_vel(self):
            if hasattr(self, '_m_x_vel'):
                return self._m_x_vel

            self._m_x_vel = ((self.x_vel_value * -1) if self.x_vel_sign else self.x_vel_value)
            return getattr(self, '_m_x_vel', None)

        @property
        def x_accel(self):
            if hasattr(self, '_m_x_accel'):
                return self._m_x_accel

            self._m_x_accel = ((self.x_accel_value * -1) if self.x_accel_sign else self.x_accel_value)
            return getattr(self, '_m_x_accel', None)

        @property
        def x(self):
            if hasattr(self, '_m_x'):
                return self._m_x

            self._m_x = ((self.x_value * -1) if self.x_sign else self.x_value)
            return getattr(self, '_m_x', None)


    class String2(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.b_n = self._io.read_bits_int_be(3)
            self.p2 = self._io.read_bits_int_be(1) != 0
            self.t_b = self._io.read_bits_int_be(7)
            self.not_used = self._io.read_bits_int_be(5)
            self.y_vel_sign = self._io.read_bits_int_be(1) != 0
            self.y_vel_value = self._io.read_bits_int_be(23)
            self.y_accel_sign = self._io.read_bits_int_be(1) != 0
            self.y_accel_value = self._io.read_bits_int_be(4)
            self.y_sign = self._io.read_bits_int_be(1) != 0
            self.y_value = self._io.read_bits_int_be(26)

        @property
        def y_vel(self):
            if hasattr(self, '_m_y_vel'):
                return self._m_y_vel

            self._m_y_vel = ((self.y_vel_value * -1) if self.y_vel_sign else self.y_vel_value)
            return getattr(self, '_m_y_vel', None)

        @property
        def y_accel(self):
            if hasattr(self, '_m_y_accel'):
                return self._m_y_accel

            self._m_y_accel = ((self.y_accel_value * -1) if self.y_accel_sign else self.y_accel_value)
            return getattr(self, '_m_y_accel', None)

        @property
        def y(self):
            if hasattr(self, '_m_y'):
                return self._m_y

            self._m_y = ((self.y_value * -1) if self.y_sign else self.y_value)
            return getattr(self, '_m_y', None)


    class String3(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.p3 = self._io.read_bits_int_be(1) != 0
            self.gamma_n_sign = self._io.read_bits_int_be(1) != 0
            self.gamma_n_value = self._io.read_bits_int_be(10)
            self.not_used = self._io.read_bits_int_be(1) != 0
            self.p = self._io.read_bits_int_be(2)
            self.l_n = self._io.read_bits_int_be(1) != 0
            self.z_vel_sign = self._io.read_bits_int_be(1) != 0
            self.z_vel_value = self._io.read_bits_int_be(23)
            self.z_accel_sign = self._io.read_bits_int_be(1) != 0
            self.z_accel_value = self._io.read_bits_int_be(4)
            self.z_sign = self._io.read_bits_int_be(1) != 0
            self.z_value = self._io.read_bits_int_be(26)

        @property
        def gamma_n(self):
            if hasattr(self, '_m_gamma_n'):
                return self._m_gamma_n

            self._m_gamma_n = ((self.gamma_n_value * -1) if self.gamma_n_sign else self.gamma_n_value)
            return getattr(self, '_m_gamma_n', None)

        @property
        def z_vel(self):
            if hasattr(self, '_m_z_vel'):
                return self._m_z_vel

            self._m_z_vel = ((self.z_vel_value * -1) if self.z_vel_sign else self.z_vel_value)
            return getattr(self, '_m_z_vel', None)

        @property
        def z_accel(self):
            if hasattr(self, '_m_z_accel'):
                return self._m_z_accel

            self._m_z_accel = ((self.z_accel_value * -1) if self.z_accel_sign else self.z_accel_value)
            return getattr(self, '_m_z_accel', None)

        @property
        def z(self):
            if hasattr(self, '_m_z'):
                return self._m_z

            self._m_z = ((self.z_value * -1) if self.z_sign else self.z_value)
            return getattr(self, '_m_z', None)


