# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class Gps(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.tlm = Gps.Tlm(self._io, self, self._root)
        self.how = Gps.How(self._io, self, self._root)
        _on = self.how.subframe_id
        if _on == 1:
            self.body = Gps.Subframe1(self._io, self, self._root)
        elif _on == 2:
            self.body = Gps.Subframe2(self._io, self, self._root)
        elif _on == 3:
            self.body = Gps.Subframe3(self._io, self, self._root)
        elif _on == 4:
            self.body = Gps.Subframe4(self._io, self, self._root)

    class Subframe1(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.week_no = self._io.read_bits_int_be(10)
            self.code = self._io.read_bits_int_be(2)
            self.sv_accuracy = self._io.read_bits_int_be(4)
            self.sv_health = self._io.read_bits_int_be(6)
            self.iodc_msb = self._io.read_bits_int_be(2)
            self.l2_p_data_flag = self._io.read_bits_int_be(1) != 0
            self.reserved1 = self._io.read_bits_int_be(23)
            self.reserved2 = self._io.read_bits_int_be(24)
            self.reserved3 = self._io.read_bits_int_be(24)
            self.reserved4 = self._io.read_bits_int_be(16)
            self._io.align_to_byte()
            self.t_gd = self._io.read_s1()
            self.iodc_lsb = self._io.read_u1()
            self.t_oc = self._io.read_u2be()
            self.af_2 = self._io.read_s1()
            self.af_1 = self._io.read_s2be()
            self.af_0_sign = self._io.read_bits_int_be(1) != 0
            self.af_0_value = self._io.read_bits_int_be(21)
            self.reserved5 = self._io.read_bits_int_be(2)

        @property
        def af_0(self):
            if hasattr(self, '_m_af_0'):
                return self._m_af_0

            self._m_af_0 = ((self.af_0_value - (1 << 21)) if self.af_0_sign else self.af_0_value)
            return getattr(self, '_m_af_0', None)


    class Subframe3(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.c_ic = self._io.read_s2be()
            self.omega_0 = self._io.read_s4be()
            self.c_is = self._io.read_s2be()
            self.i_0 = self._io.read_s4be()
            self.c_rc = self._io.read_s2be()
            self.omega = self._io.read_s4be()
            self.omega_dot_sign = self._io.read_bits_int_be(1) != 0
            self.omega_dot_value = self._io.read_bits_int_be(23)
            self._io.align_to_byte()
            self.iode = self._io.read_u1()
            self.idot_sign = self._io.read_bits_int_be(1) != 0
            self.idot_value = self._io.read_bits_int_be(13)
            self.reserved = self._io.read_bits_int_be(2)

        @property
        def omega_dot(self):
            if hasattr(self, '_m_omega_dot'):
                return self._m_omega_dot

            self._m_omega_dot = ((self.omega_dot_value - (1 << 23)) if self.omega_dot_sign else self.omega_dot_value)
            return getattr(self, '_m_omega_dot', None)

        @property
        def idot(self):
            if hasattr(self, '_m_idot'):
                return self._m_idot

            self._m_idot = ((self.idot_value - (1 << 13)) if self.idot_sign else self.idot_value)
            return getattr(self, '_m_idot', None)


    class Subframe4(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.data_id = self._io.read_bits_int_be(2)
            self.page_id = self._io.read_bits_int_be(6)
            self._io.align_to_byte()
            _on = self.page_id
            if _on == 56:
                self.body = Gps.Subframe4.IonosphereData(self._io, self, self._root)

        class IonosphereData(KaitaiStruct):
            def __init__(self, _io, _parent=None, _root=None):
                self._io = _io
                self._parent = _parent
                self._root = _root if _root else self
                self._read()

            def _read(self):
                self.a0 = self._io.read_s1()
                self.a1 = self._io.read_s1()
                self.a2 = self._io.read_s1()
                self.a3 = self._io.read_s1()
                self.b0 = self._io.read_s1()
                self.b1 = self._io.read_s1()
                self.b2 = self._io.read_s1()
                self.b3 = self._io.read_s1()



    class How(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.tow_count = self._io.read_bits_int_be(17)
            self.alert = self._io.read_bits_int_be(1) != 0
            self.anti_spoof = self._io.read_bits_int_be(1) != 0
            self.subframe_id = self._io.read_bits_int_be(3)
            self.reserved = self._io.read_bits_int_be(2)


    class Tlm(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.preamble = self._io.read_bytes(1)
            if not self.preamble == b"\x8B":
                raise kaitaistruct.ValidationNotEqualError(b"\x8B", self.preamble, self._io, u"/types/tlm/seq/0")
            self.tlm = self._io.read_bits_int_be(14)
            self.integrity_status = self._io.read_bits_int_be(1) != 0
            self.reserved = self._io.read_bits_int_be(1) != 0


    class Subframe2(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.iode = self._io.read_u1()
            self.c_rs = self._io.read_s2be()
            self.delta_n = self._io.read_s2be()
            self.m_0 = self._io.read_s4be()
            self.c_uc = self._io.read_s2be()
            self.e = self._io.read_s4be()
            self.c_us = self._io.read_s2be()
            self.sqrt_a = self._io.read_u4be()
            self.t_oe = self._io.read_u2be()
            self.fit_interval_flag = self._io.read_bits_int_be(1) != 0
            self.aoda = self._io.read_bits_int_be(5)
            self.reserved = self._io.read_bits_int_be(2)



