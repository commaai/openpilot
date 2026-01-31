# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO
from enum import Enum


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class Ubx(KaitaiStruct):

    class GnssType(Enum):
        gps = 0
        sbas = 1
        galileo = 2
        beidou = 3
        imes = 4
        qzss = 5
        glonass = 6
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.magic = self._io.read_bytes(2)
        if not self.magic == b"\xB5\x62":
            raise kaitaistruct.ValidationNotEqualError(b"\xB5\x62", self.magic, self._io, u"/seq/0")
        self.msg_type = self._io.read_u2be()
        self.length = self._io.read_u2le()
        _on = self.msg_type
        if _on == 2569:
            self.body = Ubx.MonHw(self._io, self, self._root)
        elif _on == 533:
            self.body = Ubx.RxmRawx(self._io, self, self._root)
        elif _on == 531:
            self.body = Ubx.RxmSfrbx(self._io, self, self._root)
        elif _on == 309:
            self.body = Ubx.NavSat(self._io, self, self._root)
        elif _on == 2571:
            self.body = Ubx.MonHw2(self._io, self, self._root)
        elif _on == 263:
            self.body = Ubx.NavPvt(self._io, self, self._root)

    class RxmRawx(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.rcv_tow = self._io.read_f8le()
            self.week = self._io.read_u2le()
            self.leap_s = self._io.read_s1()
            self.num_meas = self._io.read_u1()
            self.rec_stat = self._io.read_u1()
            self.reserved1 = self._io.read_bytes(3)
            self._raw_meas = []
            self.meas = []
            for i in range(self.num_meas):
                self._raw_meas.append(self._io.read_bytes(32))
                _io__raw_meas = KaitaiStream(BytesIO(self._raw_meas[i]))
                self.meas.append(Ubx.RxmRawx.Measurement(_io__raw_meas, self, self._root))


        class Measurement(KaitaiStruct):
            def __init__(self, _io, _parent=None, _root=None):
                self._io = _io
                self._parent = _parent
                self._root = _root if _root else self
                self._read()

            def _read(self):
                self.pr_mes = self._io.read_f8le()
                self.cp_mes = self._io.read_f8le()
                self.do_mes = self._io.read_f4le()
                self.gnss_id = KaitaiStream.resolve_enum(Ubx.GnssType, self._io.read_u1())
                self.sv_id = self._io.read_u1()
                self.reserved2 = self._io.read_bytes(1)
                self.freq_id = self._io.read_u1()
                self.lock_time = self._io.read_u2le()
                self.cno = self._io.read_u1()
                self.pr_stdev = self._io.read_u1()
                self.cp_stdev = self._io.read_u1()
                self.do_stdev = self._io.read_u1()
                self.trk_stat = self._io.read_u1()
                self.reserved3 = self._io.read_bytes(1)



    class RxmSfrbx(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.gnss_id = KaitaiStream.resolve_enum(Ubx.GnssType, self._io.read_u1())
            self.sv_id = self._io.read_u1()
            self.reserved1 = self._io.read_bytes(1)
            self.freq_id = self._io.read_u1()
            self.num_words = self._io.read_u1()
            self.reserved2 = self._io.read_bytes(1)
            self.version = self._io.read_u1()
            self.reserved3 = self._io.read_bytes(1)
            self.body = []
            for i in range(self.num_words):
                self.body.append(self._io.read_u4le())



    class NavSat(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.itow = self._io.read_u4le()
            self.version = self._io.read_u1()
            self.num_svs = self._io.read_u1()
            self.reserved = self._io.read_bytes(2)
            self._raw_svs = []
            self.svs = []
            for i in range(self.num_svs):
                self._raw_svs.append(self._io.read_bytes(12))
                _io__raw_svs = KaitaiStream(BytesIO(self._raw_svs[i]))
                self.svs.append(Ubx.NavSat.Nav(_io__raw_svs, self, self._root))


        class Nav(KaitaiStruct):
            def __init__(self, _io, _parent=None, _root=None):
                self._io = _io
                self._parent = _parent
                self._root = _root if _root else self
                self._read()

            def _read(self):
                self.gnss_id = KaitaiStream.resolve_enum(Ubx.GnssType, self._io.read_u1())
                self.sv_id = self._io.read_u1()
                self.cno = self._io.read_u1()
                self.elev = self._io.read_s1()
                self.azim = self._io.read_s2le()
                self.pr_res = self._io.read_s2le()
                self.flags = self._io.read_u4le()



    class NavPvt(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.i_tow = self._io.read_u4le()
            self.year = self._io.read_u2le()
            self.month = self._io.read_u1()
            self.day = self._io.read_u1()
            self.hour = self._io.read_u1()
            self.min = self._io.read_u1()
            self.sec = self._io.read_u1()
            self.valid = self._io.read_u1()
            self.t_acc = self._io.read_u4le()
            self.nano = self._io.read_s4le()
            self.fix_type = self._io.read_u1()
            self.flags = self._io.read_u1()
            self.flags2 = self._io.read_u1()
            self.num_sv = self._io.read_u1()
            self.lon = self._io.read_s4le()
            self.lat = self._io.read_s4le()
            self.height = self._io.read_s4le()
            self.h_msl = self._io.read_s4le()
            self.h_acc = self._io.read_u4le()
            self.v_acc = self._io.read_u4le()
            self.vel_n = self._io.read_s4le()
            self.vel_e = self._io.read_s4le()
            self.vel_d = self._io.read_s4le()
            self.g_speed = self._io.read_s4le()
            self.head_mot = self._io.read_s4le()
            self.s_acc = self._io.read_s4le()
            self.head_acc = self._io.read_u4le()
            self.p_dop = self._io.read_u2le()
            self.flags3 = self._io.read_u1()
            self.reserved1 = self._io.read_bytes(5)
            self.head_veh = self._io.read_s4le()
            self.mag_dec = self._io.read_s2le()
            self.mag_acc = self._io.read_u2le()


    class MonHw2(KaitaiStruct):

        class ConfigSource(Enum):
            flash = 102
            otp = 111
            config_pins = 112
            rom = 113
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.ofs_i = self._io.read_s1()
            self.mag_i = self._io.read_u1()
            self.ofs_q = self._io.read_s1()
            self.mag_q = self._io.read_u1()
            self.cfg_source = KaitaiStream.resolve_enum(Ubx.MonHw2.ConfigSource, self._io.read_u1())
            self.reserved1 = self._io.read_bytes(3)
            self.low_lev_cfg = self._io.read_u4le()
            self.reserved2 = self._io.read_bytes(8)
            self.post_status = self._io.read_u4le()
            self.reserved3 = self._io.read_bytes(4)


    class MonHw(KaitaiStruct):

        class AntennaStatus(Enum):
            init = 0
            dontknow = 1
            ok = 2
            short = 3
            open = 4

        class AntennaPower(Enum):
            false = 0
            true = 1
            dontknow = 2
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.pin_sel = self._io.read_u4le()
            self.pin_bank = self._io.read_u4le()
            self.pin_dir = self._io.read_u4le()
            self.pin_val = self._io.read_u4le()
            self.noise_per_ms = self._io.read_u2le()
            self.agc_cnt = self._io.read_u2le()
            self.a_status = KaitaiStream.resolve_enum(Ubx.MonHw.AntennaStatus, self._io.read_u1())
            self.a_power = KaitaiStream.resolve_enum(Ubx.MonHw.AntennaPower, self._io.read_u1())
            self.flags = self._io.read_u1()
            self.reserved1 = self._io.read_bytes(1)
            self.used_mask = self._io.read_u4le()
            self.vp = self._io.read_bytes(17)
            self.jam_ind = self._io.read_u1()
            self.reserved2 = self._io.read_bytes(2)
            self.pin_irq = self._io.read_u4le()
            self.pull_h = self._io.read_u4le()
            self.pull_l = self._io.read_u4le()


    @property
    def checksum(self):
        if hasattr(self, '_m_checksum'):
            return self._m_checksum

        _pos = self._io.pos()
        self._io.seek((self.length + 6))
        self._m_checksum = self._io.read_u2le()
        self._io.seek(_pos)
        return getattr(self, '_m_checksum', None)


