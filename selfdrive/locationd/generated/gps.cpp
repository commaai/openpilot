// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "gps.h"
#include "kaitai/exceptions.h"

gps_t::gps_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent, gps_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = this;
    m_tlm = 0;
    m_how = 0;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void gps_t::_read() {
    m_tlm = new tlm_t(m__io, this, m__root);
    m_how = new how_t(m__io, this, m__root);
    n_body = true;
    switch (how()->subframe_id()) {
    case 1: {
        n_body = false;
        m_body = new subframe_1_t(m__io, this, m__root);
        break;
    }
    case 2: {
        n_body = false;
        m_body = new subframe_2_t(m__io, this, m__root);
        break;
    }
    case 3: {
        n_body = false;
        m_body = new subframe_3_t(m__io, this, m__root);
        break;
    }
    }
}

gps_t::~gps_t() {
    _clean_up();
}

void gps_t::_clean_up() {
    if (m_tlm) {
        delete m_tlm; m_tlm = 0;
    }
    if (m_how) {
        delete m_how; m_how = 0;
    }
    if (!n_body) {
        if (m_body) {
            delete m_body; m_body = 0;
        }
    }
}

gps_t::subframe_1_t::subframe_1_t(kaitai::kstream* p__io, gps_t* p__parent, gps_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void gps_t::subframe_1_t::_read() {
    m_week_no = m__io->read_bits_int_be(10);
    m_code = m__io->read_bits_int_be(2);
    m_sv_accuracy = m__io->read_bits_int_be(4);
    m_sv_health = m__io->read_bits_int_be(6);
    m_iodc_msb = m__io->read_bits_int_be(2);
    m_l2_p_data_flag = m__io->read_bits_int_be(1);
    m_reserved1 = m__io->read_bits_int_be(23);
    m_reserved2 = m__io->read_bits_int_be(24);
    m_reserved3 = m__io->read_bits_int_be(24);
    m_reserved4 = m__io->read_bits_int_be(16);
    m__io->align_to_byte();
    m_t_gd = m__io->read_s1();
    m_iodc_lsb = m__io->read_u1();
    m_t_oc = m__io->read_u2be();
    m_af_2 = m__io->read_s1();
    m_af_1 = m__io->read_s2be();
    m_af_0 = m__io->read_bits_int_be(22);
    m_reserved5 = m__io->read_bits_int_be(2);
}

gps_t::subframe_1_t::~subframe_1_t() {
    _clean_up();
}

void gps_t::subframe_1_t::_clean_up() {
}

gps_t::subframe_3_t::subframe_3_t(kaitai::kstream* p__io, gps_t* p__parent, gps_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void gps_t::subframe_3_t::_read() {
    m_c_ic = m__io->read_s2be();
    m_omega_0 = m__io->read_s4be();
    m_c_is = m__io->read_s2be();
    m_i_0 = m__io->read_s4be();
    m_c_rc = m__io->read_s2be();
    m_omega = m__io->read_s4be();
    m_omega_dot = m__io->read_bits_int_be(24);
    m__io->align_to_byte();
    m_iode = m__io->read_u1();
    m_idot = m__io->read_bits_int_be(14);
    m_reserved = m__io->read_bits_int_be(2);
}

gps_t::subframe_3_t::~subframe_3_t() {
    _clean_up();
}

void gps_t::subframe_3_t::_clean_up() {
}

gps_t::how_t::how_t(kaitai::kstream* p__io, gps_t* p__parent, gps_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void gps_t::how_t::_read() {
    m_tow_count = m__io->read_bits_int_be(17);
    m_alert = m__io->read_bits_int_be(1);
    m_anti_spoof = m__io->read_bits_int_be(1);
    m_subframe_id = m__io->read_bits_int_be(3);
    m_reserved = m__io->read_bits_int_be(2);
}

gps_t::how_t::~how_t() {
    _clean_up();
}

void gps_t::how_t::_clean_up() {
}

gps_t::tlm_t::tlm_t(kaitai::kstream* p__io, gps_t* p__parent, gps_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void gps_t::tlm_t::_read() {
    m_magic = m__io->read_bytes(1);
    if (!(magic() == std::string("\x8B", 1))) {
        throw kaitai::validation_not_equal_error<std::string>(std::string("\x8B", 1), magic(), _io(), std::string("/types/tlm/seq/0"));
    }
    m_tlm = m__io->read_bits_int_be(14);
    m_integrity_status = m__io->read_bits_int_be(1);
    m_reserved = m__io->read_bits_int_be(1);
}

gps_t::tlm_t::~tlm_t() {
    _clean_up();
}

void gps_t::tlm_t::_clean_up() {
}

gps_t::subframe_2_t::subframe_2_t(kaitai::kstream* p__io, gps_t* p__parent, gps_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void gps_t::subframe_2_t::_read() {
    m_iode = m__io->read_u1();
    m_c_rs = m__io->read_s2be();
    m_delta_n = m__io->read_s2be();
    m_m_0 = m__io->read_s4be();
    m_c_uc = m__io->read_s2be();
    m_e = m__io->read_s4be();
    m_c_us = m__io->read_s2be();
    m_sqrt_a = m__io->read_u4be();
    m_t_oe = m__io->read_u2be();
    m_fit_interval_flag = m__io->read_bits_int_be(1);
    m_aoda = m__io->read_bits_int_be(5);
    m_reserved = m__io->read_bits_int_be(2);
}

gps_t::subframe_2_t::~subframe_2_t() {
    _clean_up();
}

void gps_t::subframe_2_t::_clean_up() {
}
