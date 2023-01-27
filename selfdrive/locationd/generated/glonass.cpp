// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "glonass.h"

glonass_t::glonass_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = this;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::_read() {
    m_idle_chip = m__io->read_bits_int_be(1);
    m_string_number = m__io->read_bits_int_be(4);
    //m__io->align_to_byte();
    switch (string_number()) {
    case 4: {
        m_data = new string_4_t(m__io, this, m__root);
        break;
    }
    case 1: {
        m_data = new string_1_t(m__io, this, m__root);
        break;
    }
    case 3: {
        m_data = new string_3_t(m__io, this, m__root);
        break;
    }
    case 2: {
        m_data = new string_2_t(m__io, this, m__root);
        break;
    }
    default: {
        m_data = new string_non_immediate_t(m__io, this, m__root);
        break;
    }
    }
    m_hamming_code = m__io->read_bits_int_be(8);
    m_pad_1 = m__io->read_bits_int_be(11);
    m_superframe_number = m__io->read_bits_int_be(16);
    m_pad_2 = m__io->read_bits_int_be(8);
    m_frame_number = m__io->read_bits_int_be(8);
}

glonass_t::~glonass_t() {
    _clean_up();
}

void glonass_t::_clean_up() {
    if (m_data) {
        delete m_data; m_data = 0;
    }
}

glonass_t::string_4_t::string_4_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::string_4_t::_read() {
    m_tau_n = m__io->read_bits_int_be(22);
    m_delta_tau_n = m__io->read_bits_int_be(5);
    m_e_n = m__io->read_bits_int_be(5);
    m_not_used_1 = m__io->read_bits_int_be(14);
    m_p4 = m__io->read_bits_int_be(1);
    m_f_t = m__io->read_bits_int_be(4);
    m_not_used_2 = m__io->read_bits_int_be(3);
    m_n_t = m__io->read_bits_int_be(11);
    m_n = m__io->read_bits_int_be(5);
    m_m = m__io->read_bits_int_be(2);
}

glonass_t::string_4_t::~string_4_t() {
    _clean_up();
}

void glonass_t::string_4_t::_clean_up() {
}

glonass_t::string_non_immediate_t::string_non_immediate_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::string_non_immediate_t::_read() {
    m_data_1 = m__io->read_bits_int_be(64);
    m_data_2 = m__io->read_bits_int_be(8);
}

glonass_t::string_non_immediate_t::~string_non_immediate_t() {
    _clean_up();
}

void glonass_t::string_non_immediate_t::_clean_up() {
}

glonass_t::string_1_t::string_1_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::string_1_t::_read() {
    m_not_used = m__io->read_bits_int_be(2);
    m_p1 = m__io->read_bits_int_be(2);
    m_t_k = m__io->read_bits_int_be(12);
    m_x_vel = m__io->read_bits_int_be(24);
    m_x_speedup = m__io->read_bits_int_be(5);
    m_x = m__io->read_bits_int_be(27);
}

glonass_t::string_1_t::~string_1_t() {
    _clean_up();
}

void glonass_t::string_1_t::_clean_up() {
}

glonass_t::string_2_t::string_2_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::string_2_t::_read() {
    m_b_n = m__io->read_bits_int_be(3);
    m_p2 = m__io->read_bits_int_be(1);
    m_t_b = m__io->read_bits_int_be(7);
    m_not_used = m__io->read_bits_int_be(5);
    m_y_vel = m__io->read_bits_int_be(24);
    m_y_speedup = m__io->read_bits_int_be(5);
    m_y = m__io->read_bits_int_be(27);
}

glonass_t::string_2_t::~string_2_t() {
    _clean_up();
}

void glonass_t::string_2_t::_clean_up() {
}

glonass_t::string_3_t::string_3_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::string_3_t::_read() {
    m_p3 = m__io->read_bits_int_be(1);
    m_gamma_n = m__io->read_bits_int_be(11);
    m_not_used = m__io->read_bits_int_be(1);
    m_p = m__io->read_bits_int_be(2);
    m_l_n = m__io->read_bits_int_be(1);
    m_z_vel = m__io->read_bits_int_be(24);
    m_z_speedup = m__io->read_bits_int_be(5);
    m_z = m__io->read_bits_int_be(27);
}

glonass_t::string_3_t::~string_3_t() {
    _clean_up();
}

void glonass_t::string_3_t::_clean_up() {
}
