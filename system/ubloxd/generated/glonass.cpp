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
    case 5: {
        m_data = new string_5_t(m__io, this, m__root);
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
    f_tau_n = false;
    f_delta_tau_n = false;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::string_4_t::_read() {
    m_tau_n_sign = m__io->read_bits_int_be(1);
    m_tau_n_value = m__io->read_bits_int_be(21);
    m_delta_tau_n_sign = m__io->read_bits_int_be(1);
    m_delta_tau_n_value = m__io->read_bits_int_be(4);
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

int32_t glonass_t::string_4_t::tau_n() {
    if (f_tau_n)
        return m_tau_n;
    m_tau_n = ((tau_n_sign()) ? ((tau_n_value() * -1)) : (tau_n_value()));
    f_tau_n = true;
    return m_tau_n;
}

int32_t glonass_t::string_4_t::delta_tau_n() {
    if (f_delta_tau_n)
        return m_delta_tau_n;
    m_delta_tau_n = ((delta_tau_n_sign()) ? ((delta_tau_n_value() * -1)) : (delta_tau_n_value()));
    f_delta_tau_n = true;
    return m_delta_tau_n;
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

glonass_t::string_5_t::string_5_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::string_5_t::_read() {
    m_n_a = m__io->read_bits_int_be(11);
    m_tau_c = m__io->read_bits_int_be(32);
    m_not_used = m__io->read_bits_int_be(1);
    m_n_4 = m__io->read_bits_int_be(5);
    m_tau_gps = m__io->read_bits_int_be(22);
    m_l_n = m__io->read_bits_int_be(1);
}

glonass_t::string_5_t::~string_5_t() {
    _clean_up();
}

void glonass_t::string_5_t::_clean_up() {
}

glonass_t::string_1_t::string_1_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    f_x_vel = false;
    f_x_accel = false;
    f_x = false;

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
    m_x_vel_sign = m__io->read_bits_int_be(1);
    m_x_vel_value = m__io->read_bits_int_be(23);
    m_x_accel_sign = m__io->read_bits_int_be(1);
    m_x_accel_value = m__io->read_bits_int_be(4);
    m_x_sign = m__io->read_bits_int_be(1);
    m_x_value = m__io->read_bits_int_be(26);
}

glonass_t::string_1_t::~string_1_t() {
    _clean_up();
}

void glonass_t::string_1_t::_clean_up() {
}

int32_t glonass_t::string_1_t::x_vel() {
    if (f_x_vel)
        return m_x_vel;
    m_x_vel = ((x_vel_sign()) ? ((x_vel_value() * -1)) : (x_vel_value()));
    f_x_vel = true;
    return m_x_vel;
}

int32_t glonass_t::string_1_t::x_accel() {
    if (f_x_accel)
        return m_x_accel;
    m_x_accel = ((x_accel_sign()) ? ((x_accel_value() * -1)) : (x_accel_value()));
    f_x_accel = true;
    return m_x_accel;
}

int32_t glonass_t::string_1_t::x() {
    if (f_x)
        return m_x;
    m_x = ((x_sign()) ? ((x_value() * -1)) : (x_value()));
    f_x = true;
    return m_x;
}

glonass_t::string_2_t::string_2_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    f_y_vel = false;
    f_y_accel = false;
    f_y = false;

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
    m_y_vel_sign = m__io->read_bits_int_be(1);
    m_y_vel_value = m__io->read_bits_int_be(23);
    m_y_accel_sign = m__io->read_bits_int_be(1);
    m_y_accel_value = m__io->read_bits_int_be(4);
    m_y_sign = m__io->read_bits_int_be(1);
    m_y_value = m__io->read_bits_int_be(26);
}

glonass_t::string_2_t::~string_2_t() {
    _clean_up();
}

void glonass_t::string_2_t::_clean_up() {
}

int32_t glonass_t::string_2_t::y_vel() {
    if (f_y_vel)
        return m_y_vel;
    m_y_vel = ((y_vel_sign()) ? ((y_vel_value() * -1)) : (y_vel_value()));
    f_y_vel = true;
    return m_y_vel;
}

int32_t glonass_t::string_2_t::y_accel() {
    if (f_y_accel)
        return m_y_accel;
    m_y_accel = ((y_accel_sign()) ? ((y_accel_value() * -1)) : (y_accel_value()));
    f_y_accel = true;
    return m_y_accel;
}

int32_t glonass_t::string_2_t::y() {
    if (f_y)
        return m_y;
    m_y = ((y_sign()) ? ((y_value() * -1)) : (y_value()));
    f_y = true;
    return m_y;
}

glonass_t::string_3_t::string_3_t(kaitai::kstream* p__io, glonass_t* p__parent, glonass_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    f_gamma_n = false;
    f_z_vel = false;
    f_z_accel = false;
    f_z = false;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void glonass_t::string_3_t::_read() {
    m_p3 = m__io->read_bits_int_be(1);
    m_gamma_n_sign = m__io->read_bits_int_be(1);
    m_gamma_n_value = m__io->read_bits_int_be(10);
    m_not_used = m__io->read_bits_int_be(1);
    m_p = m__io->read_bits_int_be(2);
    m_l_n = m__io->read_bits_int_be(1);
    m_z_vel_sign = m__io->read_bits_int_be(1);
    m_z_vel_value = m__io->read_bits_int_be(23);
    m_z_accel_sign = m__io->read_bits_int_be(1);
    m_z_accel_value = m__io->read_bits_int_be(4);
    m_z_sign = m__io->read_bits_int_be(1);
    m_z_value = m__io->read_bits_int_be(26);
}

glonass_t::string_3_t::~string_3_t() {
    _clean_up();
}

void glonass_t::string_3_t::_clean_up() {
}

int32_t glonass_t::string_3_t::gamma_n() {
    if (f_gamma_n)
        return m_gamma_n;
    m_gamma_n = ((gamma_n_sign()) ? ((gamma_n_value() * -1)) : (gamma_n_value()));
    f_gamma_n = true;
    return m_gamma_n;
}

int32_t glonass_t::string_3_t::z_vel() {
    if (f_z_vel)
        return m_z_vel;
    m_z_vel = ((z_vel_sign()) ? ((z_vel_value() * -1)) : (z_vel_value()));
    f_z_vel = true;
    return m_z_vel;
}

int32_t glonass_t::string_3_t::z_accel() {
    if (f_z_accel)
        return m_z_accel;
    m_z_accel = ((z_accel_sign()) ? ((z_accel_value() * -1)) : (z_accel_value()));
    f_z_accel = true;
    return m_z_accel;
}

int32_t glonass_t::string_3_t::z() {
    if (f_z)
        return m_z;
    m_z = ((z_sign()) ? ((z_value() * -1)) : (z_value()));
    f_z = true;
    return m_z;
}
