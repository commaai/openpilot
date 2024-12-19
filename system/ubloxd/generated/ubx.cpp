// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "ubx.h"
#include "kaitai/exceptions.h"

ubx_t::ubx_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = this;
    f_checksum = false;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::_read() {
    m_magic = m__io->read_bytes(2);
    if (!(magic() == std::string("\xB5\x62", 2))) {
        throw kaitai::validation_not_equal_error<std::string>(std::string("\xB5\x62", 2), magic(), _io(), std::string("/seq/0"));
    }
    m_msg_type = m__io->read_u2be();
    m_length = m__io->read_u2le();
    n_body = true;
    switch (msg_type()) {
    case 2569: {
        n_body = false;
        m_body = new mon_hw_t(m__io, this, m__root);
        break;
    }
    case 533: {
        n_body = false;
        m_body = new rxm_rawx_t(m__io, this, m__root);
        break;
    }
    case 531: {
        n_body = false;
        m_body = new rxm_sfrbx_t(m__io, this, m__root);
        break;
    }
    case 309: {
        n_body = false;
        m_body = new nav_sat_t(m__io, this, m__root);
        break;
    }
    case 2571: {
        n_body = false;
        m_body = new mon_hw2_t(m__io, this, m__root);
        break;
    }
    case 263: {
        n_body = false;
        m_body = new nav_pvt_t(m__io, this, m__root);
        break;
    }
    }
}

ubx_t::~ubx_t() {
    _clean_up();
}

void ubx_t::_clean_up() {
    if (!n_body) {
        if (m_body) {
            delete m_body; m_body = 0;
        }
    }
    if (f_checksum) {
    }
}

ubx_t::rxm_rawx_t::rxm_rawx_t(kaitai::kstream* p__io, ubx_t* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    m_meas = 0;
    m__raw_meas = 0;
    m__io__raw_meas = 0;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::rxm_rawx_t::_read() {
    m_rcv_tow = m__io->read_f8le();
    m_week = m__io->read_u2le();
    m_leap_s = m__io->read_s1();
    m_num_meas = m__io->read_u1();
    m_rec_stat = m__io->read_u1();
    m_reserved1 = m__io->read_bytes(3);
    m__raw_meas = new std::vector<std::string>();
    m__io__raw_meas = new std::vector<kaitai::kstream*>();
    m_meas = new std::vector<measurement_t*>();
    const int l_meas = num_meas();
    for (int i = 0; i < l_meas; i++) {
        m__raw_meas->push_back(m__io->read_bytes(32));
        kaitai::kstream* io__raw_meas = new kaitai::kstream(m__raw_meas->at(m__raw_meas->size() - 1));
        m__io__raw_meas->push_back(io__raw_meas);
        m_meas->push_back(new measurement_t(io__raw_meas, this, m__root));
    }
}

ubx_t::rxm_rawx_t::~rxm_rawx_t() {
    _clean_up();
}

void ubx_t::rxm_rawx_t::_clean_up() {
    if (m__raw_meas) {
        delete m__raw_meas; m__raw_meas = 0;
    }
    if (m__io__raw_meas) {
        for (std::vector<kaitai::kstream*>::iterator it = m__io__raw_meas->begin(); it != m__io__raw_meas->end(); ++it) {
            delete *it;
        }
        delete m__io__raw_meas; m__io__raw_meas = 0;
    }
    if (m_meas) {
        for (std::vector<measurement_t*>::iterator it = m_meas->begin(); it != m_meas->end(); ++it) {
            delete *it;
        }
        delete m_meas; m_meas = 0;
    }
}

ubx_t::rxm_rawx_t::measurement_t::measurement_t(kaitai::kstream* p__io, ubx_t::rxm_rawx_t* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::rxm_rawx_t::measurement_t::_read() {
    m_pr_mes = m__io->read_f8le();
    m_cp_mes = m__io->read_f8le();
    m_do_mes = m__io->read_f4le();
    m_gnss_id = static_cast<ubx_t::gnss_type_t>(m__io->read_u1());
    m_sv_id = m__io->read_u1();
    m_reserved2 = m__io->read_bytes(1);
    m_freq_id = m__io->read_u1();
    m_lock_time = m__io->read_u2le();
    m_cno = m__io->read_u1();
    m_pr_stdev = m__io->read_u1();
    m_cp_stdev = m__io->read_u1();
    m_do_stdev = m__io->read_u1();
    m_trk_stat = m__io->read_u1();
    m_reserved3 = m__io->read_bytes(1);
}

ubx_t::rxm_rawx_t::measurement_t::~measurement_t() {
    _clean_up();
}

void ubx_t::rxm_rawx_t::measurement_t::_clean_up() {
}

ubx_t::rxm_sfrbx_t::rxm_sfrbx_t(kaitai::kstream* p__io, ubx_t* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    m_body = 0;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::rxm_sfrbx_t::_read() {
    m_gnss_id = static_cast<ubx_t::gnss_type_t>(m__io->read_u1());
    m_sv_id = m__io->read_u1();
    m_reserved1 = m__io->read_bytes(1);
    m_freq_id = m__io->read_u1();
    m_num_words = m__io->read_u1();
    m_reserved2 = m__io->read_bytes(1);
    m_version = m__io->read_u1();
    m_reserved3 = m__io->read_bytes(1);
    m_body = new std::vector<uint32_t>();
    const int l_body = num_words();
    for (int i = 0; i < l_body; i++) {
        m_body->push_back(m__io->read_u4le());
    }
}

ubx_t::rxm_sfrbx_t::~rxm_sfrbx_t() {
    _clean_up();
}

void ubx_t::rxm_sfrbx_t::_clean_up() {
    if (m_body) {
        delete m_body; m_body = 0;
    }
}

ubx_t::nav_sat_t::nav_sat_t(kaitai::kstream* p__io, ubx_t* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    m_svs = 0;
    m__raw_svs = 0;
    m__io__raw_svs = 0;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::nav_sat_t::_read() {
    m_itow = m__io->read_u4le();
    m_version = m__io->read_u1();
    m_num_svs = m__io->read_u1();
    m_reserved = m__io->read_bytes(2);
    m__raw_svs = new std::vector<std::string>();
    m__io__raw_svs = new std::vector<kaitai::kstream*>();
    m_svs = new std::vector<nav_t*>();
    const int l_svs = num_svs();
    for (int i = 0; i < l_svs; i++) {
        m__raw_svs->push_back(m__io->read_bytes(12));
        kaitai::kstream* io__raw_svs = new kaitai::kstream(m__raw_svs->at(m__raw_svs->size() - 1));
        m__io__raw_svs->push_back(io__raw_svs);
        m_svs->push_back(new nav_t(io__raw_svs, this, m__root));
    }
}

ubx_t::nav_sat_t::~nav_sat_t() {
    _clean_up();
}

void ubx_t::nav_sat_t::_clean_up() {
    if (m__raw_svs) {
        delete m__raw_svs; m__raw_svs = 0;
    }
    if (m__io__raw_svs) {
        for (std::vector<kaitai::kstream*>::iterator it = m__io__raw_svs->begin(); it != m__io__raw_svs->end(); ++it) {
            delete *it;
        }
        delete m__io__raw_svs; m__io__raw_svs = 0;
    }
    if (m_svs) {
        for (std::vector<nav_t*>::iterator it = m_svs->begin(); it != m_svs->end(); ++it) {
            delete *it;
        }
        delete m_svs; m_svs = 0;
    }
}

ubx_t::nav_sat_t::nav_t::nav_t(kaitai::kstream* p__io, ubx_t::nav_sat_t* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::nav_sat_t::nav_t::_read() {
    m_gnss_id = static_cast<ubx_t::gnss_type_t>(m__io->read_u1());
    m_sv_id = m__io->read_u1();
    m_cno = m__io->read_u1();
    m_elev = m__io->read_s1();
    m_azim = m__io->read_s2le();
    m_pr_res = m__io->read_s2le();
    m_flags = m__io->read_u4le();
}

ubx_t::nav_sat_t::nav_t::~nav_t() {
    _clean_up();
}

void ubx_t::nav_sat_t::nav_t::_clean_up() {
}

ubx_t::nav_pvt_t::nav_pvt_t(kaitai::kstream* p__io, ubx_t* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::nav_pvt_t::_read() {
    m_i_tow = m__io->read_u4le();
    m_year = m__io->read_u2le();
    m_month = m__io->read_u1();
    m_day = m__io->read_u1();
    m_hour = m__io->read_u1();
    m_min = m__io->read_u1();
    m_sec = m__io->read_u1();
    m_valid = m__io->read_u1();
    m_t_acc = m__io->read_u4le();
    m_nano = m__io->read_s4le();
    m_fix_type = m__io->read_u1();
    m_flags = m__io->read_u1();
    m_flags2 = m__io->read_u1();
    m_num_sv = m__io->read_u1();
    m_lon = m__io->read_s4le();
    m_lat = m__io->read_s4le();
    m_height = m__io->read_s4le();
    m_h_msl = m__io->read_s4le();
    m_h_acc = m__io->read_u4le();
    m_v_acc = m__io->read_u4le();
    m_vel_n = m__io->read_s4le();
    m_vel_e = m__io->read_s4le();
    m_vel_d = m__io->read_s4le();
    m_g_speed = m__io->read_s4le();
    m_head_mot = m__io->read_s4le();
    m_s_acc = m__io->read_s4le();
    m_head_acc = m__io->read_u4le();
    m_p_dop = m__io->read_u2le();
    m_flags3 = m__io->read_u1();
    m_reserved1 = m__io->read_bytes(5);
    m_head_veh = m__io->read_s4le();
    m_mag_dec = m__io->read_s2le();
    m_mag_acc = m__io->read_u2le();
}

ubx_t::nav_pvt_t::~nav_pvt_t() {
    _clean_up();
}

void ubx_t::nav_pvt_t::_clean_up() {
}

ubx_t::mon_hw2_t::mon_hw2_t(kaitai::kstream* p__io, ubx_t* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::mon_hw2_t::_read() {
    m_ofs_i = m__io->read_s1();
    m_mag_i = m__io->read_u1();
    m_ofs_q = m__io->read_s1();
    m_mag_q = m__io->read_u1();
    m_cfg_source = static_cast<ubx_t::mon_hw2_t::config_source_t>(m__io->read_u1());
    m_reserved1 = m__io->read_bytes(3);
    m_low_lev_cfg = m__io->read_u4le();
    m_reserved2 = m__io->read_bytes(8);
    m_post_status = m__io->read_u4le();
    m_reserved3 = m__io->read_bytes(4);
}

ubx_t::mon_hw2_t::~mon_hw2_t() {
    _clean_up();
}

void ubx_t::mon_hw2_t::_clean_up() {
}

ubx_t::mon_hw_t::mon_hw_t(kaitai::kstream* p__io, ubx_t* p__parent, ubx_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void ubx_t::mon_hw_t::_read() {
    m_pin_sel = m__io->read_u4le();
    m_pin_bank = m__io->read_u4le();
    m_pin_dir = m__io->read_u4le();
    m_pin_val = m__io->read_u4le();
    m_noise_per_ms = m__io->read_u2le();
    m_agc_cnt = m__io->read_u2le();
    m_a_status = static_cast<ubx_t::mon_hw_t::antenna_status_t>(m__io->read_u1());
    m_a_power = static_cast<ubx_t::mon_hw_t::antenna_power_t>(m__io->read_u1());
    m_flags = m__io->read_u1();
    m_reserved1 = m__io->read_bytes(1);
    m_used_mask = m__io->read_u4le();
    m_vp = m__io->read_bytes(17);
    m_jam_ind = m__io->read_u1();
    m_reserved2 = m__io->read_bytes(2);
    m_pin_irq = m__io->read_u4le();
    m_pull_h = m__io->read_u4le();
    m_pull_l = m__io->read_u4le();
}

ubx_t::mon_hw_t::~mon_hw_t() {
    _clean_up();
}

void ubx_t::mon_hw_t::_clean_up() {
}

uint16_t ubx_t::checksum() {
    if (f_checksum)
        return m_checksum;
    std::streampos _pos = m__io->pos();
    m__io->seek((length() + 6));
    m_checksum = m__io->read_u2le();
    m__io->seek(_pos);
    f_checksum = true;
    return m_checksum;
}
