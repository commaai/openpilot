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
