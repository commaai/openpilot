#ifndef GPS_H_
#define GPS_H_

// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "kaitai/kaitaistruct.h"
#include <stdint.h>

#if KAITAI_STRUCT_VERSION < 9000L
#error "Incompatible Kaitai Struct C++/STL API: version 0.9 or later is required"
#endif

class gps_t : public kaitai::kstruct {

public:
    class tlm_t;
    class how_t;

    gps_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = 0, gps_t* p__root = 0);

private:
    void _read();
    void _clean_up();

public:
    ~gps_t();

    class tlm_t : public kaitai::kstruct {

    public:

        tlm_t(kaitai::kstream* p__io, gps_t* p__parent = 0, gps_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~tlm_t();

    private:
        std::string m_magic;
        uint64_t m_tlm;
        bool m_integrity_status;
        bool m_reserved;
        gps_t* m__root;
        gps_t* m__parent;

    public:
        std::string magic() const { return m_magic; }
        uint64_t tlm() const { return m_tlm; }
        bool integrity_status() const { return m_integrity_status; }
        bool reserved() const { return m_reserved; }
        gps_t* _root() const { return m__root; }
        gps_t* _parent() const { return m__parent; }
    };

    class how_t : public kaitai::kstruct {

    public:

        how_t(kaitai::kstream* p__io, gps_t* p__parent = 0, gps_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~how_t();

    private:
        uint64_t m_tow_count;
        bool m_alert;
        bool m_anti_spoof;
        uint64_t m_subframe_id;
        uint64_t m_reserved;
        gps_t* m__root;
        gps_t* m__parent;

    public:
        uint64_t tow_count() const { return m_tow_count; }
        bool alert() const { return m_alert; }
        bool anti_spoof() const { return m_anti_spoof; }
        uint64_t subframe_id() const { return m_subframe_id; }
        uint64_t reserved() const { return m_reserved; }
        gps_t* _root() const { return m__root; }
        gps_t* _parent() const { return m__parent; }
    };

private:
    tlm_t* m_tlm;
    how_t* m_how;
    gps_t* m__root;
    kaitai::kstruct* m__parent;

public:
    tlm_t* tlm() const { return m_tlm; }
    how_t* how() const { return m_how; }
    gps_t* _root() const { return m__root; }
    kaitai::kstruct* _parent() const { return m__parent; }
};

#endif  // GPS_H_
