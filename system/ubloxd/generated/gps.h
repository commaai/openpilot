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
    class subframe_1_t;
    class subframe_3_t;
    class subframe_4_t;
    class how_t;
    class tlm_t;
    class subframe_2_t;

    gps_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = 0, gps_t* p__root = 0);

private:
    void _read();
    void _clean_up();

public:
    ~gps_t();

    class subframe_1_t : public kaitai::kstruct {

    public:

        subframe_1_t(kaitai::kstream* p__io, gps_t* p__parent = 0, gps_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~subframe_1_t();

    private:
        bool f_af_0;
        int32_t m_af_0;

    public:
        int32_t af_0();

    private:
        uint64_t m_week_no;
        uint64_t m_code;
        uint64_t m_sv_accuracy;
        uint64_t m_sv_health;
        uint64_t m_iodc_msb;
        bool m_l2_p_data_flag;
        uint64_t m_reserved1;
        uint64_t m_reserved2;
        uint64_t m_reserved3;
        uint64_t m_reserved4;
        int8_t m_t_gd;
        uint8_t m_iodc_lsb;
        uint16_t m_t_oc;
        int8_t m_af_2;
        int16_t m_af_1;
        bool m_af_0_sign;
        uint64_t m_af_0_value;
        uint64_t m_reserved5;
        gps_t* m__root;
        gps_t* m__parent;

    public:
        uint64_t week_no() const { return m_week_no; }
        uint64_t code() const { return m_code; }
        uint64_t sv_accuracy() const { return m_sv_accuracy; }
        uint64_t sv_health() const { return m_sv_health; }
        uint64_t iodc_msb() const { return m_iodc_msb; }
        bool l2_p_data_flag() const { return m_l2_p_data_flag; }
        uint64_t reserved1() const { return m_reserved1; }
        uint64_t reserved2() const { return m_reserved2; }
        uint64_t reserved3() const { return m_reserved3; }
        uint64_t reserved4() const { return m_reserved4; }
        int8_t t_gd() const { return m_t_gd; }
        uint8_t iodc_lsb() const { return m_iodc_lsb; }
        uint16_t t_oc() const { return m_t_oc; }
        int8_t af_2() const { return m_af_2; }
        int16_t af_1() const { return m_af_1; }
        bool af_0_sign() const { return m_af_0_sign; }
        uint64_t af_0_value() const { return m_af_0_value; }
        uint64_t reserved5() const { return m_reserved5; }
        gps_t* _root() const { return m__root; }
        gps_t* _parent() const { return m__parent; }
    };

    class subframe_3_t : public kaitai::kstruct {

    public:

        subframe_3_t(kaitai::kstream* p__io, gps_t* p__parent = 0, gps_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~subframe_3_t();

    private:
        bool f_omega_dot;
        int32_t m_omega_dot;

    public:
        int32_t omega_dot();

    private:
        bool f_idot;
        int32_t m_idot;

    public:
        int32_t idot();

    private:
        int16_t m_c_ic;
        int32_t m_omega_0;
        int16_t m_c_is;
        int32_t m_i_0;
        int16_t m_c_rc;
        int32_t m_omega;
        bool m_omega_dot_sign;
        uint64_t m_omega_dot_value;
        uint8_t m_iode;
        bool m_idot_sign;
        uint64_t m_idot_value;
        uint64_t m_reserved;
        gps_t* m__root;
        gps_t* m__parent;

    public:
        int16_t c_ic() const { return m_c_ic; }
        int32_t omega_0() const { return m_omega_0; }
        int16_t c_is() const { return m_c_is; }
        int32_t i_0() const { return m_i_0; }
        int16_t c_rc() const { return m_c_rc; }
        int32_t omega() const { return m_omega; }
        bool omega_dot_sign() const { return m_omega_dot_sign; }
        uint64_t omega_dot_value() const { return m_omega_dot_value; }
        uint8_t iode() const { return m_iode; }
        bool idot_sign() const { return m_idot_sign; }
        uint64_t idot_value() const { return m_idot_value; }
        uint64_t reserved() const { return m_reserved; }
        gps_t* _root() const { return m__root; }
        gps_t* _parent() const { return m__parent; }
    };

    class subframe_4_t : public kaitai::kstruct {

    public:
        class ionosphere_data_t;

        subframe_4_t(kaitai::kstream* p__io, gps_t* p__parent = 0, gps_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~subframe_4_t();

        class ionosphere_data_t : public kaitai::kstruct {

        public:

            ionosphere_data_t(kaitai::kstream* p__io, gps_t::subframe_4_t* p__parent = 0, gps_t* p__root = 0);

        private:
            void _read();
            void _clean_up();

        public:
            ~ionosphere_data_t();

        private:
            int8_t m_a0;
            int8_t m_a1;
            int8_t m_a2;
            int8_t m_a3;
            int8_t m_b0;
            int8_t m_b1;
            int8_t m_b2;
            int8_t m_b3;
            gps_t* m__root;
            gps_t::subframe_4_t* m__parent;

        public:
            int8_t a0() const { return m_a0; }
            int8_t a1() const { return m_a1; }
            int8_t a2() const { return m_a2; }
            int8_t a3() const { return m_a3; }
            int8_t b0() const { return m_b0; }
            int8_t b1() const { return m_b1; }
            int8_t b2() const { return m_b2; }
            int8_t b3() const { return m_b3; }
            gps_t* _root() const { return m__root; }
            gps_t::subframe_4_t* _parent() const { return m__parent; }
        };

    private:
        uint64_t m_data_id;
        uint64_t m_page_id;
        ionosphere_data_t* m_body;
        bool n_body;

    public:
        bool _is_null_body() { body(); return n_body; };

    private:
        gps_t* m__root;
        gps_t* m__parent;

    public:
        uint64_t data_id() const { return m_data_id; }
        uint64_t page_id() const { return m_page_id; }
        ionosphere_data_t* body() const { return m_body; }
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

    class tlm_t : public kaitai::kstruct {

    public:

        tlm_t(kaitai::kstream* p__io, gps_t* p__parent = 0, gps_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~tlm_t();

    private:
        std::string m_preamble;
        uint64_t m_tlm;
        bool m_integrity_status;
        bool m_reserved;
        gps_t* m__root;
        gps_t* m__parent;

    public:
        std::string preamble() const { return m_preamble; }
        uint64_t tlm() const { return m_tlm; }
        bool integrity_status() const { return m_integrity_status; }
        bool reserved() const { return m_reserved; }
        gps_t* _root() const { return m__root; }
        gps_t* _parent() const { return m__parent; }
    };

    class subframe_2_t : public kaitai::kstruct {

    public:

        subframe_2_t(kaitai::kstream* p__io, gps_t* p__parent = 0, gps_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~subframe_2_t();

    private:
        uint8_t m_iode;
        int16_t m_c_rs;
        int16_t m_delta_n;
        int32_t m_m_0;
        int16_t m_c_uc;
        int32_t m_e;
        int16_t m_c_us;
        uint32_t m_sqrt_a;
        uint16_t m_t_oe;
        bool m_fit_interval_flag;
        uint64_t m_aoda;
        uint64_t m_reserved;
        gps_t* m__root;
        gps_t* m__parent;

    public:
        uint8_t iode() const { return m_iode; }
        int16_t c_rs() const { return m_c_rs; }
        int16_t delta_n() const { return m_delta_n; }
        int32_t m_0() const { return m_m_0; }
        int16_t c_uc() const { return m_c_uc; }
        int32_t e() const { return m_e; }
        int16_t c_us() const { return m_c_us; }
        uint32_t sqrt_a() const { return m_sqrt_a; }
        uint16_t t_oe() const { return m_t_oe; }
        bool fit_interval_flag() const { return m_fit_interval_flag; }
        uint64_t aoda() const { return m_aoda; }
        uint64_t reserved() const { return m_reserved; }
        gps_t* _root() const { return m__root; }
        gps_t* _parent() const { return m__parent; }
    };

private:
    tlm_t* m_tlm;
    how_t* m_how;
    kaitai::kstruct* m_body;
    bool n_body;

public:
    bool _is_null_body() { body(); return n_body; };

private:
    gps_t* m__root;
    kaitai::kstruct* m__parent;

public:
    tlm_t* tlm() const { return m_tlm; }
    how_t* how() const { return m_how; }
    kaitai::kstruct* body() const { return m_body; }
    gps_t* _root() const { return m__root; }
    kaitai::kstruct* _parent() const { return m__parent; }
};

#endif  // GPS_H_
