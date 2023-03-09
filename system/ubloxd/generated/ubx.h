#ifndef UBX_H_
#define UBX_H_

// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "kaitai/kaitaistruct.h"
#include <stdint.h>
#include <vector>

#if KAITAI_STRUCT_VERSION < 9000L
#error "Incompatible Kaitai Struct C++/STL API: version 0.9 or later is required"
#endif

class ubx_t : public kaitai::kstruct {

public:
    class rxm_rawx_t;
    class rxm_sfrbx_t;
    class nav_pvt_t;
    class mon_hw2_t;
    class mon_hw_t;

    enum gnss_type_t {
        GNSS_TYPE_GPS = 0,
        GNSS_TYPE_SBAS = 1,
        GNSS_TYPE_GALILEO = 2,
        GNSS_TYPE_BEIDOU = 3,
        GNSS_TYPE_IMES = 4,
        GNSS_TYPE_QZSS = 5,
        GNSS_TYPE_GLONASS = 6
    };

    ubx_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = 0, ubx_t* p__root = 0);

private:
    void _read();
    void _clean_up();

public:
    ~ubx_t();

    class rxm_rawx_t : public kaitai::kstruct {

    public:
        class meas_t;

        rxm_rawx_t(kaitai::kstream* p__io, ubx_t* p__parent = 0, ubx_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~rxm_rawx_t();

        class meas_t : public kaitai::kstruct {

        public:

            meas_t(kaitai::kstream* p__io, ubx_t::rxm_rawx_t* p__parent = 0, ubx_t* p__root = 0);

        private:
            void _read();
            void _clean_up();

        public:
            ~meas_t();

        private:
            double m_pr_mes;
            double m_cp_mes;
            float m_do_mes;
            gnss_type_t m_gnss_id;
            uint8_t m_sv_id;
            std::string m_reserved2;
            uint8_t m_freq_id;
            uint16_t m_lock_time;
            uint8_t m_cno;
            uint8_t m_pr_stdev;
            uint8_t m_cp_stdev;
            uint8_t m_do_stdev;
            uint8_t m_trk_stat;
            std::string m_reserved3;
            ubx_t* m__root;
            ubx_t::rxm_rawx_t* m__parent;

        public:
            double pr_mes() const { return m_pr_mes; }
            double cp_mes() const { return m_cp_mes; }
            float do_mes() const { return m_do_mes; }
            gnss_type_t gnss_id() const { return m_gnss_id; }
            uint8_t sv_id() const { return m_sv_id; }
            std::string reserved2() const { return m_reserved2; }
            uint8_t freq_id() const { return m_freq_id; }
            uint16_t lock_time() const { return m_lock_time; }
            uint8_t cno() const { return m_cno; }
            uint8_t pr_stdev() const { return m_pr_stdev; }
            uint8_t cp_stdev() const { return m_cp_stdev; }
            uint8_t do_stdev() const { return m_do_stdev; }
            uint8_t trk_stat() const { return m_trk_stat; }
            std::string reserved3() const { return m_reserved3; }
            ubx_t* _root() const { return m__root; }
            ubx_t::rxm_rawx_t* _parent() const { return m__parent; }
        };

    private:
        double m_rcv_tow;
        uint16_t m_week;
        int8_t m_leap_s;
        uint8_t m_num_meas;
        uint8_t m_rec_stat;
        std::string m_reserved1;
        std::vector<meas_t*>* m_measurements;
        ubx_t* m__root;
        ubx_t* m__parent;
        std::vector<std::string>* m__raw_measurements;
        std::vector<kaitai::kstream*>* m__io__raw_measurements;

    public:
        double rcv_tow() const { return m_rcv_tow; }
        uint16_t week() const { return m_week; }
        int8_t leap_s() const { return m_leap_s; }
        uint8_t num_meas() const { return m_num_meas; }
        uint8_t rec_stat() const { return m_rec_stat; }
        std::string reserved1() const { return m_reserved1; }
        std::vector<meas_t*>* measurements() const { return m_measurements; }
        ubx_t* _root() const { return m__root; }
        ubx_t* _parent() const { return m__parent; }
        std::vector<std::string>* _raw_measurements() const { return m__raw_measurements; }
        std::vector<kaitai::kstream*>* _io__raw_measurements() const { return m__io__raw_measurements; }
    };

    class rxm_sfrbx_t : public kaitai::kstruct {

    public:

        rxm_sfrbx_t(kaitai::kstream* p__io, ubx_t* p__parent = 0, ubx_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~rxm_sfrbx_t();

    private:
        gnss_type_t m_gnss_id;
        uint8_t m_sv_id;
        std::string m_reserved1;
        uint8_t m_freq_id;
        uint8_t m_num_words;
        std::string m_reserved2;
        uint8_t m_version;
        std::string m_reserved3;
        std::vector<uint32_t>* m_body;
        ubx_t* m__root;
        ubx_t* m__parent;

    public:
        gnss_type_t gnss_id() const { return m_gnss_id; }
        uint8_t sv_id() const { return m_sv_id; }
        std::string reserved1() const { return m_reserved1; }
        uint8_t freq_id() const { return m_freq_id; }
        uint8_t num_words() const { return m_num_words; }
        std::string reserved2() const { return m_reserved2; }
        uint8_t version() const { return m_version; }
        std::string reserved3() const { return m_reserved3; }
        std::vector<uint32_t>* body() const { return m_body; }
        ubx_t* _root() const { return m__root; }
        ubx_t* _parent() const { return m__parent; }
    };

    class nav_pvt_t : public kaitai::kstruct {

    public:

        nav_pvt_t(kaitai::kstream* p__io, ubx_t* p__parent = 0, ubx_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~nav_pvt_t();

    private:
        uint32_t m_i_tow;
        uint16_t m_year;
        uint8_t m_month;
        uint8_t m_day;
        uint8_t m_hour;
        uint8_t m_min;
        uint8_t m_sec;
        uint8_t m_valid;
        uint32_t m_t_acc;
        int32_t m_nano;
        uint8_t m_fix_type;
        uint8_t m_flags;
        uint8_t m_flags2;
        uint8_t m_num_sv;
        int32_t m_lon;
        int32_t m_lat;
        int32_t m_height;
        int32_t m_h_msl;
        uint32_t m_h_acc;
        uint32_t m_v_acc;
        int32_t m_vel_n;
        int32_t m_vel_e;
        int32_t m_vel_d;
        int32_t m_g_speed;
        int32_t m_head_mot;
        int32_t m_s_acc;
        uint32_t m_head_acc;
        uint16_t m_p_dop;
        uint8_t m_flags3;
        std::string m_reserved1;
        int32_t m_head_veh;
        int16_t m_mag_dec;
        uint16_t m_mag_acc;
        ubx_t* m__root;
        ubx_t* m__parent;

    public:
        uint32_t i_tow() const { return m_i_tow; }
        uint16_t year() const { return m_year; }
        uint8_t month() const { return m_month; }
        uint8_t day() const { return m_day; }
        uint8_t hour() const { return m_hour; }
        uint8_t min() const { return m_min; }
        uint8_t sec() const { return m_sec; }
        uint8_t valid() const { return m_valid; }
        uint32_t t_acc() const { return m_t_acc; }
        int32_t nano() const { return m_nano; }
        uint8_t fix_type() const { return m_fix_type; }
        uint8_t flags() const { return m_flags; }
        uint8_t flags2() const { return m_flags2; }
        uint8_t num_sv() const { return m_num_sv; }
        int32_t lon() const { return m_lon; }
        int32_t lat() const { return m_lat; }
        int32_t height() const { return m_height; }
        int32_t h_msl() const { return m_h_msl; }
        uint32_t h_acc() const { return m_h_acc; }
        uint32_t v_acc() const { return m_v_acc; }
        int32_t vel_n() const { return m_vel_n; }
        int32_t vel_e() const { return m_vel_e; }
        int32_t vel_d() const { return m_vel_d; }
        int32_t g_speed() const { return m_g_speed; }
        int32_t head_mot() const { return m_head_mot; }
        int32_t s_acc() const { return m_s_acc; }
        uint32_t head_acc() const { return m_head_acc; }
        uint16_t p_dop() const { return m_p_dop; }
        uint8_t flags3() const { return m_flags3; }
        std::string reserved1() const { return m_reserved1; }
        int32_t head_veh() const { return m_head_veh; }
        int16_t mag_dec() const { return m_mag_dec; }
        uint16_t mag_acc() const { return m_mag_acc; }
        ubx_t* _root() const { return m__root; }
        ubx_t* _parent() const { return m__parent; }
    };

    class mon_hw2_t : public kaitai::kstruct {

    public:

        enum config_source_t {
            CONFIG_SOURCE_FLASH = 102,
            CONFIG_SOURCE_OTP = 111,
            CONFIG_SOURCE_CONFIG_PINS = 112,
            CONFIG_SOURCE_ROM = 113
        };

        mon_hw2_t(kaitai::kstream* p__io, ubx_t* p__parent = 0, ubx_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~mon_hw2_t();

    private:
        int8_t m_ofs_i;
        uint8_t m_mag_i;
        int8_t m_ofs_q;
        uint8_t m_mag_q;
        config_source_t m_cfg_source;
        std::string m_reserved1;
        uint32_t m_low_lev_cfg;
        std::string m_reserved2;
        uint32_t m_post_status;
        std::string m_reserved3;
        ubx_t* m__root;
        ubx_t* m__parent;

    public:
        int8_t ofs_i() const { return m_ofs_i; }
        uint8_t mag_i() const { return m_mag_i; }
        int8_t ofs_q() const { return m_ofs_q; }
        uint8_t mag_q() const { return m_mag_q; }
        config_source_t cfg_source() const { return m_cfg_source; }
        std::string reserved1() const { return m_reserved1; }
        uint32_t low_lev_cfg() const { return m_low_lev_cfg; }
        std::string reserved2() const { return m_reserved2; }
        uint32_t post_status() const { return m_post_status; }
        std::string reserved3() const { return m_reserved3; }
        ubx_t* _root() const { return m__root; }
        ubx_t* _parent() const { return m__parent; }
    };

    class mon_hw_t : public kaitai::kstruct {

    public:

        enum antenna_status_t {
            ANTENNA_STATUS_INIT = 0,
            ANTENNA_STATUS_DONTKNOW = 1,
            ANTENNA_STATUS_OK = 2,
            ANTENNA_STATUS_SHORT = 3,
            ANTENNA_STATUS_OPEN = 4
        };

        enum antenna_power_t {
            ANTENNA_POWER_FALSE = 0,
            ANTENNA_POWER_TRUE = 1,
            ANTENNA_POWER_DONTKNOW = 2
        };

        mon_hw_t(kaitai::kstream* p__io, ubx_t* p__parent = 0, ubx_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~mon_hw_t();

    private:
        uint32_t m_pin_sel;
        uint32_t m_pin_bank;
        uint32_t m_pin_dir;
        uint32_t m_pin_val;
        uint16_t m_noise_per_ms;
        uint16_t m_agc_cnt;
        antenna_status_t m_a_status;
        antenna_power_t m_a_power;
        uint8_t m_flags;
        std::string m_reserved1;
        uint32_t m_used_mask;
        std::string m_vp;
        uint8_t m_jam_ind;
        std::string m_reserved2;
        uint32_t m_pin_irq;
        uint32_t m_pull_h;
        uint32_t m_pull_l;
        ubx_t* m__root;
        ubx_t* m__parent;

    public:
        uint32_t pin_sel() const { return m_pin_sel; }
        uint32_t pin_bank() const { return m_pin_bank; }
        uint32_t pin_dir() const { return m_pin_dir; }
        uint32_t pin_val() const { return m_pin_val; }
        uint16_t noise_per_ms() const { return m_noise_per_ms; }
        uint16_t agc_cnt() const { return m_agc_cnt; }
        antenna_status_t a_status() const { return m_a_status; }
        antenna_power_t a_power() const { return m_a_power; }
        uint8_t flags() const { return m_flags; }
        std::string reserved1() const { return m_reserved1; }
        uint32_t used_mask() const { return m_used_mask; }
        std::string vp() const { return m_vp; }
        uint8_t jam_ind() const { return m_jam_ind; }
        std::string reserved2() const { return m_reserved2; }
        uint32_t pin_irq() const { return m_pin_irq; }
        uint32_t pull_h() const { return m_pull_h; }
        uint32_t pull_l() const { return m_pull_l; }
        ubx_t* _root() const { return m__root; }
        ubx_t* _parent() const { return m__parent; }
    };

private:
    bool f_checksum;
    uint16_t m_checksum;

public:
    uint16_t checksum();

private:
    std::string m_magic;
    uint16_t m_msg_type;
    uint16_t m_length;
    kaitai::kstruct* m_body;
    bool n_body;

public:
    bool _is_null_body() { body(); return n_body; };

private:
    ubx_t* m__root;
    kaitai::kstruct* m__parent;

public:
    std::string magic() const { return m_magic; }
    uint16_t msg_type() const { return m_msg_type; }
    uint16_t length() const { return m_length; }
    kaitai::kstruct* body() const { return m_body; }
    ubx_t* _root() const { return m__root; }
    kaitai::kstruct* _parent() const { return m__parent; }
};

#endif  // UBX_H_
