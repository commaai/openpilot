#ifndef GLONASS_H_
#define GLONASS_H_

// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "kaitai/kaitaistruct.h"
#include <stdint.h>

#if KAITAI_STRUCT_VERSION < 9000L
#error "Incompatible Kaitai Struct C++/STL API: version 0.9 or later is required"
#endif

class glonass_t : public kaitai::kstruct {

public:
    class string_4_t;
    class string_non_immediate_t;
    class string_5_t;
    class string_1_t;
    class string_2_t;
    class string_3_t;

    glonass_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = 0, glonass_t* p__root = 0);

private:
    void _read();
    void _clean_up();

public:
    ~glonass_t();

    class string_4_t : public kaitai::kstruct {

    public:

        string_4_t(kaitai::kstream* p__io, glonass_t* p__parent = 0, glonass_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~string_4_t();

    private:
        bool f_tau_n;
        int32_t m_tau_n;

    public:
        int32_t tau_n();

    private:
        bool f_delta_tau_n;
        int32_t m_delta_tau_n;

    public:
        int32_t delta_tau_n();

    private:
        bool m_tau_n_sign;
        uint64_t m_tau_n_value;
        bool m_delta_tau_n_sign;
        uint64_t m_delta_tau_n_value;
        uint64_t m_e_n;
        uint64_t m_not_used_1;
        bool m_p4;
        uint64_t m_f_t;
        uint64_t m_not_used_2;
        uint64_t m_n_t;
        uint64_t m_n;
        uint64_t m_m;
        glonass_t* m__root;
        glonass_t* m__parent;

    public:
        bool tau_n_sign() const { return m_tau_n_sign; }
        uint64_t tau_n_value() const { return m_tau_n_value; }
        bool delta_tau_n_sign() const { return m_delta_tau_n_sign; }
        uint64_t delta_tau_n_value() const { return m_delta_tau_n_value; }
        uint64_t e_n() const { return m_e_n; }
        uint64_t not_used_1() const { return m_not_used_1; }
        bool p4() const { return m_p4; }
        uint64_t f_t() const { return m_f_t; }
        uint64_t not_used_2() const { return m_not_used_2; }
        uint64_t n_t() const { return m_n_t; }
        uint64_t n() const { return m_n; }
        uint64_t m() const { return m_m; }
        glonass_t* _root() const { return m__root; }
        glonass_t* _parent() const { return m__parent; }
    };

    class string_non_immediate_t : public kaitai::kstruct {

    public:

        string_non_immediate_t(kaitai::kstream* p__io, glonass_t* p__parent = 0, glonass_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~string_non_immediate_t();

    private:
        uint64_t m_data_1;
        uint64_t m_data_2;
        glonass_t* m__root;
        glonass_t* m__parent;

    public:
        uint64_t data_1() const { return m_data_1; }
        uint64_t data_2() const { return m_data_2; }
        glonass_t* _root() const { return m__root; }
        glonass_t* _parent() const { return m__parent; }
    };

    class string_5_t : public kaitai::kstruct {

    public:

        string_5_t(kaitai::kstream* p__io, glonass_t* p__parent = 0, glonass_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~string_5_t();

    private:
        uint64_t m_n_a;
        uint64_t m_tau_c;
        bool m_not_used;
        uint64_t m_n_4;
        uint64_t m_tau_gps;
        bool m_l_n;
        glonass_t* m__root;
        glonass_t* m__parent;

    public:
        uint64_t n_a() const { return m_n_a; }
        uint64_t tau_c() const { return m_tau_c; }
        bool not_used() const { return m_not_used; }
        uint64_t n_4() const { return m_n_4; }
        uint64_t tau_gps() const { return m_tau_gps; }
        bool l_n() const { return m_l_n; }
        glonass_t* _root() const { return m__root; }
        glonass_t* _parent() const { return m__parent; }
    };

    class string_1_t : public kaitai::kstruct {

    public:

        string_1_t(kaitai::kstream* p__io, glonass_t* p__parent = 0, glonass_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~string_1_t();

    private:
        bool f_x_vel;
        int32_t m_x_vel;

    public:
        int32_t x_vel();

    private:
        bool f_x_accel;
        int32_t m_x_accel;

    public:
        int32_t x_accel();

    private:
        bool f_x;
        int32_t m_x;

    public:
        int32_t x();

    private:
        uint64_t m_not_used;
        uint64_t m_p1;
        uint64_t m_t_k;
        bool m_x_vel_sign;
        uint64_t m_x_vel_value;
        bool m_x_accel_sign;
        uint64_t m_x_accel_value;
        bool m_x_sign;
        uint64_t m_x_value;
        glonass_t* m__root;
        glonass_t* m__parent;

    public:
        uint64_t not_used() const { return m_not_used; }
        uint64_t p1() const { return m_p1; }
        uint64_t t_k() const { return m_t_k; }
        bool x_vel_sign() const { return m_x_vel_sign; }
        uint64_t x_vel_value() const { return m_x_vel_value; }
        bool x_accel_sign() const { return m_x_accel_sign; }
        uint64_t x_accel_value() const { return m_x_accel_value; }
        bool x_sign() const { return m_x_sign; }
        uint64_t x_value() const { return m_x_value; }
        glonass_t* _root() const { return m__root; }
        glonass_t* _parent() const { return m__parent; }
    };

    class string_2_t : public kaitai::kstruct {

    public:

        string_2_t(kaitai::kstream* p__io, glonass_t* p__parent = 0, glonass_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~string_2_t();

    private:
        bool f_y_vel;
        int32_t m_y_vel;

    public:
        int32_t y_vel();

    private:
        bool f_y_accel;
        int32_t m_y_accel;

    public:
        int32_t y_accel();

    private:
        bool f_y;
        int32_t m_y;

    public:
        int32_t y();

    private:
        uint64_t m_b_n;
        bool m_p2;
        uint64_t m_t_b;
        uint64_t m_not_used;
        bool m_y_vel_sign;
        uint64_t m_y_vel_value;
        bool m_y_accel_sign;
        uint64_t m_y_accel_value;
        bool m_y_sign;
        uint64_t m_y_value;
        glonass_t* m__root;
        glonass_t* m__parent;

    public:
        uint64_t b_n() const { return m_b_n; }
        bool p2() const { return m_p2; }
        uint64_t t_b() const { return m_t_b; }
        uint64_t not_used() const { return m_not_used; }
        bool y_vel_sign() const { return m_y_vel_sign; }
        uint64_t y_vel_value() const { return m_y_vel_value; }
        bool y_accel_sign() const { return m_y_accel_sign; }
        uint64_t y_accel_value() const { return m_y_accel_value; }
        bool y_sign() const { return m_y_sign; }
        uint64_t y_value() const { return m_y_value; }
        glonass_t* _root() const { return m__root; }
        glonass_t* _parent() const { return m__parent; }
    };

    class string_3_t : public kaitai::kstruct {

    public:

        string_3_t(kaitai::kstream* p__io, glonass_t* p__parent = 0, glonass_t* p__root = 0);

    private:
        void _read();
        void _clean_up();

    public:
        ~string_3_t();

    private:
        bool f_gamma_n;
        int32_t m_gamma_n;

    public:
        int32_t gamma_n();

    private:
        bool f_z_vel;
        int32_t m_z_vel;

    public:
        int32_t z_vel();

    private:
        bool f_z_accel;
        int32_t m_z_accel;

    public:
        int32_t z_accel();

    private:
        bool f_z;
        int32_t m_z;

    public:
        int32_t z();

    private:
        bool m_p3;
        bool m_gamma_n_sign;
        uint64_t m_gamma_n_value;
        bool m_not_used;
        uint64_t m_p;
        bool m_l_n;
        bool m_z_vel_sign;
        uint64_t m_z_vel_value;
        bool m_z_accel_sign;
        uint64_t m_z_accel_value;
        bool m_z_sign;
        uint64_t m_z_value;
        glonass_t* m__root;
        glonass_t* m__parent;

    public:
        bool p3() const { return m_p3; }
        bool gamma_n_sign() const { return m_gamma_n_sign; }
        uint64_t gamma_n_value() const { return m_gamma_n_value; }
        bool not_used() const { return m_not_used; }
        uint64_t p() const { return m_p; }
        bool l_n() const { return m_l_n; }
        bool z_vel_sign() const { return m_z_vel_sign; }
        uint64_t z_vel_value() const { return m_z_vel_value; }
        bool z_accel_sign() const { return m_z_accel_sign; }
        uint64_t z_accel_value() const { return m_z_accel_value; }
        bool z_sign() const { return m_z_sign; }
        uint64_t z_value() const { return m_z_value; }
        glonass_t* _root() const { return m__root; }
        glonass_t* _parent() const { return m__parent; }
    };

private:
    bool m_idle_chip;
    uint64_t m_string_number;
    kaitai::kstruct* m_data;
    uint64_t m_hamming_code;
    uint64_t m_pad_1;
    uint64_t m_superframe_number;
    uint64_t m_pad_2;
    uint64_t m_frame_number;
    glonass_t* m__root;
    kaitai::kstruct* m__parent;

public:
    bool idle_chip() const { return m_idle_chip; }
    uint64_t string_number() const { return m_string_number; }
    kaitai::kstruct* data() const { return m_data; }
    uint64_t hamming_code() const { return m_hamming_code; }
    uint64_t pad_1() const { return m_pad_1; }
    uint64_t superframe_number() const { return m_superframe_number; }
    uint64_t pad_2() const { return m_pad_2; }
    uint64_t frame_number() const { return m_frame_number; }
    glonass_t* _root() const { return m__root; }
    kaitai::kstruct* _parent() const { return m__parent; }
};

#endif  // GLONASS_H_
