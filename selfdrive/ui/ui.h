#ifndef UIC_DECLARATIONS
    #define UIC_DECLARATIONS
    // Calibration status values from controlsd.py
    #define CALIBRATION_UNCALIBRATED 0
    #define CALIBRATION_CALIBRATED 1
    #define CALIBRATION_INVALID 2

    #define STATUS_STOPPED 0
    #define STATUS_DISENGAGED 1
    #define STATUS_ENGAGED 2
    #define STATUS_WARNING 3
    #define STATUS_ALERT 4
    #define STATUS_MAX 5

    #define ALERTSIZE_NONE 0
    #define ALERTSIZE_SMALL 1
    #define ALERTSIZE_MID 2
    #define ALERTSIZE_FULL 3

    #define UI_BUF_COUNT 4

    extern const int vwp_w;
    extern const int vwp_h;
    extern const int nav_w;
    extern const int nav_ww;
    extern const int sbr_w;
    extern const int bdr_s;
    extern const int box_x;
    extern const int box_y;
    extern const int box_w;
    extern const int box_h;
    extern const int viz_w;
    extern const int header_h;


    extern const uint8_t bg_colors[][4];

    extern const uint8_t alert_colors[][4];

    extern const int alert_sizes[];



    typedef struct UICstmButton {
        char btn_name[6];
        char btn_label[6];
        char btn_label2[11];
    } UICstmButton;

    typedef struct UIScene {
        int frontview;

        int transformed_width, transformed_height;

        uint64_t model_ts;
        ModelData model;

        float mpc_x[50];
        float mpc_y[50];

        bool world_objects_visible;
        mat3 warp_matrix;           // transformed box -> frame.
        mat4 extrinsic_matrix;      // Last row is 0 so we can use mat4.

        float v_cruise;
        uint64_t v_cruise_update_ts;
        float v_ego;
        float curvature;
        int engaged;
        bool engageable;

        bool uilayout_sidebarcollapsed;
        bool uilayout_mapenabled;
        // responsive layout
        int ui_viz_rx;
        int ui_viz_rw;
        int ui_viz_ro;

        int lead_status;
        float lead_d_rel, lead_y_rel, lead_v_rel;

        int front_box_x, front_box_y, front_box_width, front_box_height;

        uint64_t alert_ts;
        char alert_text1[1024];
        char alert_text2[1024];
        uint8_t alert_size;
        float alert_blinkingrate;

        float awareness_status;

        uint64_t started_ts;
        //BB CPU TEMP
        uint16_t maxCpuTemp;
        uint32_t maxBatTemp;
        float gpsAccuracy ;
        float freeSpace;
        float angleSteers;
        float angleSteersDes;
        //BB END CPU TEMP
        // Used to display calibration progress
        int cal_status;
        int cal_perc;
        // Used to show gps planner status
        bool gps_planner_active;

    } UIScene;

    typedef struct UIState {
        pthread_mutex_t lock;
        pthread_cond_t bg_cond;

        FramebufferState *fb;
        int fb_w, fb_h;
        EGLDisplay display;
        EGLSurface surface;

        NVGcontext *vg;
        //BB
        UICstmButton btns[6];
        char btns_status[6];
        char car_model[40];
        char car_folder[20];
        zsock_t *uiButtonInfo_sock;
        void *uiButtonInfo_sock_raw;
        zsock_t *uiCustomAlert_sock;
        void *uiCustomAlert_sock_raw;
        zsock_t *uiSetCar_sock;
        void *uiSetCar_sock_raw;
        zsock_t *uiPlaySound_sock;
        void *uiPlaySound_sock_raw;
        zsock_t *uiButtonStatus_sock;
        void *uiButtonStatus_sock_raw; 
        int btns_x[6];
        int btns_y[6];
        int btns_r[6];
        int custom_message_status;
        char custom_message[120];
        int img_logo;
        int img_logo2;
        int img_car;
        int tri_state_switch;
        long tri_state_switch_last_read;
        //BB END
        int font_courbd;
        int font_sans_regular;
        int font_sans_semibold;
        int font_sans_bold;
        int img_wheel;
        
        

        zsock_t *thermal_sock;
        void *thermal_sock_raw;
        zsock_t *model_sock;
        void *model_sock_raw;
        zsock_t *live100_sock;
        void *live100_sock_raw;
        zsock_t *livecalibration_sock;
        void *livecalibration_sock_raw;
        zsock_t *live20_sock;
        void *live20_sock_raw;
        zsock_t *livempc_sock;
        void *livempc_sock_raw;
        zsock_t *plus_sock;
        void *plus_sock_raw;
        zsock_t *gps_sock;
        void *gps_sock_raw;

        zsock_t *uilayout_sock;
        void *uilayout_sock_raw;

        int plus_state;

        // vision state
        bool vision_connected;
        bool vision_connect_firstrun;
        int ipc_fd;

        VIPCBuf bufs[UI_BUF_COUNT];
        VIPCBuf front_bufs[UI_BUF_COUNT];
        int cur_vision_idx;
        int cur_vision_front_idx;

        GLuint frame_program;
        GLuint frame_texs[UI_BUF_COUNT];
        GLuint frame_front_texs[UI_BUF_COUNT];

        GLint frame_pos_loc, frame_texcoord_loc;
        GLint frame_texture_loc, frame_transform_loc;

        GLuint line_program;
        GLint line_pos_loc, line_color_loc;
        GLint line_transform_loc;

        unsigned int rgb_width, rgb_height, rgb_stride;
        size_t rgb_buf_len;
        mat4 rgb_transform;

        unsigned int rgb_front_width, rgb_front_height, rgb_front_stride;
        size_t rgb_front_buf_len;

        bool intrinsic_matrix_loaded;
        mat3 intrinsic_matrix;

        UIScene scene;

        bool awake;
        int awake_timeout;

        int status;
        bool is_metric;
        bool passive;
        int alert_size;
        float alert_blinking_alpha;
        bool alert_blinked;
        bool acc_enabled;

        float light_sensor;
    } UIState;

#endif