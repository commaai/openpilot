#ifndef __LINUX_MSM_CAMSENSOR_SDK_H
#define __LINUX_MSM_CAMSENSOR_SDK_H

#define KVERSION 0x1

#define MAX_POWER_CONFIG      12
#define GPIO_OUT_LOW          (0 << 1)
#define GPIO_OUT_HIGH         (1 << 1)
#define CSI_EMBED_DATA        0x12
#define CSI_RESERVED_DATA_0   0x13
#define CSI_YUV422_8          0x1E
#define CSI_RAW8              0x2A
#define CSI_RAW10             0x2B
#define CSI_RAW12             0x2C
#define CSI_DECODE_6BIT         0
#define CSI_DECODE_8BIT         1
#define CSI_DECODE_10BIT        2
#define CSI_DECODE_12BIT        3
#define CSI_DECODE_DPCM_10_8_10 5
#define MAX_CID                 16
#define I2C_SEQ_REG_DATA_MAX    1024
#define I2C_REG_DATA_MAX       (8*1024)

#define MSM_V4L2_PIX_FMT_META v4l2_fourcc('M', 'E', 'T', 'A') /* META */
#define MSM_V4L2_PIX_FMT_SBGGR14 v4l2_fourcc('B', 'G', '1', '4')
	/* 14  BGBG.. GRGR.. */
#define MSM_V4L2_PIX_FMT_SGBRG14 v4l2_fourcc('G', 'B', '1', '4')
	/* 14  GBGB.. RGRG.. */
#define MSM_V4L2_PIX_FMT_SGRBG14 v4l2_fourcc('B', 'A', '1', '4')
	/* 14  GRGR.. BGBG.. */
#define MSM_V4L2_PIX_FMT_SRGGB14 v4l2_fourcc('R', 'G', '1', '4')
	/* 14  RGRG.. GBGB.. */

#define MAX_ACTUATOR_REG_TBL_SIZE 8
#define MAX_ACTUATOR_REGION       5
#define NUM_ACTUATOR_DIR          2
#define MAX_ACTUATOR_SCENARIO     8
#define MAX_ACT_MOD_NAME_SIZE     32
#define MAX_ACT_NAME_SIZE         32
#define MAX_ACTUATOR_INIT_SET     120
#define MAX_I2C_REG_SET           12

#define MAX_LED_TRIGGERS          3

#define MSM_EEPROM_MEMORY_MAP_MAX_SIZE  80
#define MSM_EEPROM_MAX_MEM_MAP_CNT      8

enum msm_sensor_camera_id_t {
	CAMERA_0,
	CAMERA_1,
	CAMERA_2,
	CAMERA_3,
	MAX_CAMERAS,
};

enum i2c_freq_mode_t {
	I2C_STANDARD_MODE,
	I2C_FAST_MODE,
	I2C_CUSTOM_MODE,
	I2C_FAST_PLUS_MODE,
	I2C_MAX_MODES,
};

enum camb_position_t {
	BACK_CAMERA_B,
	FRONT_CAMERA_B,
	AUX_CAMERA_B = 0x100,
	INVALID_CAMERA_B,
};

enum msm_sensor_power_seq_type_t {
	SENSOR_CLK,
	SENSOR_GPIO,
	SENSOR_VREG,
	SENSOR_I2C_MUX,
	SENSOR_I2C,
};

enum msm_camera_i2c_reg_addr_type {
	MSM_CAMERA_I2C_BYTE_ADDR = 1,
	MSM_CAMERA_I2C_WORD_ADDR,
	MSM_CAMERA_I2C_3B_ADDR,
	MSM_CAMERA_I2C_ADDR_TYPE_MAX,
};

enum msm_camera_i2c_data_type {
	MSM_CAMERA_I2C_BYTE_DATA = 1,
	MSM_CAMERA_I2C_WORD_DATA,
	MSM_CAMERA_I2C_DWORD_DATA,
	MSM_CAMERA_I2C_SET_BYTE_MASK,
	MSM_CAMERA_I2C_UNSET_BYTE_MASK,
	MSM_CAMERA_I2C_SET_WORD_MASK,
	MSM_CAMERA_I2C_UNSET_WORD_MASK,
	MSM_CAMERA_I2C_SET_BYTE_WRITE_MASK_DATA,
	MSM_CAMERA_I2C_SEQ_DATA,
	MSM_CAMERA_I2C_DATA_TYPE_MAX,
};

enum msm_sensor_power_seq_gpio_t {
	SENSOR_GPIO_RESET,
	SENSOR_GPIO_STANDBY,
	SENSOR_GPIO_AF_PWDM,
	SENSOR_GPIO_VIO,
	SENSOR_GPIO_VANA,
	SENSOR_GPIO_VDIG,
	SENSOR_GPIO_VAF,
	SENSOR_GPIO_FL_EN,
	SENSOR_GPIO_FL_NOW,
	SENSOR_GPIO_FL_RESET,
	SENSOR_GPIO_CUSTOM1,
	SENSOR_GPIO_CUSTOM2,
	SENSOR_GPIO_MAX,
};

enum msm_camera_vreg_name_t {
	CAM_VDIG,
	CAM_VIO,
	CAM_VANA,
	CAM_VAF,
	CAM_V_CUSTOM1,
	CAM_V_CUSTOM2,
	CAM_VREG_MAX,
};

enum msm_sensor_clk_type_t {
	SENSOR_CAM_MCLK,
	SENSOR_CAM_CLK,
	SENSOR_CAM_CLK_MAX,
};

enum camerab_mode_t {
	CAMERA_MODE_2D_B = (1<<0),
	CAMERA_MODE_3D_B = (1<<1),
	CAMERA_MODE_INVALID = (1<<2),
};

enum msm_actuator_data_type {
	MSM_ACTUATOR_BYTE_DATA = 1,
	MSM_ACTUATOR_WORD_DATA,
};

enum msm_actuator_addr_type {
	MSM_ACTUATOR_BYTE_ADDR = 1,
	MSM_ACTUATOR_WORD_ADDR,
};

enum msm_actuator_write_type {
	MSM_ACTUATOR_WRITE_HW_DAMP,
	MSM_ACTUATOR_WRITE_DAC,
	MSM_ACTUATOR_WRITE,
	MSM_ACTUATOR_WRITE_DIR_REG,
	MSM_ACTUATOR_POLL,
	MSM_ACTUATOR_READ_WRITE,
};

enum msm_actuator_i2c_operation {
	MSM_ACT_WRITE = 0,
	MSM_ACT_POLL,
};

enum actuator_type {
	ACTUATOR_VCM,
	ACTUATOR_PIEZO,
	ACTUATOR_HVCM,
	ACTUATOR_BIVCM,
};

enum msm_flash_driver_type {
	FLASH_DRIVER_PMIC,
	FLASH_DRIVER_I2C,
	FLASH_DRIVER_GPIO,
	FLASH_DRIVER_DEFAULT
};

enum msm_flash_cfg_type_t {
	CFG_FLASH_INIT,
	CFG_FLASH_RELEASE,
	CFG_FLASH_OFF,
	CFG_FLASH_LOW,
	CFG_FLASH_HIGH,
};

enum msm_sensor_output_format_t {
	MSM_SENSOR_BAYER,
	MSM_SENSOR_YCBCR,
	MSM_SENSOR_META,
};

struct msm_sensor_power_setting {
	enum msm_sensor_power_seq_type_t seq_type;
	unsigned short seq_val;
	long config_val;
	unsigned short delay;
	void *data[10];
};

struct msm_sensor_power_setting_array {
	struct msm_sensor_power_setting  power_setting_a[MAX_POWER_CONFIG];
	struct msm_sensor_power_setting *power_setting;
	unsigned short size;
	struct msm_sensor_power_setting  power_down_setting_a[MAX_POWER_CONFIG];
	struct msm_sensor_power_setting *power_down_setting;
	unsigned short size_down;
};

enum msm_camera_i2c_operation {
	MSM_CAM_WRITE = 0,
	MSM_CAM_POLL,
	MSM_CAM_READ,
};

struct msm_sensor_i2c_sync_params {
	unsigned int cid;
	int csid;
	unsigned short line;
	unsigned short delay;
};

struct msm_camera_reg_settings_t {
	uint16_t reg_addr;
	enum msm_camera_i2c_reg_addr_type addr_type;
	uint16_t reg_data;
	enum msm_camera_i2c_data_type data_type;
	enum msm_camera_i2c_operation i2c_operation;
	uint16_t delay;
};

struct msm_eeprom_mem_map_t {
	int slave_addr;
	struct msm_camera_reg_settings_t
		mem_settings[MSM_EEPROM_MEMORY_MAP_MAX_SIZE];
	int memory_map_size;
};

struct msm_eeprom_memory_map_array {
	struct msm_eeprom_mem_map_t memory_map[MSM_EEPROM_MAX_MEM_MAP_CNT];
	uint32_t msm_size_of_max_mappings;
};

struct msm_sensor_init_params {
	/* mask of modes supported: 2D, 3D */
	int                 modes_supported;
	/* sensor position: front, back */
	enum camb_position_t position;
	/* sensor mount angle */
	unsigned int            sensor_mount_angle;
};

struct msm_sensor_id_info_t {
	unsigned short sensor_id_reg_addr;
	unsigned short sensor_id;
	unsigned short sensor_id_mask;
  // added in LeEco
  unsigned char module_id;
  unsigned char vcm_id;
};

struct msm_camera_sensor_slave_info {
	char sensor_name[32];
	char eeprom_name[32];
	char actuator_name[32];
	char ois_name[32];
	char flash_name[32];
	enum msm_sensor_camera_id_t camera_id;
	unsigned short slave_addr;
	enum i2c_freq_mode_t i2c_freq_mode;
	enum msm_camera_i2c_reg_addr_type addr_type;
	struct msm_sensor_id_info_t sensor_id_info;
	struct msm_sensor_power_setting_array power_setting_array;
	unsigned char  is_init_params_valid;
	struct msm_sensor_init_params sensor_init_params;
	enum msm_sensor_output_format_t output_format;
};

struct msm_camera_i2c_reg_array {
	unsigned short reg_addr;
	unsigned short reg_data;
	unsigned int delay;
};

struct msm_camera_i2c_reg_setting {
	struct msm_camera_i2c_reg_array *reg_setting;
	unsigned short size;
	enum msm_camera_i2c_reg_addr_type addr_type;
	enum msm_camera_i2c_data_type data_type;
	unsigned short delay;
};

struct msm_camera_csid_vc_cfg {
	unsigned char cid;
	unsigned char dt;
	unsigned char decode_format;
};

struct msm_camera_csid_lut_params {
	unsigned char num_cid;
	struct msm_camera_csid_vc_cfg vc_cfg_a[MAX_CID];
	struct msm_camera_csid_vc_cfg *vc_cfg[MAX_CID];
};

struct msm_camera_csid_params {
	unsigned char lane_cnt;
	unsigned short lane_assign;
	unsigned char phy_sel;
	unsigned int csi_clk;
	struct msm_camera_csid_lut_params lut_params;
	unsigned char csi_3p_sel;
};

struct msm_camera_csid_testmode_parms {
	unsigned int num_bytes_per_line;
	unsigned int num_lines;
	unsigned int h_blanking_count;
	unsigned int v_blanking_count;
	unsigned int payload_mode;
};

struct msm_camera_csiphy_params {
	unsigned char lane_cnt;
	unsigned char settle_cnt;
	unsigned short lane_mask;
	unsigned char combo_mode;
	unsigned char csid_core;
	unsigned int csiphy_clk;
	unsigned char csi_3phase;
};

struct msm_camera_i2c_seq_reg_array {
	unsigned short reg_addr;
	unsigned char reg_data[I2C_SEQ_REG_DATA_MAX];
	unsigned short reg_data_size;
};

struct msm_camera_i2c_seq_reg_setting {
	struct msm_camera_i2c_seq_reg_array *reg_setting;
	unsigned short size;
	enum msm_camera_i2c_reg_addr_type addr_type;
	unsigned short delay;
};

struct msm_actuator_reg_params_t {
	enum msm_actuator_write_type reg_write_type;
	unsigned int hw_mask;
	unsigned short reg_addr;
	unsigned short hw_shift;
	unsigned short data_shift;
	unsigned short data_type;
	unsigned short addr_type;
	unsigned short reg_data;
	unsigned short delay;
};


struct damping_params_t {
	unsigned int damping_step;
	unsigned int damping_delay;
	unsigned int hw_params;
};

struct region_params_t {
	/* [0] = ForwardDirection Macro boundary
	   [1] = ReverseDirection Inf boundary
	*/
	unsigned short step_bound[2];
	unsigned short code_per_step;
	/* qvalue for converting float type numbers to integer format */
	unsigned int qvalue;
};

struct reg_settings_t {
	unsigned short reg_addr;
	enum msm_actuator_addr_type addr_type;
	unsigned short reg_data;
	enum msm_actuator_data_type data_type;
	enum msm_actuator_i2c_operation i2c_operation;
	unsigned int delay;
};

struct msm_camera_i2c_reg_setting_array {
	struct msm_camera_i2c_reg_array reg_setting_a[MAX_I2C_REG_SET];
	unsigned short size;
	enum msm_camera_i2c_reg_addr_type addr_type;
	enum msm_camera_i2c_data_type data_type;
	unsigned short delay;
};
#endif /* __LINUX_MSM_CAM_SENSOR_H */
