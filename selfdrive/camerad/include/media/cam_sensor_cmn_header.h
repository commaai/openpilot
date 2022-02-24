/* Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 and
 * only version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#ifndef _CAM_SENSOR_CMN_HEADER_
#define _CAM_SENSOR_CMN_HEADER_

#include <stdbool.h>
#include <media/cam_sensor.h>
#include <media/cam_req_mgr.h>

#define MAX_REGULATOR 5
#define MAX_POWER_CONFIG 12

#define MAX_PER_FRAME_ARRAY 32
#define BATCH_SIZE_MAX      16

#define CAM_SENSOR_NAME    "cam-sensor"
#define CAM_ACTUATOR_NAME  "cam-actuator"
#define CAM_CSIPHY_NAME    "cam-csiphy"
#define CAM_FLASH_NAME     "cam-flash"
#define CAM_EEPROM_NAME    "cam-eeprom"
#define CAM_OIS_NAME       "cam-ois"

#define MAX_SYSTEM_PIPELINE_DELAY 2

#define CAM_PKT_NOP_OPCODE 127

enum camera_sensor_cmd_type {
	CAMERA_SENSOR_CMD_TYPE_INVALID,
	CAMERA_SENSOR_CMD_TYPE_PROBE,
	CAMERA_SENSOR_CMD_TYPE_PWR_UP,
	CAMERA_SENSOR_CMD_TYPE_PWR_DOWN,
	CAMERA_SENSOR_CMD_TYPE_I2C_INFO,
	CAMERA_SENSOR_CMD_TYPE_I2C_RNDM_WR,
	CAMERA_SENSOR_CMD_TYPE_I2C_RNDM_RD,
	CAMERA_SENSOR_CMD_TYPE_I2C_CONT_WR,
	CAMERA_SENSOR_CMD_TYPE_I2C_CONT_RD,
	CAMERA_SENSOR_CMD_TYPE_WAIT,
	CAMERA_SENSOR_FLASH_CMD_TYPE_INIT_INFO,
	CAMERA_SENSOR_FLASH_CMD_TYPE_FIRE,
	CAMERA_SENSOR_FLASH_CMD_TYPE_RER,
	CAMERA_SENSOR_FLASH_CMD_TYPE_QUERYCURR,
	CAMERA_SENSOR_FLASH_CMD_TYPE_WIDGET,
	CAMERA_SENSOR_CMD_TYPE_RD_DATA,
	CAMERA_SENSOR_FLASH_CMD_TYPE_INIT_FIRE,
	CAMERA_SENSOR_CMD_TYPE_MAX,
};

enum camera_sensor_i2c_op_code {
	CAMERA_SENSOR_I2C_OP_INVALID,
	CAMERA_SENSOR_I2C_OP_RNDM_WR,
	CAMERA_SENSOR_I2C_OP_RNDM_WR_VERF,
	CAMERA_SENSOR_I2C_OP_CONT_WR_BRST,
	CAMERA_SENSOR_I2C_OP_CONT_WR_BRST_VERF,
	CAMERA_SENSOR_I2C_OP_CONT_WR_SEQN,
	CAMERA_SENSOR_I2C_OP_CONT_WR_SEQN_VERF,
	CAMERA_SENSOR_I2C_OP_MAX,
};

enum camera_sensor_wait_op_code {
	CAMERA_SENSOR_WAIT_OP_INVALID,
	CAMERA_SENSOR_WAIT_OP_COND,
	CAMERA_SENSOR_WAIT_OP_HW_UCND,
	CAMERA_SENSOR_WAIT_OP_SW_UCND,
	CAMERA_SENSOR_WAIT_OP_MAX,
};

enum camera_flash_opcode {
	CAMERA_SENSOR_FLASH_OP_INVALID,
	CAMERA_SENSOR_FLASH_OP_OFF,
	CAMERA_SENSOR_FLASH_OP_FIRELOW,
	CAMERA_SENSOR_FLASH_OP_FIREHIGH,
	CAMERA_SENSOR_FLASH_OP_MAX,
};

enum camera_sensor_i2c_type {
	CAMERA_SENSOR_I2C_TYPE_INVALID,
	CAMERA_SENSOR_I2C_TYPE_BYTE,
	CAMERA_SENSOR_I2C_TYPE_WORD,
	CAMERA_SENSOR_I2C_TYPE_3B,
	CAMERA_SENSOR_I2C_TYPE_DWORD,
	CAMERA_SENSOR_I2C_TYPE_MAX,
};

enum i2c_freq_mode {
	I2C_STANDARD_MODE,
	I2C_FAST_MODE,
	I2C_CUSTOM_MODE,
	I2C_FAST_PLUS_MODE,
	I2C_MAX_MODES,
};

enum position_roll {
	ROLL_0       = 0,
	ROLL_90      = 90,
	ROLL_180     = 180,
	ROLL_270     = 270,
	ROLL_INVALID = 360,
};

enum position_yaw {
	FRONT_CAMERA_YAW = 0,
	REAR_CAMERA_YAW  = 180,
	INVALID_YAW      = 360,
};

enum position_pitch {
	LEVEL_PITCH    = 0,
	INVALID_PITCH  = 360,
};

enum sensor_sub_module {
	SUB_MODULE_SENSOR,
	SUB_MODULE_ACTUATOR,
	SUB_MODULE_EEPROM,
	SUB_MODULE_LED_FLASH,
	SUB_MODULE_CSID,
	SUB_MODULE_CSIPHY,
	SUB_MODULE_OIS,
	SUB_MODULE_EXT,
	SUB_MODULE_MAX,
};

enum msm_camera_power_seq_type {
	SENSOR_MCLK,
	SENSOR_VANA,
	SENSOR_VDIG,
	SENSOR_VIO,
	SENSOR_VAF,
	SENSOR_VAF_PWDM,
	SENSOR_CUSTOM_REG1,
	SENSOR_CUSTOM_REG2,
	SENSOR_RESET,
	SENSOR_STANDBY,
	SENSOR_CUSTOM_GPIO1,
	SENSOR_CUSTOM_GPIO2,
	SENSOR_SEQ_TYPE_MAX,
};

enum cam_sensor_packet_opcodes {
	CAM_SENSOR_PACKET_OPCODE_SENSOR_STREAMON,
	CAM_SENSOR_PACKET_OPCODE_SENSOR_UPDATE,
	CAM_SENSOR_PACKET_OPCODE_SENSOR_INITIAL_CONFIG,
	CAM_SENSOR_PACKET_OPCODE_SENSOR_PROBE,
	CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG,
	CAM_SENSOR_PACKET_OPCODE_SENSOR_STREAMOFF,
	CAM_SENSOR_PACKET_OPCODE_SENSOR_NOP = 127
};

enum cam_actuator_packet_opcodes {
	CAM_ACTUATOR_PACKET_OPCODE_INIT,
	CAM_ACTUATOR_PACKET_AUTO_MOVE_LENS,
	CAM_ACTUATOR_PACKET_MANUAL_MOVE_LENS
};

enum cam_eeprom_packet_opcodes {
	CAM_EEPROM_PACKET_OPCODE_INIT
};

enum cam_ois_packet_opcodes {
	CAM_OIS_PACKET_OPCODE_INIT,
	CAM_OIS_PACKET_OPCODE_OIS_CONTROL
};

enum msm_bus_perf_setting {
	S_INIT,
	S_PREVIEW,
	S_VIDEO,
	S_CAPTURE,
	S_ZSL,
	S_STEREO_VIDEO,
	S_STEREO_CAPTURE,
	S_DEFAULT,
	S_LIVESHOT,
	S_DUAL,
	S_EXIT
};

enum msm_camera_device_type_t {
	MSM_CAMERA_I2C_DEVICE,
	MSM_CAMERA_PLATFORM_DEVICE,
	MSM_CAMERA_SPI_DEVICE,
};

enum cam_flash_device_type {
	CAMERA_FLASH_DEVICE_TYPE_PMIC = 0,
	CAMERA_FLASH_DEVICE_TYPE_I2C,
	CAMERA_FLASH_DEVICE_TYPE_GPIO,
};

enum cci_i2c_master_t {
	MASTER_0,
	MASTER_1,
	MASTER_MAX,
};

enum camera_vreg_type {
	VREG_TYPE_DEFAULT,
	VREG_TYPE_CUSTOM,
};

enum cam_sensor_i2c_cmd_type {
	CAM_SENSOR_I2C_WRITE_RANDOM,
	CAM_SENSOR_I2C_WRITE_BURST,
	CAM_SENSOR_I2C_WRITE_SEQ,
	CAM_SENSOR_I2C_READ,
	CAM_SENSOR_I2C_POLL
};

struct common_header {
	uint16_t    first_word;
	uint8_t     third_byte;
	uint8_t     cmd_type;
};

struct camera_vreg_t {
	const char *reg_name;
	int min_voltage;
	int max_voltage;
	int op_mode;
	uint32_t delay;
	const char *custom_vreg_name;
	enum camera_vreg_type type;
};

struct msm_camera_gpio_num_info {
	uint16_t gpio_num[SENSOR_SEQ_TYPE_MAX];
	uint8_t valid[SENSOR_SEQ_TYPE_MAX];
};

struct msm_cam_clk_info {
	const char *clk_name;
	long clk_rate;
	uint32_t delay;
};

struct msm_pinctrl_info {
	struct pinctrl *pinctrl;
	struct pinctrl_state *gpio_state_active;
	struct pinctrl_state *gpio_state_suspend;
	bool use_pinctrl;
};

struct cam_sensor_i2c_reg_array {
	uint32_t reg_addr;
	uint32_t reg_data;
	uint32_t delay;
	uint32_t data_mask;
};

struct cam_sensor_i2c_reg_setting {
	struct cam_sensor_i2c_reg_array *reg_setting;
	unsigned short size;
	enum camera_sensor_i2c_type addr_type;
	enum camera_sensor_i2c_type data_type;
	unsigned short delay;
};

/*struct i2c_settings_list {
	struct cam_sensor_i2c_reg_setting i2c_settings;
	enum cam_sensor_i2c_cmd_type op_code;
	struct list_head list;
};

struct i2c_settings_array {
	struct list_head list_head;
	int32_t is_settings_valid;
	int64_t request_id;
};

struct i2c_data_settings {
	struct i2c_settings_array init_settings;
	struct i2c_settings_array config_settings;
	struct i2c_settings_array streamon_settings;
	struct i2c_settings_array streamoff_settings;
	struct i2c_settings_array *per_frame;
};*/

struct cam_sensor_power_ctrl_t {
	struct device *dev;
	struct cam_sensor_power_setting *power_setting;
	uint16_t power_setting_size;
	struct cam_sensor_power_setting *power_down_setting;
	uint16_t power_down_setting_size;
	struct msm_camera_gpio_num_info *gpio_num_info;
	struct msm_pinctrl_info pinctrl_info;
	uint8_t cam_pinctrl_status;
};

struct cam_camera_slave_info {
	uint16_t sensor_slave_addr;
	uint16_t sensor_id_reg_addr;
	uint16_t sensor_id;
	uint16_t sensor_id_mask;
};

struct msm_sensor_init_params {
	int modes_supported;
	unsigned int sensor_mount_angle;
};

enum msm_sensor_camera_id_t {
	CAMERA_0,
	CAMERA_1,
	CAMERA_2,
	CAMERA_3,
	CAMERA_4,
	CAMERA_5,
	CAMERA_6,
	MAX_CAMERAS,
};

struct msm_sensor_id_info_t {
	unsigned short sensor_id_reg_addr;
	unsigned short sensor_id;
	unsigned short sensor_id_mask;
};

enum msm_sensor_output_format_t {
	MSM_SENSOR_BAYER,
	MSM_SENSOR_YCBCR,
	MSM_SENSOR_META,
};

struct cam_sensor_power_setting {
	enum msm_camera_power_seq_type seq_type;
	unsigned short seq_val;
	long config_val;
	unsigned short delay;
	void *data[10];
};

struct cam_sensor_board_info {
	struct cam_camera_slave_info slave_info;
	int32_t sensor_mount_angle;
	int32_t secure_mode;
	int modes_supported;
	int32_t pos_roll;
	int32_t pos_yaw;
	int32_t pos_pitch;
	int32_t  subdev_id[SUB_MODULE_MAX];
	int32_t  subdev_intf[SUB_MODULE_MAX];
	const char *misc_regulator;
	struct cam_sensor_power_ctrl_t power_info;
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

struct msm_camera_gpio_conf {
	void *cam_gpiomux_conf_tbl;
	uint8_t cam_gpiomux_conf_tbl_size;
	struct gpio *cam_gpio_common_tbl;
	uint8_t cam_gpio_common_tbl_size;
	struct gpio *cam_gpio_req_tbl;
	uint8_t cam_gpio_req_tbl_size;
	uint32_t gpio_no_mux;
	uint32_t *camera_off_table;
	uint8_t camera_off_table_size;
	uint32_t *camera_on_table;
	uint8_t camera_on_table_size;
	struct msm_camera_gpio_num_info *gpio_num_info;
};

/*for tof camera  Begin*/
enum EEPROM_DATA_OP_T{
	EEPROM_DEFAULT_DATA = 0,
	EEPROM_INIT_DATA,
	EEPROM_CONFIG_DATA,
	EEPROM_STREAMON_DATA,
	EEPROM_STREAMOFF_DATA,
	EEPROM_OTHER_DATA,
};
/*for tof camera  End*/
#endif /* _CAM_SENSOR_CMN_HEADER_ */
