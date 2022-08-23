// Default addresses for AS5048B
#define AS5048_ADDRESS_LEFT 0x40
#define AS5048_ADDRESS_RIGHT 0x41
#define UNKNOWN_IMU 0x68

#define AS5048B_PROG_REG 0x03
#define AS5048B_ADDR_REG 0x15
#define AS5048B_ZEROMSB_REG 0x16 //bits 0..7
#define AS5048B_ZEROLSB_REG 0x17 //bits 0..5
#define AS5048B_GAIN_REG 0xFA
#define AS5048B_DIAG_REG 0xFB
#define AS5048B_MAGNMSB_REG 0xFC //bits 0..7
#define AS5048B_MAGNLSB_REG 0xFD //bits 0..5
#define AS5048B_ANGLMSB_REG 0xFE //bits 0..7
#define AS5048B_ANGLLSB_REG 0xFF //bits 0..5

extern I2C_HandleTypeDef hi2c1;

const uint8_t init_imu_regaddr[] = {0x76, 0x4c, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53};
const uint8_t init_imu_data[] =    {0x00, 0x12, 0x2f, 0x26, 0x67, 0x04, 0x00, 0x00};


void angle_sensor_read(uint16_t *sensor_angle) {
  uint8_t buf[2];
  if (HAL_I2C_Mem_Read(&hi2c1, (AS5048_ADDRESS_LEFT<<1), AS5048B_ANGLMSB_REG, I2C_MEMADD_SIZE_8BIT, buf, 2, 10) == HAL_OK) {
    sensor_angle[0] = (buf[0] << 6) | (buf[1] & 0x3F);
  }
  if (HAL_I2C_Mem_Read(&hi2c1, (AS5048_ADDRESS_RIGHT<<1), AS5048B_ANGLMSB_REG, I2C_MEMADD_SIZE_8BIT, buf, 2, 10) == HAL_OK) {
    sensor_angle[1] = (buf[0] << 6) | (buf[1] & 0x3F);
  }
}

void IMU_soft_init(void) {
  i2c_port_init();

  for (int8_t i = 3; i > 0; i--) {
    SW_I2C_WriteControl_8Bit((UNKNOWN_IMU<<1), 0x75, 0x00);
  }
  for (int8_t i = sizeof(init_imu_regaddr)-1; i >= 0; i--) {
    SW_I2C_WriteControl_8Bit((UNKNOWN_IMU<<1), init_imu_regaddr[i], init_imu_data[i]);
  }
}

void IMU_soft_sensor_read(uint16_t *unknown_imu_data) {
  static uint8_t buf[12];

  SW_I2C_WriteControl_8Bit((UNKNOWN_IMU<<1), 0x76, 0x00);
  SW_I2C_Multi_ReadnControl_8Bit((UNKNOWN_IMU<<1), 0x1F, 6, buf);
  SW_I2C_Multi_ReadnControl_8Bit((UNKNOWN_IMU<<1), 0x25, 6, &buf[6]);

  for (int8_t i = 5; i >= 0; i--) {
    unknown_imu_data[i] = (buf[i*2] << 8) | (buf[(i*2)+1]);
  }
}
