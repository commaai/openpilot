# TODO: these are also defined in a header

# GPIO pin definitions
class GPIO:
  # both GPIO_STM_RST_N and GPIO_LTE_RST_N are misnamed, they are high to reset
  HUB_RST_N = 30
  UBLOX_RST_N = 32
  UBLOX_SAFEBOOT_N = 33
  GNSS_PWR_EN = 34 # SCHEMATIC LABEL: GPIO_UBLOX_PWR_EN

  STM_RST_N = 124
  STM_BOOT0 = 134
  STM_PWR_EN_N = 41 # because STM32H7 RST doesn't generate a full power-on-reset

  SIREN = 42
  SOM_ST_IO = 49

  LTE_RST_N = 50
  LTE_PWRKEY = 116
  LTE_BOOT = 52

  # GPIO_CAM0_DVDD_EN = /sys/kernel/debug/regulator/camera_rear_ldo
  CAM0_AVDD_EN = 8
  CAM0_RSTN = 9
  CAM1_RSTN = 7
  CAM2_RSTN = 12

  # Sensor interrupts
  BMX055_ACCEL_INT = 21
  BMX055_GYRO_INT = 23
  BMX055_MAGN_INT = 87
  LSM_INT = 84
