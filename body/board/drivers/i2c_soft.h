#define SW_I2C_WAIT_TIME  22	

#define I2C_READ       0x01
#define READ_CMD       1
#define WRITE_CMD      0


void SW_I2C_init(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;

  GPIO_InitStructure.Speed = GPIO_SPEED_FREQ_HIGH;
  GPIO_InitStructure.Mode = GPIO_MODE_OUTPUT_PP;
  
  GPIO_InitStructure.Pin = SW_I2C1_SCL_PIN;
  HAL_GPIO_Init(SW_I2C1_SCL_GPIO, &GPIO_InitStructure);
  GPIO_InitStructure.Pin   = SW_I2C1_SDA_PIN;
  HAL_GPIO_Init(SW_I2C1_SDA_GPIO, &GPIO_InitStructure);
}

uint8_t SW_I2C_ReadVal_SDA(void)
{
  uint8_t ret;
  ret = (uint16_t)HAL_GPIO_ReadPin(SW_I2C1_SDA_GPIO, SW_I2C1_SDA_PIN);
  return ret;
}
  
void sda_high(void)
{
  HAL_GPIO_WritePin(SW_I2C1_SDA_GPIO, SW_I2C1_SDA_PIN, GPIO_PIN_SET);
}


void sda_low(void)
{
  HAL_GPIO_WritePin(SW_I2C1_SDA_GPIO, SW_I2C1_SDA_PIN, GPIO_PIN_RESET);
}


void scl_high(void)
{
  HAL_GPIO_WritePin(SW_I2C1_SCL_GPIO, SW_I2C1_SCL_PIN, GPIO_PIN_SET);
}


void scl_low(void)
{
  HAL_GPIO_WritePin(SW_I2C1_SCL_GPIO, SW_I2C1_SCL_PIN, GPIO_PIN_RESET);
}

void sda_out(uint8_t out)
{
  if (out) {
    sda_high();
  } else {
    sda_low();
  }
}

void sda_in_mode(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;

  GPIO_InitStructure.Speed=GPIO_SPEED_FREQ_HIGH;
  GPIO_InitStructure.Mode=GPIO_MODE_INPUT;
  
  GPIO_InitStructure.Pin   = SW_I2C1_SDA_PIN;
  HAL_GPIO_Init(SW_I2C1_SDA_GPIO, &GPIO_InitStructure);
}

void sda_out_mode(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;

  GPIO_InitStructure.Speed=GPIO_SPEED_FREQ_HIGH;
  GPIO_InitStructure.Mode=GPIO_MODE_OUTPUT_OD;
  
  GPIO_InitStructure.Pin   = SW_I2C1_SDA_PIN;
  HAL_GPIO_Init(SW_I2C1_SDA_GPIO, &GPIO_InitStructure);
}

void i2c_clk_data_out(void)
{
  scl_high();
  delay(SW_I2C_WAIT_TIME);
  scl_low();
}

void i2c_port_init(void)
{
  sda_high();
  scl_high();
}

void i2c_start_condition(void)
{
  sda_high();
  scl_high();

  delay(SW_I2C_WAIT_TIME);
  sda_low();
  delay(SW_I2C_WAIT_TIME);
  scl_low();

  delay(SW_I2C_WAIT_TIME << 1);
}

void i2c_stop_condition(void)
{
  sda_low();
  scl_high();

  delay(SW_I2C_WAIT_TIME);
  sda_high();
  delay(SW_I2C_WAIT_TIME);
}

uint8_t i2c_check_ack(void)
{
  uint8_t ack;
  int i;
  unsigned int temp;

  sda_in_mode();

  scl_high();

  ack = 0;
  delay(SW_I2C_WAIT_TIME);

  for (i = 10; i > 0; i--) {
    temp = !(SW_I2C_ReadVal_SDA());	
    if (temp)	
    {
      ack = 1;
      break;
    }
  }
  scl_low();
  sda_out_mode();	

  delay(SW_I2C_WAIT_TIME);
  return ack;
}

void i2c_check_not_ack(void)
{
  sda_in_mode();
  i2c_clk_data_out();
  sda_out_mode();
  delay(SW_I2C_WAIT_TIME);
}

void i2c_slave_address(uint8_t IICID, uint8_t readwrite)
{
  int x;

  if (readwrite) {
    IICID |= I2C_READ;
  } else {
    IICID &= ~I2C_READ;
  }

  scl_low();

  for (x = 7; x >= 0; x--) {
    sda_out(IICID & (1 << x));
    delay(SW_I2C_WAIT_TIME);
    i2c_clk_data_out();
  }
}

void i2c_register_address(uint8_t addr)
{
  int x;

  scl_low();

  for (x = 7; x >= 0; x--) {
    sda_out(addr & (1 << x));
    delay(SW_I2C_WAIT_TIME);
    i2c_clk_data_out();
  }
}

void i2c_send_ack(void)
{
  sda_out_mode();
  sda_low();

  delay(SW_I2C_WAIT_TIME);
  scl_high();

  delay(SW_I2C_WAIT_TIME << 1);

  sda_low();
  delay(SW_I2C_WAIT_TIME << 1);

  scl_low();

  sda_out_mode();

  delay(SW_I2C_WAIT_TIME);
}

void SW_I2C_Write_Data(uint8_t data)
{
  int x;

  scl_low();

  for (x = 7; x >= 0; x--) {
    sda_out(data & (1 << x));
    delay(SW_I2C_WAIT_TIME);
    i2c_clk_data_out();
  }
}

uint8_t SW_I2C_Read_Data(void)
{
  int x;
  uint8_t readdata = 0;

  sda_in_mode();

  for (x = 8; x--;) {
    scl_high();

    readdata <<= 1;
    if (SW_I2C_ReadVal_SDA()) { readdata |= 0x01; }

    delay(SW_I2C_WAIT_TIME);
    scl_low();

    delay(SW_I2C_WAIT_TIME);
  }

  sda_out_mode();
  return readdata;
}

uint8_t SW_I2C_WriteControl_8Bit(uint8_t IICID, uint8_t regaddr, uint8_t data)
{
  uint8_t returnack = true;

  i2c_start_condition();

  i2c_slave_address(IICID, WRITE_CMD);
  if (!i2c_check_ack()) { returnack = false; }

  delay(SW_I2C_WAIT_TIME);

  i2c_register_address(regaddr);
  if (!i2c_check_ack()) { returnack = false; }

  delay(SW_I2C_WAIT_TIME);

  SW_I2C_Write_Data(data);
  if (!i2c_check_ack()) { returnack = false; }

  delay(SW_I2C_WAIT_TIME);

  i2c_stop_condition();

  return returnack;
}

uint8_t SW_I2C_Multi_ReadnControl_8Bit(uint8_t IICID, uint8_t regaddr, uint8_t rcnt, uint8_t (*pdata))
{
  uint8_t returnack = true;
  uint8_t index;

  i2c_port_init();

  i2c_start_condition();

  i2c_slave_address(IICID, WRITE_CMD);
  if (!i2c_check_ack()) { returnack = false; }

  delay(SW_I2C_WAIT_TIME);

  i2c_register_address(regaddr);
  if (!i2c_check_ack()) { returnack = false; }

  delay(SW_I2C_WAIT_TIME);

  i2c_start_condition();

  i2c_slave_address(IICID, READ_CMD);
  if (!i2c_check_ack()) { returnack = false; }

  for ( index = 0 ; index < (rcnt-1) ; index++) {
    delay(SW_I2C_WAIT_TIME);
    pdata[index] = SW_I2C_Read_Data();
    i2c_send_ack();
  }

  pdata[rcnt-1] = SW_I2C_Read_Data();

  i2c_check_not_ack();

  i2c_stop_condition();

  return returnack;
}
