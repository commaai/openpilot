
#include "rtc_definitions.h"

#define YEAR_OFFSET 2000U

uint8_t to_bcd(uint16_t value){
  return (((value / 10U) & 0x0FU) << 4U) | ((value % 10U) & 0x0FU);
}

uint16_t from_bcd(uint8_t value){
  return (((value & 0xF0U) >> 4U) * 10U) + (value & 0x0FU);
}

void rtc_set_time(timestamp_t time){
  print("Setting RTC time\n");

  // Disable write protection
  disable_bdomain_protection();
  RTC->WPR = 0xCA;
  RTC->WPR = 0x53;

  // Enable initialization mode
  register_set_bits(&(RTC->ISR), RTC_ISR_INIT);
  while((RTC->ISR & RTC_ISR_INITF) == 0){}

  // Set time
  RTC->TR = (to_bcd(time.hour) << RTC_TR_HU_Pos) | (to_bcd(time.minute) << RTC_TR_MNU_Pos) | (to_bcd(time.second) << RTC_TR_SU_Pos);
  RTC->DR = (to_bcd(time.year - YEAR_OFFSET) << RTC_DR_YU_Pos) | (time.weekday << RTC_DR_WDU_Pos) | (to_bcd(time.month) << RTC_DR_MU_Pos) | (to_bcd(time.day) << RTC_DR_DU_Pos);

  // Set options
  register_set(&(RTC->CR), 0U, 0xFCFFFFU);

  // Disable initalization mode
  register_clear_bits(&(RTC->ISR), RTC_ISR_INIT);

  // Wait for synchronization
  while((RTC->ISR & RTC_ISR_RSF) == 0){}

  // Re-enable write protection
  RTC->WPR = 0x00;
  enable_bdomain_protection();
}

timestamp_t rtc_get_time(void){
  timestamp_t result;

  // Wait until the register sync flag is set
  while((RTC->ISR & RTC_ISR_RSF) == 0){}

  // Read time and date registers. Since our HSE > 7*LSE, this should be fine.
  uint32_t time = RTC->TR;
  uint32_t date = RTC->DR;

  // Parse values
  result.year = from_bcd((date & (RTC_DR_YT | RTC_DR_YU)) >> RTC_DR_YU_Pos) + YEAR_OFFSET;
  result.month = from_bcd((date & (RTC_DR_MT | RTC_DR_MU)) >> RTC_DR_MU_Pos);
  result.day = from_bcd((date & (RTC_DR_DT | RTC_DR_DU)) >> RTC_DR_DU_Pos);
  result.weekday = ((date & RTC_DR_WDU) >> RTC_DR_WDU_Pos);
  result.hour = from_bcd((time & (RTC_TR_HT | RTC_TR_HU)) >> RTC_TR_HU_Pos);
  result.minute = from_bcd((time & (RTC_TR_MNT | RTC_TR_MNU)) >> RTC_TR_MNU_Pos);
  result.second = from_bcd((time & (RTC_TR_ST | RTC_TR_SU)) >> RTC_TR_SU_Pos);

  return result;
}
